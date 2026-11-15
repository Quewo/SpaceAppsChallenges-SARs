# SpaceAppsChallenges-SARs

Repo for the NASA Space Apps Challenge: **Dancing with the SARs** — Multi-Temporal SAR & Terrain Classification Pipeline (Case 3: `T > 4`, without `SME2`).

---

## Data Import & Streaming Behavior ([`notebooks/01_daten_suchen_und_laden.ipynb`](notebooks/01_daten_suchen_und_laden.ipynb))

- **Authentication:** Reads `EARTHDATA_TOKEN` from `.env` and authenticates via `earthaccess.login(strategy="environment")`.
- **AOI-Only Remote Slicing (`earthaccess.open`):** SAR granules (`NISAR` `.h5` and `OPERA` Cloud-Optimized GeoTIFFs) are **not** downloaded in full to disk. Instead, they are streamed via HTTP range requests and clipped/sliced to the padded `AOI_BBOX` before reprojection to the target UTM grid. Legacy `datapool.asf.alaska.edu` OPERA links are automatically mapped to their `cumulus.asf.earthdatacloud.nasa.gov/OPERA/` HTTPS endpoints via the granule's `direct` S3 URL so bearer-token auth works uniformly across 2025–2026 archives.
- **Static DEM Cache:** Only `NASADEM_HGT` (`~7.5 MB` `.zip` per $1^\circ \times 1^\circ$ tile) is downloaded and unzipped into `data/raw/<RUN_NAME>/nasadem/` and clipped to the AOI.
- **OPERA + NISAR Temporal Coverage:** `OPERA_L2_RTC-S1_V1` is queried across the full `START_DATE .. END_DATE` interval (not restricted to pre-NISAR dates), providing deep historical Sentinel-1 C-band time series (`2016+`) as well as 2026 dates overlapping NISAR L-band for cross-sensor alignment and gap-filling.
- **Same-Date Mosaicking & Coverage Filtering:** Multiple bursts/frames on the same acquisition date are reprojected to `target_grid` and merged (`mean` for continuous backscatter/coherence, `first` via `combine_first` for `DSWx` classes). Prior to export, any timestep with valid-pixel coverage `< MIN_VALID_PCT` (default `80.0%`) is dropped.

---

## Inputs (Exact Specification)

### 1. Credentials (`.env` in repository root)
| Variable | Type | Required | Description |
| :--- | :--- | :--- | :--- |
| `EARTHDATA_TOKEN` | `str` | Yes | NASA Earthdata Login Bearer token |

### 2. Notebook Configuration (Section 2 of `notebooks/01_daten_suchen_und_laden.ipynb`)
| Parameter | Type | Default | Exact Format / Semantics |
| :--- | :--- | :--- | :--- |
| `RUN_NAME` | `str` | `"oberrhein_test"` | Output subfolder name under `data/raw/` and `data/processed/` |
| `AOI_GEOJSON` | `str \| Path \| None` | `None` | Optional path to vector AOI (reprojected to `EPSG:4326` to derive `AOI_BBOX`); if `None`, `AOI_BBOX` is used directly |
| `AOI_BBOX` | `tuple[float, float, float, float]` | `(8.28, 49.02, 8.58, 49.28)` | `(west, south, east, north)` in `EPSG:4326` (WGS84 decimal degrees) |
| `START_DATE` | `str` | `"2025-01-01"` | Inclusive start date `YYYY-MM-DD` (UTC) |
| `END_DATE` | `str` | `"2026-11-14"` | Inclusive end date `YYYY-MM-DD` (UTC) |
| `MIN_TIMESTEPS` | `int` | `5` | Minimum unique timestamps check (`T > 4`) |
| `MAX_TIMESTEPS_PER_COLLECTION` | `int \| None` | `16` | Maximum evenly spaced unique acquisition dates per temporal collection; `None` selects all available dates |
| `TARGET_RES_M` | `float` | `30.0` | Output pixel spacing in meters on the auto-estimated UTM grid |
| `MIN_VALID_PCT` | `float` | `80.0` | Minimum percentage `[0.0, 100.0]` of finite pixels within the AOI required to keep a timestep (`0.0` disables filtering) |
| `INCLUDE_NISAR_GCOV` | `bool` | `True` | Query `NISAR_L2_GCOV_V1` + `NISAR_L2_GCOV_PROVISIONAL_V1` (deduplicated) |
| `INCLUDE_NISAR_GUNW` | `bool` | `True` | Query `NISAR_L2_GUNW_V1` + `NISAR_L2_GUNW_PROVISIONAL_V1` (deduplicated) |
| `INCLUDE_OPERA_RTC_S1` | `bool` | `True` | Query `OPERA_L2_RTC-S1_V1` |
| `INCLUDE_NASADEM` | `bool` | `True` | Query `NASADEM_HGT` (`version="001"`, static, no temporal filter) |
| `INCLUDE_OPERA_DSWX_S1` | `bool` | `False` | Query `OPERA_L3_DSWX-S1_V1` (`_WTR.tif` / `_BWTR.tif`) |
| `RUN_FETCH` | `bool` | `True` | `True` streams remote data, aligns, filters, and exports NetCDFs; `False` reloads existing `.nc` files from `OUT_DIR` to re-filter/re-export locally |

---

## Output Format (Exact Specification)

All aligned outputs are written to `data/processed/<RUN_NAME>/`.

### 1. Shared Grid & NetCDF-4 Container Convention
Every `.nc` file is CF/GDAL-compatible NetCDF-4 (`engine="h5netcdf"`, `zlib=True`, `complevel=4`) on the **exact same UTM grid**:
- **CRS (`spatial_ref`):** `int64` scalar coordinate with CF/GDAL attributes `crs_wkt`, `spatial_ref`, and `GeoTransform` (auto-estimated UTM zone from `AOI_BBOX`, e.g., `EPSG:32632`).
- **`y` coordinate:** `float64`, pixel centers in meters, strictly descending North-to-South (`step = -TARGET_RES_M`), bounds snapped to multiples of `TARGET_RES_M`.
- **`x` coordinate:** `float64`, pixel centers in meters, strictly ascending West-to-East (`step = +TARGET_RES_M`), bounds snapped to multiples of `TARGET_RES_M`.
- **`time` coordinate *(temporal cubes only)*:** `datetime64[ns]`, sorted ascending acquisition dates (`YYYY-MM-DD`, UTC).
- **Data dtype & missing values:** All data variables are `float32`; missing coverage, masked pixels, and non-positive linear backscatter values ($\gamma^0 \le 0$) are stored as `NaN`.

### 2. Output Files
| File | Dimensions | Data Variables (`float32`) | Units / Range & Derivation |
| :--- | :--- | :--- | :--- |
| `nasadem_static.nc` | `(y, x)` | `elevation_m`, `slope_deg`, `aspect_deg` | Elevation in `m`; slope (`[0, 90]°`) and aspect (`[0, 360)°`) via `np.gradient` at `TARGET_RES_M` |
| `opera_rtc_s1.nc` | `(time, y, x)` | `VV_dB`, `VH_dB` *(or `HH_dB`, `HV_dB`)*, `VH_minus_VV_dB` | Backscatter in `dB` ($10 \log_{10}(\gamma^0)$); `VH_minus_VV_dB = VH_dB - VV_dB` when both exist |
| `nisar_gcov.nc` | `(time, y, x)` | Subset of `HHHH_dB`, `HVHV_dB`, `VHVH_dB`, `VVVV_dB` | L-band diagonal covariance intensity from `/science/LSAR/GCOV/grids/frequencyA` (or `frequencyB`) in `dB` ($10 \log_{10}(\gamma^0)$) |
| `nisar_gunw.nc` | `(time, y, x)` | Subset of `coherence_HH`, `coherence_VV`, `coherence_HV`, `coherence_VH` | Dimensionless interferometric `coherenceMagnitude` in `[0, 1]` from `/science/LSAR/GUNW/grids/frequencyA/unwrappedInterferogram/<pol>/coherenceMagnitude` |
| `opera_dswx_s1.nc` *(optional)* | `(time, y, x)` | `dswx_wtr` | Nearest-neighbor resampled OPERA DSWx-S1 water class value |
| `manifest.csv` | Tabular CSV | Columns: `dataset,date,variables,valid_pct,file` | `dataset`: cube name; `date`: `YYYY-MM-DD` or `"static"`; `variables`: comma-separated variable names; `valid_pct`: `float` in `[0.0, 100.0]` rounded to 2 decimals; `file`: repository-relative POSIX/Windows path to `.nc` |

---
---

## Validated Training Data Acquisition ([`notebooks/02_validated_training_data.ipynb`](notebooks/02_validated_training_data.ipynb))

Builds leakage-free, quality-masked `10 m` training datasets pairing multi-temporal C-band (`OPERA_L2_RTC-S1_V1`) and L-band (`NISAR_L2_GCOV_V1` + optional `NISAR_L2_GUNW_PROVISIONAL_V1`) SAR features (**X**) with cross-sensor validated 6-class surface labels (**y**) and `NASADEM_HGT` topography.

- **Authentication & Storage Footprint:** Authenticates via `EARTHDATA_TOKEN` in `.env` (`earthaccess.login(strategy="environment")`). All remote GeoTIFFs (`OPERA RTC-S1`, `OPERA DSWx-HLS`, `ESA WorldCover`) and HDF5s (`NISAR GCOV`, `NISAR GUNW`) are streamed via AOI-only HTTP range requests and cropped in memory — only compact `NASADEM_HGT` (`v001`) `.zip` tiles are cached locally under `data/raw/<RUN_NAME>_<aoi>/nasadem/`.
- **Strict Science Validation + Provisional `GUNW` Coherence:** Keeps `ALLOW_PROVISIONAL_NISAR = False` so `OPERA_L2_RTC-S1_V1`, `NISAR_L2_GCOV_V1`, `OPERA_L3_DSWX-HLS_V1`, and `NASADEM_HGT` use strictly validated science releases, while enabling `ALLOW_PROVISIONAL_GUNW = True` (`NISAR_L2_GUNW_PROVISIONAL_V1`) as an auxiliary L-band interferometric coherence feature (`coherence_HH` + `has_gunw`) because `V1` GUNW is not yet published.
- **Catalog Pre-Filtering & Temporal Pairing:**
  - Excludes single-polarization NISAR GCOV modes via `EXCLUDED_NISAR_POL_MODES = ("_SHSH_", "_NASV_", "_SVSV_")` in `search_collection`, retaining dual-pol `DHDH` (`HHHH`, `HVHV`) acquisitions, and deduplicates overlapping V1/Provisional granule IDs.
  - Filters `opera_rtc_s1`, `nisar_gcov`, and `nisar_gunw` dates per AOI via `filter_dates_by_aoi_coverage` so the union of CMR `GPolygons` (falling back to `BoundingRectangles`) covers `>= MIN_BBOX_COVERAGE_PCT` (`85.0%`) of the AOI bbox before streaming.
  - Reprocessing-deduplicates `OPERA_L3_DSWX-HLS_V1` (`deduplicate_dswx_reprocessing` keeps the latest production timestamp per tile/acquisition/sensor) and pre-filters to `±DSWX_MAX_TIME_DIFF_DAYS` (`10` days) around selected SAR dates.
  - Extracts `EndingDateTime` (`use_end=True`, secondary interferogram pass date) for `nisar_gunw` and pre-filters to dates covering `>= 85.0%` of the AOI within `±GUNW_MAX_TIME_DIFF_DAYS` (`12` days) of selected `nisar_gcov` dates.
- **Radiometric, Geometry & Quality Masking:**
  - `OPERA_L2_RTC-S1_V1` (`MASK_OPERA_LAYOVER_SHADOW = True`) streams each granule's `*_mask.tif` (`Resampling.nearest`) and masks every non-zero pixel (layover, shadow, invalid).
  - `NISAR_L2_GCOV_V1` enforces `(mask > 0) & (mask < 255)` and `numberOfLooks > 0`.
  - `NISAR_L2_GUNW_PROVISIONAL_V1` decodes `/science/LSAR/GUNW/grids/frequencyA|B/unwrappedInterferogram/mask` in native `uint64` (`low_b = m & 0xFF; (m != 255) & ((low_b % 10) > 0) & (((low_b // 10) % 10) > 0) & (((m >> 8) & 0xFFFF) == 0)`), preserving valid pixels carrying bit-24 ionospheric phase filtering (`16777228`) while excluding fill/invalid subswath and bits `8..23` anomaly flags; coherence is restricted to `[0.0, 1.0]`.
  - Linear SAR backscatter ($\gamma^0 > 0$) is converted to decibels (`to_db`) and clamped to `[DB_MIN, DB_MAX] = [-40.0, 20.0] dB`; per-timestamp `3x3` texture features require all pixels in the `3x3` window across SAR + static bands to be finite (`valid_3x3`). Any acquisition with `valid_3x3 < MIN_VALID_PCT` (`80.0%`) or `0` labeled pixels is excluded (`kept_in_cube = False`).
- **Cross-Sensor Validated Label Fusion (`y`, `uint8`):**
  1. **Static Baseline:** Streams `ESA WorldCover 10m v200 (2021)` public S3 COGs (`ESA_WorldCover_10m_2021_v200_<tile>_Map.tif`), maps codes to project classes `0..5` via `WORLDCOVER_TO_CLASS`, and when `REQUIRE_HOMOGENEOUS_3X3 = True` sets any pixel whose `3x3` neighborhood is non-uniform to `IGNORE_LABEL = 255`.
  2. **Dynamic Water / Wetland Validation:** Pairs each SAR timestamp to the closest `OPERA_L3_DSWX-HLS_V1` acquisition within `±DSWX_MAX_TIME_DIFF_DAYS` (`10` days). With `HIGH_CONFIDENCE_DSWX_ONLY = True`, keeps clear land (`WTR == 0`), high-confidence open water (`WTR == 1` & `CONF == 1`), and high-confidence partial water (`WTR == 2` & `CONF == 3`). Gated to flatter terrain (`slope_deg <= MAX_WATER_SLOPE_DEG = 15.0°`): `WTR == 1` sets `label = 1` (`Water & Flood`); `WTR == 2` over classes `{0, 2, 3, 4}` sets `label = 3` (`Wetlands`); `WTR == 0` over static class `1` sets `255` (stale water guard).
  3. **Custom & Geometry Masks:** Optional `VALIDATED_LABELS_GEOJSON` polygons override fused labels; pixels with invalid slope or `slope_deg > MAX_SLOPE_DEG` (`60.0°`) are set to `255`; and non-`valid_3x3` pixels are masked to `255` in `labels_<sensor>.nc` so `labeled_but_sar_nan == 0` across every cube.
- **Leakage-Free Spatial-Block Split & Two-Stage Stratified Sampling:**
  - Deterministic `3000.0 m` UTM grid (`assign_spatial_block_splits`): each pixel's block index `(bx, by) = floor(coord / BLOCK_SIZE_M)` is hashed via `sha256(f"{RANDOM_SEED}:{aoi_name}:{bx}:{by}")` to `[0, 1)` and assigned to `train` (`0`, `< 0.70`), `val` (`1`, `0.70..0.85`), or `test` (`2`, `>= 0.85`), guaranteeing zero spatial leakage across timestamps.
  - **Stage 1 (Per-Timestamp Pre-Sampling):** Samples up to `MAX_SAMPLES_PER_CLASS_PER_TIMESTAMP` pixels per `(split, class)` on each date using a deterministic timestamp seed.
  - **Stage 2 (AOI-Balanced Global Stratified Export):** Merges all timestamps and water-fills up to `MAX_SAMPLES_PER_CLASS_PER_SPLIT` rows per `(split, class)` evenly across AOIs (`_sample_balanced_by_aoi`).
  - When `MAX_TIMESTEPS_PER_AOI is None` (`"overnight"` profile), `opera_rtc_s1` streams in date batches (`BATCH_SAR_DATES = 25`), caching per-AOI `.parquet`, `__manifest.json`, and `.done` markers under `data/training/<RUN_NAME>/_aoi_cache/` for safe resume and bounded memory, while saving `MAX_CUBE_TIMESTEPS_PER_AOI = 16` evenly spaced dates into `opera_rtc_s1.nc` and `labels_opera_rtc_s1.nc`.

---

## Inputs & Parameters — `notebooks/02_validated_training_data.ipynb` (Exact Specification)

### 1. Run Profiles (`PROFILE_CONFIGS`, controlled by `RUN_PROFILE`)
| Parameter / Profile | `"test"` | `"overnight"` | Semantics |
| :--- | :--- | :--- | :--- |
| `run_name` (`RUN_NAME`) | `"validated_train_test_v1"` | `"validated_train_overnight_v1"` | Output subfolder under `data/training/` |
| `aoi_keys` | 3 AOIs: `oberrhein`, `alps_inntal`, `danube_delta` | All 9 AOIs in `ALL_TRAINING_AOIS` | Subset of multi-biome AOIs to process |
| `max_timesteps_per_aoi` (`MAX_TIMESTEPS_PER_AOI`) | `5` *(~5.5 months separation via `select_dates_evenly`)* | `None` *(no separation; all coverage-valid dates)* | Cap on unique temporal SAR dates streamed per AOI for tabular sampling |
| `max_cube_timesteps_per_aoi` (`MAX_CUBE_TIMESTEPS_PER_AOI`) | `None` *(saves all valid sampled dates)* | `16` *(evenly spaced OPERA RTC-S1 dates in `.nc`)* | Cap on `opera_rtc_s1.nc` timestamps when streaming `BATCH_SAR_DATES = 25`; `nisar_gcov.nc` & `nisar_gunw.nc` keep all valid dates |
| `max_samples_per_class_per_timestamp` | `{"train": 1000, "val": 300, "test": 300}` | `{"train": 1500, "val": 500, "test": 500}` | Max pixels sampled per `(split, class)` per SAR date before concatenating |
| `max_samples_per_class_per_split` | `{"train": 4000, "val": 1000, "test": 1000}` | `{"train": 250000, "val": 50000, "test": 50000}` | Global balanced export cap per `(split, class)` across all AOIs |
| `save_intermediate_cubes` (`SAVE_INTERMEDIATE_CUBES`) | `True` | `True` | Export full `10 m` UTM `.nc` cubes per AOI alongside tabular datasets |

### 2. Multi-Biome AOI Catalog (`ALL_TRAINING_AOIS`)
All bounding boxes are specified as `(west, south, east, north)` in `EPSG:4326` (~`15–20 km` footprints aligned to single `NASADEM` / `WorldCover` tiles).

| AOI Key | `EPSG:4326` Bounding Box `(w, s, e, n)` | Auto UTM CRS | Biome & Target Land-Cover Characteristics |
| :--- | :--- | :--- | :--- |
| `oberrhein` | `(8.33, 49.05, 8.51, 49.21)` | `EPSG:32632` | Temperate valley (DE): forest, cropland, Rhine river, urban |
| `alps_inntal` | `(11.30, 47.22, 11.48, 47.38)` | `EPSG:32632` | Alpine (AT): steep bare rock, mountain forest, valley urban |
| `danube_delta` | `(29.10, 45.02, 29.28, 45.18)` | `EPSG:32635` | Delta wetlands (RO): reedbeds, channels, seasonal flood |
| `po_valley` | `(11.48, 44.80, 11.66, 44.96)` | `EPSG:32632` | Floodplain (IT): intensive cropland, Ferrara urban, river |
| `mekong_can_tho` | `(105.70, 10.02, 105.86, 10.18)` | `EPSG:32648` | Monsoon delta (VN): flooded rice paddies, Hau river, urban |
| `atchafalaya_la` | `(-91.50, 29.38, -91.32, 29.54)` | `EPSG:32615` | Subtropical basin (US): forested swamp, coastal marsh, water |
| `nile_cairo` | `(31.15, 30.02, 31.33, 30.18)` | `EPSG:32636` | Hyper-arid (EG): flat desert bare ground, mega-city, Nile, crops |
| `pantanal_br` | `(-58.68, -27.48, -58.50, -27.32)` | `EPSG:32721` | Southern Hemisphere floodplain (AR/PY, verified dual-pol `DHDH`): seasonal wetland, savanna, gallery forest |
| `phoenix_az` | `(-112.22, 33.36, -112.04, 33.52)` | `EPSG:32612` | Arid Southwest (US): desert scrub, sprawl urban, irrigated ag |

### 3. Pipeline & Validation Parameters (Cell 4 & Cell 6 / Cell 12)
| Parameter | Type | Default | Exact Format / Semantics |
| :--- | :--- | :--- | :--- |
| `RUN_PROFILE` | `str` | `"overnight"` | Active profile key in `PROFILE_CONFIGS`: `"test"` or `"overnight"` |
| `START_DATE` | `str` | `"2025-01-01"` | Inclusive UTC start date `YYYY-MM-DD` for SAR queries (DSWx padded `-10d`) |
| `END_DATE` | `str` | `"2026-11-14"` | Inclusive UTC end date `YYYY-MM-DD` for SAR queries (DSWx padded `+10d`) |
| `TARGET_RES_M` | `float` | `10.0` | Shared UTM output pixel spacing in meters (`10 m x 10 m`) |
| `MIN_VALID_PCT` | `float` | `80.0` | Minimum percentage `[0.0, 100.0]` of `valid_3x3` pixels required to retain a SAR timestamp |
| `MIN_BBOX_COVERAGE_PCT` | `float` | `85.0` | Minimum percentage `[0.0, 100.0]` of AOI bbox covered by combined CMR `GPolygons` per SAR date (`<= 0` disables) |
| `DB_MIN` | `float` | `-40.0` | Physical minimum backscatter clamp in `dB` (`to_db` lower bound) |
| `DB_MAX` | `float` | `20.0` | Physical maximum backscatter clamp in `dB` (`to_db` upper bound) |
| `ALLOW_PROVISIONAL_NISAR` | `bool` | `False` | `False` queries only `NISAR_L2_GCOV_V1`; `True` adds `NISAR_L2_GCOV_PROVISIONAL_V1` |
| `ALLOW_PROVISIONAL_GUNW` | `bool` | `True` | `True` adds `NISAR_L2_GUNW_PROVISIONAL_V1` alongside `NISAR_L2_GUNW_V1` |
| `EXCLUDED_NISAR_POL_MODES` | `tuple[str, ...]` | `("_SHSH_", "_NASV_", "_SVSV_")` | Case-insensitive `native-id` tokens skipped during `nisar_gcov` catalog search |
| `MASK_OPERA_LAYOVER_SHADOW` | `bool` | `True` | Streams OPERA RTC-S1 `*_mask.tif` (`0 = valid`; non-zero masked to `NaN`) |
| `MAX_SLOPE_DEG` | `float \| None` | `60.0` | Maximum terrain slope in degrees; pixels with `slope_deg > MAX_SLOPE_DEG` or `NaN` slope receive `IGNORE_LABEL` (`None` disables upper cap) |
| `MAX_WATER_SLOPE_DEG` | `float` | `15.0` | Maximum slope in degrees allowed for `DSWx-HLS` dynamic water/wetland overrides |
| `REQUIRE_HOMOGENEOUS_3X3` | `bool` | `True` | Requires identical WorldCover class across all 9 pixels in `3x3` window; otherwise sets static label to `IGNORE_LABEL` |
| `INCLUDE_OPERA_RTC_S1` | `bool` | `True` | Enable `OPERA_L2_RTC-S1_V1` streaming & export |
| `INCLUDE_NISAR_GCOV` | `bool` | `True` | Enable `NISAR_L2_GCOV_V1` (+ provisional if `ALLOW_PROVISIONAL_NISAR=True`) streaming & export |
| `INCLUDE_NISAR_GUNW` | `bool` | `True` | Enable `NISAR_L2_GUNW` coherence extraction (`coherence_HH`, `has_gunw`) |
| `INCLUDE_NASADEM` | `bool` | `True` | Enable `NASADEM_HGT` (`version="001"`) static DEM & terrain derivative computation |
| `INCLUDE_ESA_WORLDCOVER` | `bool` | `True` | Enable `ESA WorldCover 10m v200 (2021)` S3 COG static label baseline |
| `INCLUDE_DSWX_HLS` | `bool` | `True` | Enable `OPERA_L3_DSWX-HLS_V1` optical dynamic water/wetland validation |
| `HIGH_CONFIDENCE_DSWX_ONLY` | `bool` | `True` | Requires `_CONF.tif` (`1` for open water `WTR==1`, `3` for partial water `WTR==2`; keeps clear `WTR==0`) |
| `DSWX_MAX_TIME_DIFF_DAYS` | `int` | `10` | Maximum `|t_sar - t_dswx|` days for dynamic optical water matching |
| `GUNW_MAX_TIME_DIFF_DAYS` | `int` | `12` | Maximum `|t_gcov - t_gunw|` days to pair `nisar_gunw` `coherence_HH` to `nisar_gcov` |
| `VALIDATED_LABELS_GEOJSON` | `str \| Path \| None` | `None` | Optional vector polygon file with integer `class_id` (`0..5`) overriding all other labels |
| `BLOCK_SIZE_M` | `float` | `3000.0` | Square spatial split block side length in UTM meters |
| `SPLIT_RATIOS` | `tuple[float, float, float]` | `(0.70, 0.15, 0.15)` | Deterministic `(train, val, test)` target block proportions |
| `SPLIT_ID_TO_NAME` | `dict[int, str]` | `{0: "train", 1: "val", 2: "test"}` | Integer encoding stored in `split` column |
| `RANDOM_SEED` | `int` | `42` | RNG & SHA-256 hash seed for deterministic spatial splits and sampling |
| `RUN_FETCH` | `bool` | `True` | `True` streams remote granules & updates `_aoi_cache`; `False` reloads existing `.nc` cubes locally |
| `IGNORE_LABEL` | `int` | `255` | `uint8` ignore/mask index across labels (`labels_*.nc`) and WorldCover |
| `BATCH_SAR_DATES` | `int \| None` | `None` (`test`) / `25` (`overnight`) | Derived automatically: `None if MAX_TIMESTEPS_PER_AOI is not None else 25`; bounds RAM when streaming all `OPERA` dates |

### 4. Target Class Schema (`CLASS_DEFINITIONS`) & WorldCover Mapping (`WORLDCOVER_TO_CLASS`)
| Project `label` (`uint8`) | Class Name | Hex Color | Mapped `ESA WorldCover v200` Codes & Dynamic Overrides |
| :--- | :--- | :--- | :--- |
| `0` | `"Bare Ground"` | `#64748B` | WorldCover `20` (Shrubland), `30` (Grassland), `60` (Bare/sparse vegetation), `70` (Snow/ice), `100` (Moss/lichen) |
| `1` | `"Water & Flood"` | `#0EA5E9` | WorldCover `80` (Permanent water bodies), or dynamic `DSWx-HLS` `WTR == 1` where `slope_deg <= 15.0°` |
| `2` | `"Tree Lines & Forest"` | `#22C55E` | WorldCover `10` (Tree cover) |
| `3` | `"Wetlands"` | `#14B8A6` | WorldCover `90` (Herbaceous wetland), `95` (Mangroves), or `DSWx-HLS` `WTR == 2` over classes `{0, 2, 3, 4}` where `slope_deg <= 15.0°` |
| `4` | `"Cropland"` | `#EAB308` | WorldCover `40` (Cropland) |
| `5` | `"Urban"` | `#F97316` | WorldCover `50` (Built-up) |
| `255` (`IGNORE_LABEL`) | Ignored / Masked | — | Non-homogeneous `3x3` edge, `slope_deg > 60°` / `NaN`, stale water (`WTR == 0` over static `1`), or `~valid_3x3` |

---

## Output Format — `notebooks/02_validated_training_data.ipynb` (Exact Specification)

All outputs are written under `data/training/<RUN_NAME>/`:

```text
data/training/<RUN_NAME>/
├── dataset_metadata.json
├── validation_manifest.csv
├── training_samples_opera_rtc_s1.parquet
├── training_samples_opera_rtc_s1.npz
├── training_samples_nisar_gcov.parquet
├── training_samples_nisar_gcov.npz
├── _aoi_cache/                              # Only created when MAX_TIMESTEPS_PER_AOI is None ("overnight")
│   ├── <aoi>__opera_rtc_s1.parquet
│   ├── <aoi>__nisar_gcov.parquet
│   ├── <aoi>__manifest.json
│   ├── <aoi>.done
│   └── _qa_sensor.nc, _qa_labels.nc, _qa_split.npy, _qa_meta.json
└── <aoi>/                                   # One folder per AOI when SAVE_INTERMEDIATE_CUBES = True
    ├── nasadem_static.nc
    ├── worldcover_static.nc
    ├── opera_dswx_hls.nc
    ├── opera_rtc_s1.nc
    ├── labels_opera_rtc_s1.nc
    ├── nisar_gcov.nc
    ├── labels_nisar_gcov.nc
    └── nisar_gunw.nc
```

### 1. Per-AOI NetCDF-4 Cubes (`data/training/<RUN_NAME>/<aoi>/*.nc`)
Every `.nc` file is written by `_write_cf_netcdf` (`engine="h5netcdf"`, `zlib=True`, `complevel=4`) on the **exact same `10.0 m` UTM grid** per AOI:
- **CF/GDAL CRS (`spatial_ref`):** `int64` scalar coordinate generated via `.rio.write_crs(crs).rio.write_coordinate_system()`, with every data variable encoded with `grid_mapping="spatial_ref"` and `coordinates="spatial_ref"` so `xr.open_dataset(path, engine="h5netcdf").rio.crs` returns the exact UTM EPSG directly.
- **Spatial & Temporal Coordinates:** `y` (`float64`, descending N→S, step `-10.0 m`), `x` (`float64`, ascending W→E, step `+10.0 m`), and `time` (`datetime64[ns]`, ascending `YYYY-MM-DD`, UTC, on temporal cubes).

| Cube File | Dimensions | Data Variables & `dtype` | Units / Valid Range & Exact Derivation |
| :--- | :--- | :--- | :--- |
| `nasadem_static.nc` | `(y, x)` | `elevation_m`, `slope_deg`, `aspect_deg`, `aspect_sin`, `aspect_cos` (`float32`) | Elevation in `m`; slope (`[0, 90]°`) and aspect (`[0, 360)°`, plus $\sin/\cos$) via `np.gradient` at `10.0 m` |
| `worldcover_static.nc` | `(y, x)` | `worldcover_class`, `is_homogeneous_3x3` (`uint8`) | `worldcover_class` mapped in `{0..5, 255}`; `is_homogeneous_3x3` in `{0, 1}` (`1` iff `3x3` min == max and `!= 255`) |
| `opera_dswx_hls.nc` | `(time, y, x)` | `dswx_hls_wtr` (`uint8`) | `0` = clear non-water, `1` = open water (`CONF==1`), `2` = partial water (`CONF==3`), `255` = invalid/cloudy/low-conf |
| `opera_rtc_s1.nc` | `(time, y, x)` | `VV_dB`, `VH_dB` *(or `HH_dB`, `HV_dB`)*, `VH_minus_VV_dB` (`float32`) | Mask-filtered (`*_mask.tif == 0`), `to_db` clamped `[-40.0, 20.0] dB`; `VH_minus_VV_dB = VH_dB - VV_dB`; missing/masked = `NaN`; only dates with `valid_3x3 >= 80.0%` |
| `labels_opera_rtc_s1.nc` | `(time, y, x)` | `label` (`uint8`) | Fused & masked labels in `{0..5, 255}`; `time` matches `opera_rtc_s1.nc` 1:1; guaranteed `255` wherever `~valid_3x3` |
| `nisar_gcov.nc` | `(time, y, x)` | Subset of `HHHH_dB`, `HVHV_dB`, `VVVV_dB`, `VHVH_dB`, `HH_minus_HV_dB` (`float32`) | Mask/looks-filtered (`0 < mask < 255`, `numberOfLooks > 0`), `to_db` clamped `[-40.0, 20.0] dB`; `HH_minus_HV_dB = HHHH_dB - HVHV_dB`; only dates with `valid_3x3 >= 80.0%` |
| `labels_nisar_gcov.nc` | `(time, y, x)` | `label` (`uint8`) | Fused & masked labels in `{0..5, 255}`; `time` matches `nisar_gcov.nc` 1:1; guaranteed `255` wherever `~valid_3x3` |
| `nisar_gunw.nc` | `(time, y, x)` | Subset of `coherence_HH`, `coherence_VV`, `coherence_HV`, `coherence_VH` (`float32`) | Masked via `uint64` `unwrappedInterferogram/mask`, clamped `[0.0, 1.0]`, timestamps from `EndingDateTime` within `±12d` of `nisar_gcov` |

### 2. Tabular Parquet Datasets (`training_samples_<sensor>.parquet`)
Each row represents one finite, `3x3`-valid, labeled `10 m x 10 m` pixel sample (`0` NaNs across all feature and metadata columns). Primary SAR bands (`*_dB` excluding differences) expand into raw value, `3x3` local mean (`*_mean3` via `uniform_filter`), and `3x3` local std (`*_std3` via $\sqrt{\max(E[x^2] - E[x]^2, 0)}$).

**A. `training_samples_opera_rtc_s1.parquet` (18 columns: 11 `float32` features + 7 metadata):**
1. **Features (`11`, exact column order):** `VV_dB`, `VV_dB_mean3`, `VV_dB_std3`, `VH_dB`, `VH_dB_mean3`, `VH_dB_std3`, `VH_minus_VV_dB`, `elevation_m`, `slope_deg`, `aspect_sin`, `aspect_cos` *(all `float32`)*.
2. **Metadata (`7`, exact column order):** `label` (`int64`, `0..5`), `split` (`uint8`, `0=train, 1=val, 2=test`), `block_id` (`str`, `"<aoi>_<bx>_<by>"`), `aoi` (`str`), `date` (`str`, `YYYY-MM-DD`), `x_utm` (`float64`), `y_utm` (`float64`).

**B. `training_samples_nisar_gcov.parquet` (20 columns: 13 `float32` features + 7 metadata):**
1. **Features (`13`, exact column order):** `HHHH_dB`, `HHHH_dB_mean3`, `HHHH_dB_std3`, `HVHV_dB`, `HVHV_dB_mean3`, `HVHV_dB_std3`, `HH_minus_HV_dB`, `coherence_HH`, `has_gunw`, `elevation_m`, `slope_deg`, `aspect_sin`, `aspect_cos` *(all `float32`)*.
   - **GUNW Coherence Rules:** `coherence_HH` takes the nearest `nisar_gunw.nc` slice within `<= 12` days when finite in `[0.0, 1.0]` inside `valid_3x3` (`has_gunw = 1.0`); otherwise `coherence_HH = 0.0` and `has_gunw = 0.0` (so unmatched dates/pixels never drop GCOV samples).
2. **Metadata (`7`, identical schema to `opera_rtc_s1`):** `label` (`int64`), `split` (`uint8`), `block_id` (`str`), `aoi` (`str`), `date` (`str`), `x_utm` (`float64`), `y_utm` (`float64`).

### 3. Compressed NumPy Archives (`training_samples_<sensor>.npz`)
Ready-to-train NumPy arrays matching the exact row ordering and split partitioning of the corresponding `.parquet` file:
| Archive Key | `dtype` | Shape | Contents |
| :--- | :--- | :--- | :--- |
| `X_train` | `float32` | `(N_train, F)` | Training feature matrix (`F = 11` for `opera_rtc_s1`, `F = 13` for `nisar_gcov`) |
| `y_train` | `int64` | `(N_train,)` | Training class IDs in `{0..5}` |
| `X_val` | `float32` | `(N_val, F)` | Validation feature matrix (disjoint `3000 m` spatial blocks) |
| `y_val` | `int64` | `(N_val,)` | Validation class IDs in `{0..5}` |
| `X_test` | `float32` | `(N_test, F)` | Test feature matrix (disjoint `3000 m` spatial blocks) |
| `y_test` | `int64` | `(N_test,)` | Test class IDs in `{0..5}` |
| `feature_names` | `str` (`<U*`) | `(F,)` | Ordered list matching feature columns of `training_samples_<sensor>.parquet` |
| `class_ids` | `int64` | `(6,)` | `[0, 1, 2, 3, 4, 5]` |

### 4. Manifest & Metadata (`validation_manifest.csv` & `dataset_metadata.json`)
- **`validation_manifest.csv`:** Per-(AOI, sensor, date) audit log with columns:
  - `aoi` (`str`), `sensor` (`"opera_rtc_s1" | "nisar_gcov"`), `date` (`YYYY-MM-DD`), `dswx_matched_date` (`YYYY-MM-DD` or `"none"`), `valid_3x3_pct` (`float`, rounded to 2 decimals), `labeled_pixels` (`int`), `kept_in_cube` (`bool`, `True` iff retained in `<sensor>.nc` and `labels_<sensor>.nc`).
- **`dataset_metadata.json`:** Records full reproducibility config (`run_profile`, `run_name`, `training_aois`, caps/thresholds, `allow_provisional_nisar`, `allow_provisional_gunw`, `db_clamp_range`, `classes`) plus `exports.<sensor>` containing exact `feature_cols`, per-split class `counts`, per-split `aoi_counts`, and `total_rows`.
