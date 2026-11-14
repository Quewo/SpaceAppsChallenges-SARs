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
