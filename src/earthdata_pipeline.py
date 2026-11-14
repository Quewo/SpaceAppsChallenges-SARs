"""NASA Earthdata AOI-only streaming, UTM alignment, and NetCDF ingestion pipeline.

Adapted directly from `notebooks/01_daten_suchen_und_laden.ipynb` so the web
application can both run real NASA/ASF queries for any dragged observation square
at selectable resolution and import locally exported `data/processed/<run_name>/`
NetCDF-4 cubes cleanly into the map & neural-network database.
"""
from __future__ import annotations

import logging
import os
import zipfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ENV_PATH = REPO_ROOT / ".env"
DEFAULT_RAW_DIR = REPO_ROOT / "data" / "raw"
DEFAULT_PROCESSED_DIR = REPO_ROOT / "data" / "processed"

OPERA_ARCHIVE_START = "2016-01-01"
NISAR_ARCHIVE_START = "2025-01-01"

NETCDF_FILENAMES = (
    "nasadem_static.nc",
    "opera_rtc_s1.nc",
    "nisar_gcov.nc",
    "nisar_gunw.nc",
    "opera_dswx_s1.nc",
)


@dataclass(frozen=True)
class EarthdataFetchConfig:
    """Configuration mirroring Section 2 of `notebooks/01_daten_suchen_und_laden.ipynb`."""

    run_name: str
    bbox_wgs84: Tuple[float, float, float, float]  # (west, south, east, north) EPSG:4326
    start_date: str
    end_date: str
    target_res_m: float = 30.0
    min_valid_pct: float = 80.0
    max_timesteps_per_collection: Optional[int] = 16
    min_timesteps: int = 5
    include_nisar_gcov: bool = True
    include_nisar_gunw: bool = True
    include_opera_rtc_s1: bool = True
    include_nasadem: bool = True
    include_opera_dswx_s1: bool = False


@dataclass(frozen=True)
class ProcessedRunInfo:
    """Summary metadata for a locally available processed NASA run directory."""

    run_name: str
    run_dir: Path
    south: float
    west: float
    north: float
    east: float
    resolution_m: float
    datasets: Tuple[str, ...]
    dates: Tuple[str, ...]
    has_manifest: bool


def compute_terrain_derivatives(elev_m: np.ndarray, res_m: float) -> Tuple[np.ndarray, np.ndarray]:
    """Compute `(slope_deg, aspect_deg)` from an elevation grid matching the teammate notebook."""
    arr = np.asarray(elev_m, dtype=np.float32)
    step = max(float(res_m), 1.0)
    dy_dx = np.gradient(arr, step, step)
    dz_dy, dz_dx = -dy_dx[0], dy_dx[1]
    slope_deg = np.degrees(np.arctan(np.hypot(dz_dx, dz_dy))).astype(np.float32)
    aspect_deg = (np.degrees(np.arctan2(dz_dy, -dz_dx)) % 360.0).astype(np.float32)
    invalid = ~np.isfinite(arr)
    if np.any(invalid):
        slope_deg[invalid] = np.nan
        aspect_deg[invalid] = np.nan
    return slope_deg, aspect_deg


def load_dotenv_into_environ(env_path: Path = DEFAULT_ENV_PATH) -> None:
    """Load key=value pairs from `.env` into `os.environ` without overwriting existing vars."""
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def has_earthdata_token(env_path: Path = DEFAULT_ENV_PATH) -> bool:
    """Check whether `EARTHDATA_TOKEN` is configured in process env or `.env`."""
    if os.environ.get("EARTHDATA_TOKEN"):
        return True
    if not env_path.exists():
        return False
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            if k.strip() == "EARTHDATA_TOKEN" and v.strip().strip("\"'"):
                return True
    return False


def ensure_earthdata_auth(token: Optional[str] = None, env_path: Path = DEFAULT_ENV_PATH):
    """Authenticate via `earthaccess` using environment/bearer token."""
    import earthaccess

    load_dotenv_into_environ(env_path)
    if token and token.strip():
        os.environ["EARTHDATA_TOKEN"] = token.strip()
    if not os.environ.get("EARTHDATA_TOKEN"):
        raise ValueError(
            "EARTHDATA_TOKEN not found. Add it to .env or enter your NASA Earthdata Bearer Token."
        )
    auth = earthaccess.login(strategy="environment")
    if not getattr(auth, "authenticated", False):
        raise RuntimeError("Earthdata login failed. Verify that EARTHDATA_TOKEN is valid.")
    return auth


def _extract_granule_date(granule: Any) -> Optional[str]:
    umm = granule.get("umm", {})
    te = umm.get("TemporalExtent", {})
    rdt = te.get("RangeDateTime") or {}
    sdt = te.get("SingleDateTime")
    raw = rdt.get("BeginningDateTime") or sdt
    return str(raw)[:10] if raw else None


def _select_dates_evenly(df: pd.DataFrame, max_dates: Optional[int]) -> pd.DataFrame:
    if df.empty or max_dates is None:
        return df
    unique_dates = sorted(d for d in df["date"].dropna().unique())
    if len(unique_dates) <= max_dates:
        return df
    idx = np.linspace(0, len(unique_dates) - 1, max_dates, dtype=int)
    chosen = {unique_dates[i] for i in idx}
    return df[df["date"].isin(chosen)].copy()


def _to_db(da):
    valid = da.where(da > 0)
    return (10.0 * np.log10(valid)).astype("float32")


def _mosaic_aligned(parts: list, mode: str = "mean"):
    import xarray as xr

    if len(parts) == 1:
        return parts[0].astype("float32")
    if mode == "first":
        out = parts[0]
        for nxt in parts[1:]:
            out = out.combine_first(nxt)
        return out.astype("float32")
    stacked = xr.concat(parts, dim="_mosaic")
    return stacked.mean(dim="_mosaic", skipna=True).astype("float32")


def _build_target_grid(aoi_wgs84, target_crs, res_m: float):
    import xarray as xr

    aoi_utm = aoi_wgs84.to_crs(target_crs)
    minx, miny, maxx, maxy = aoi_utm.total_bounds
    minx = np.floor(minx / res_m) * res_m
    miny = np.floor(miny / res_m) * res_m
    maxx = np.ceil(maxx / res_m) * res_m
    maxy = np.ceil(maxy / res_m) * res_m

    xs = np.arange(minx + res_m / 2.0, maxx, res_m)
    ys = np.arange(maxy - res_m / 2.0, miny, -res_m)

    grid = xr.DataArray(
        np.zeros((len(ys), len(xs)), dtype=np.float32),
        coords={"y": ys, "x": xs},
        dims=("y", "x"),
    )
    grid = grid.rio.write_crs(target_crs).rio.set_spatial_dims(x_dim="x", y_dim="y")
    return grid


def _align_to_target(da, target_grid, resampling):
    if da.rio.crs is None:
        raise ValueError("DataArray is missing CRS metadata.")
    da = da.rio.set_spatial_dims(x_dim="x", y_dim="y")
    return da.rio.reproject_match(target_grid, resampling=resampling)


def _filter_granule_urls(granules: list, suffixes: Tuple[str, ...]) -> List[Tuple[Optional[str], str]]:
    items: List[Tuple[Optional[str], str]] = []
    for g in granules:
        dt = _extract_granule_date(g)
        direct_by_name = {u.rsplit("/", 1)[-1]: u for u in g.data_links(access="direct")}
        for url in g.data_links(access="external"):
            if not url.endswith(suffixes):
                continue
            if "datapool.asf.alaska.edu" in url:
                fname = url.rsplit("/", 1)[-1]
                s3_url = direct_by_name.get(fname, "")
                if s3_url.startswith("s3://asf-cumulus-prod-opera-products/"):
                    url = s3_url.replace(
                        "s3://asf-cumulus-prod-opera-products/",
                        "https://cumulus.asf.earthdatacloud.nasa.gov/OPERA/",
                        1,
                    )
            items.append((dt, url))
    return items


def _clip_remote_cog(file_obj, aoi_wgs84, target_grid, res_m: float, resampling):
    import rioxarray
    from rioxarray.exceptions import NoDataInBounds

    da = rioxarray.open_rasterio(file_obj, masked=True).squeeze("band", drop=True)
    if da.rio.crs is None:
        return None
    minx, miny, maxx, maxy = aoi_wgs84.to_crs(da.rio.crs).total_bounds
    pad = res_m * 4.0
    try:
        clipped = da.rio.clip_box(minx - pad, miny - pad, maxx + pad, maxy + pad)
    except NoDataInBounds:
        return None
    if clipped.size == 0:
        return None
    return _align_to_target(clipped, target_grid, resampling=resampling)


def _read_nisar_epsg(group) -> int:
    proj = group["projection"]
    epsg_attr = proj.attrs.get("epsg_code")
    if epsg_attr is not None:
        return int(epsg_attr)
    return int(proj[()])


def _slice_h5_datasets(grp, dataset_paths: List[str], aoi_wgs84, target_grid, res_m: float, resampling):
    import xarray as xr

    epsg = _read_nisar_epsg(grp)
    xs = grp["xCoordinates"][:]
    ys = grp["yCoordinates"][:]

    minx, miny, maxx, maxy = aoi_wgs84.to_crs(epsg=epsg).total_bounds
    pad = res_m * 4.0
    xi = np.where((xs >= minx - pad) & (xs <= maxx + pad))[0]
    yi = np.where((ys >= miny - pad) & (ys <= maxy + pad))[0]
    if len(xi) == 0 or len(yi) == 0:
        return {}

    x0, x1 = int(xi[0]), int(xi[-1]) + 1
    y0, y1 = int(yi[0]), int(yi[-1]) + 1
    sub_x = xs[x0:x1].astype("float64")
    sub_y = ys[y0:y1].astype("float64")

    out = {}
    for dpath in dataset_paths:
        if dpath not in grp:
            continue
        arr = grp[dpath][y0:y1, x0:x1].astype("float32")
        da = xr.DataArray(arr, coords={"y": sub_y, "x": sub_x}, dims=("y", "x"))
        da = da.rio.write_crs(f"EPSG:{epsg}").rio.set_spatial_dims(x_dim="x", y_dim="y")
        out[dpath] = _align_to_target(da, target_grid, resampling=resampling)
    return out


def fetch_and_export_nasa_run(
    config: EarthdataFetchConfig,
    processed_root: Path = DEFAULT_PROCESSED_DIR,
    raw_root: Path = DEFAULT_RAW_DIR,
    token: Optional[str] = None,
    progress_cb: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    """
    Execute AOI-only remote slicing against NASA Earthdata (`earthaccess`), align all
    collections on a shared UTM grid at `config.target_res_m`, export compressed NetCDF-4
    files + `manifest.csv` under `data/processed/<run_name>/`, and return export summary.
    """
    import earthaccess
    import geopandas as gpd
    import h5py
    import rioxarray
    import xarray as xr
    from rasterio.enums import Resampling
    from shapely.geometry import box

    def _log(msg: str) -> None:
        logger.info(msg)
        if progress_cb is not None:
            progress_cb(msg)

    ensure_earthdata_auth(token=token)

    aoi_gdf = gpd.GeoDataFrame(geometry=[box(*config.bbox_wgs84)], crs="EPSG:4326")
    target_utm_crs = aoi_gdf.estimate_utm_crs()
    target_grid = _build_target_grid(aoi_gdf, target_utm_crs, float(config.target_res_m))

    out_dir = processed_root / config.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = raw_root / config.run_name / "nasadem"
    raw_dir.mkdir(parents=True, exist_ok=True)

    _log(
        f"Searching NASA Earthdata catalogs for {config.bbox_wgs84} "
        f"({config.start_date} .. {config.end_date}) @ {config.target_res_m:.0f}m"
    )

    search_plan = []
    if config.include_nisar_gcov:
        search_plan.append(
            ("nisar_gcov", ["NISAR_L2_GCOV_V1", "NISAR_L2_GCOV_PROVISIONAL_V1"], (config.start_date, config.end_date), None)
        )
    if config.include_nisar_gunw:
        search_plan.append(
            ("nisar_gunw", ["NISAR_L2_GUNW_V1", "NISAR_L2_GUNW_PROVISIONAL_V1"], (config.start_date, config.end_date), None)
        )
    if config.include_opera_rtc_s1:
        search_plan.append(
            ("opera_rtc_s1", ["OPERA_L2_RTC-S1_V1"], (config.start_date, config.end_date), None)
        )
    if config.include_nasadem:
        search_plan.append(("nasadem", ["NASADEM_HGT"], None, "001"))
    if config.include_opera_dswx_s1:
        search_plan.append(
            ("opera_dswx_s1", ["OPERA_L3_DSWX-S1_V1"], (config.start_date, config.end_date), None)
        )

    all_records: List[Dict[str, Any]] = []
    for ds_name, short_names, temp_range, ver in search_plan:
        seen_ids = set()
        for sn in short_names:
            kwargs: Dict[str, Any] = {"short_name": sn, "bounding_box": config.bbox_wgs84}
            if temp_range is not None:
                kwargs["temporal"] = temp_range
            if ver is not None:
                kwargs["version"] = ver
            for g in earthaccess.search_data(**kwargs):
                native_id = g.get("meta", {}).get("native-id", "")
                dedup_key = native_id.replace("_PROVISIONAL_", "_")
                if dedup_key in seen_ids:
                    continue
                seen_ids.add(dedup_key)
                all_records.append(
                    {
                        "dataset": ds_name,
                        "short_name": sn,
                        "granule_id": native_id,
                        "date": _extract_granule_date(g),
                        "granule": g,
                    }
                )

    catalog_df = pd.DataFrame(all_records)
    if catalog_df.empty:
        raise RuntimeError("No granules found for the chosen AOI and date range.")

    selected_parts = []
    for ds_name, grp in catalog_df.groupby("dataset"):
        if ds_name == "nasadem":
            selected_parts.append(grp)
        else:
            selected_parts.append(_select_dates_evenly(grp, config.max_timesteps_per_collection))
    selected_df = pd.concat(selected_parts, ignore_index=True)

    datasets_out: Dict[str, Any] = {}

    # 1. NASADEM static
    nasadem_df = selected_df[selected_df["dataset"] == "nasadem"]
    if config.include_nasadem and not nasadem_df.empty:
        _log("Downloading & mosaicking NASADEM elevation tiles...")
        paths = earthaccess.download(nasadem_df["granule"].tolist(), local_path=str(raw_dir))
        hgt_files: List[Path] = []
        for p in paths:
            pth = Path(p)
            if pth.suffix.lower() == ".zip":
                with zipfile.ZipFile(pth, "r") as zf:
                    zf.extractall(raw_dir)
                    hgt_files.extend(raw_dir / m for m in zf.namelist() if m.lower().endswith(".hgt"))
            elif pth.suffix.lower() == ".hgt":
                hgt_files.append(pth)
        parts = []
        minx, miny, maxx, maxy = aoi_gdf.total_bounds
        for pth in hgt_files:
            da = rioxarray.open_rasterio(pth, masked=True).squeeze("band", drop=True)
            if da.rio.crs is None:
                da = da.rio.write_crs("EPSG:4326")
            da = da.rio.clip_box(minx - 0.01, miny - 0.01, maxx + 0.01, maxy + 0.01)
            parts.append(_align_to_target(da, target_grid, resampling=Resampling.bilinear))
        if parts:
            elev = _mosaic_aligned(parts, mode="mean")
            slope_deg, aspect_deg = compute_terrain_derivatives(elev.values, float(config.target_res_m))
            datasets_out["nasadem_static"] = xr.Dataset(
                {
                    "elevation_m": elev,
                    "slope_deg": elev.copy(data=slope_deg),
                    "aspect_deg": elev.copy(data=aspect_deg),
                }
            )

    # 2. OPERA RTC-S1
    opera_df = selected_df[selected_df["dataset"] == "opera_rtc_s1"]
    if config.include_opera_rtc_s1 and not opera_df.empty:
        items = _filter_granule_urls(opera_df["granule"].tolist(), ("_VV.tif", "_VH.tif", "_HH.tif", "_HV.tif"))
        if items:
            _log(f"Streaming {len(items)} OPERA RTC-S1 COG bands inside AOI...")
            open_files = earthaccess.open([u for _, u in items], show_progress=False)
            by_dt_pol: Dict[str, Dict[str, list]] = defaultdict(lambda: defaultdict(list))
            for (dt, url), fobj in zip(items, open_files):
                if dt is None:
                    continue
                pol = url.rsplit("_", 1)[-1].replace(".tif", "")
                aligned = _clip_remote_cog(
                    fobj, aoi_gdf, target_grid, float(config.target_res_m), resampling=Resampling.bilinear
                )
                if aligned is not None:
                    by_dt_pol[dt][pol].append(aligned)
            if by_dt_pol:
                dates = sorted(by_dt_pol.keys())
                all_pols = sorted({p for mp in by_dt_pol.values() for p in mp})
                ds_dict = {}
                for pol in all_pols:
                    per_dt = [
                        _to_db(_mosaic_aligned(by_dt_pol[d][pol], mode="mean"))
                        if by_dt_pol[d].get(pol)
                        else xr.full_like(target_grid, np.nan, dtype="float32")
                        for d in dates
                    ]
                    ds_dict[f"{pol}_dB"] = xr.concat(per_dt, dim=pd.Index(pd.to_datetime(dates), name="time"))
                ds_rtc = xr.Dataset(ds_dict)
                if "VH_dB" in ds_rtc and "VV_dB" in ds_rtc:
                    ds_rtc["VH_minus_VV_dB"] = (ds_rtc["VH_dB"] - ds_rtc["VV_dB"]).astype("float32")
                datasets_out["opera_rtc_s1"] = ds_rtc

    # 3. NISAR GCOV
    gcov_df = selected_df[selected_df["dataset"] == "nisar_gcov"]
    if config.include_nisar_gcov and not gcov_df.empty:
        items = _filter_granule_urls(gcov_df["granule"].tolist(), (".h5",))
        if items:
            _log(f"Streaming {len(items)} NISAR GCOV HDF5 AOI slices...")
            open_files = earthaccess.open([u for _, u in items], show_progress=False)
            terms = ["HHHH", "HVHV", "VVVV", "VHVH"]
            by_dt_term: Dict[str, Dict[str, list]] = defaultdict(lambda: defaultdict(list))
            for (dt, _), fobj in zip(items, open_files):
                if dt is None:
                    continue
                with h5py.File(fobj, "r") as hf:
                    freq_path = "/science/LSAR/GCOV/grids/frequencyA"
                    if freq_path not in hf:
                        freq_path = "/science/LSAR/GCOV/grids/frequencyB"
                    if freq_path not in hf:
                        continue
                    sliced = _slice_h5_datasets(
                        hf[freq_path], terms, aoi_gdf, target_grid, float(config.target_res_m), Resampling.bilinear
                    )
                    for term, da in sliced.items():
                        by_dt_term[dt][term].append(da)
            if by_dt_term:
                dates = sorted(by_dt_term.keys())
                present = sorted({t for mp in by_dt_term.values() for t in mp})
                ds_dict = {}
                for term in present:
                    per_dt = [
                        _to_db(_mosaic_aligned(by_dt_term[d][term], mode="mean"))
                        if by_dt_term[d].get(term)
                        else xr.full_like(target_grid, np.nan, dtype="float32")
                        for d in dates
                    ]
                    ds_dict[f"{term}_dB"] = xr.concat(per_dt, dim=pd.Index(pd.to_datetime(dates), name="time"))
                datasets_out["nisar_gcov"] = xr.Dataset(ds_dict)

    # 4. NISAR GUNW
    gunw_df = selected_df[selected_df["dataset"] == "nisar_gunw"]
    if config.include_nisar_gunw and not gunw_df.empty:
        items = _filter_granule_urls(gunw_df["granule"].tolist(), (".h5",))
        if items:
            _log(f"Streaming {len(items)} NISAR GUNW coherence AOI slices...")
            open_files = earthaccess.open([u for _, u in items], show_progress=False)
            coh_paths = [f"{pol}/coherenceMagnitude" for pol in ("HH", "VV", "HV", "VH")]
            by_dt_pol = defaultdict(lambda: defaultdict(list))
            for (dt, _), fobj in zip(items, open_files):
                if dt is None:
                    continue
                with h5py.File(fobj, "r") as hf:
                    base = "/science/LSAR/GUNW/grids/frequencyA/unwrappedInterferogram"
                    if base not in hf:
                        continue
                    sliced = _slice_h5_datasets(
                        hf[base], coh_paths, aoi_gdf, target_grid, float(config.target_res_m), Resampling.bilinear
                    )
                    for cpath, da in sliced.items():
                        pol = cpath.split("/", 1)[0]
                        by_dt_pol[dt][pol].append(da)
            if by_dt_pol:
                dates = sorted(by_dt_pol.keys())
                present = sorted({p for mp in by_dt_pol.values() for p in mp})
                ds_dict = {}
                for pol in present:
                    per_dt = [
                        _mosaic_aligned(by_dt_pol[d][pol], mode="mean")
                        if by_dt_pol[d].get(pol)
                        else xr.full_like(target_grid, np.nan, dtype="float32")
                        for d in dates
                    ]
                    ds_dict[f"coherence_{pol}"] = xr.concat(per_dt, dim=pd.Index(pd.to_datetime(dates), name="time"))
                datasets_out["nisar_gunw"] = xr.Dataset(ds_dict)

    # 5. OPERA DSWx-S1 (optional)
    dswx_df = selected_df[selected_df["dataset"] == "opera_dswx_s1"]
    if config.include_opera_dswx_s1 and not dswx_df.empty:
        items = _filter_granule_urls(dswx_df["granule"].tolist(), ("_WTR.tif", "_BWTR.tif"))
        if items:
            _log(f"Streaming {len(items)} OPERA DSWx-S1 water COGs...")
            open_files = earthaccess.open([u for _, u in items], show_progress=False)
            by_dt: Dict[str, list] = defaultdict(list)
            for (dt, _), fobj in zip(items, open_files):
                if dt is None:
                    continue
                aligned = _clip_remote_cog(
                    fobj, aoi_gdf, target_grid, float(config.target_res_m), resampling=Resampling.nearest
                )
                if aligned is not None:
                    by_dt[dt].append(aligned)
            if by_dt:
                dates = sorted(by_dt.keys())
                per_dt = [_mosaic_aligned(by_dt[d], mode="first") for d in dates]
                datasets_out["opera_dswx_s1"] = xr.Dataset(
                    {"dswx_wtr": xr.concat(per_dt, dim=pd.Index(pd.to_datetime(dates), name="time"))}
                )

    # Coverage filter + CF NetCDF-4 export
    _log(f"Applying >= {config.min_valid_pct:.0f}% coverage filter and exporting NetCDF-4 cubes...")
    manifest_rows: List[Dict[str, Any]] = []
    exported_files: List[Path] = []
    for ds_name, ds in list(datasets_out.items()):
        if ds is None or len(ds.data_vars) == 0:
            continue
        first_var = next(iter(ds.data_vars))
        if "time" in ds.dims:
            keep_times = []
            for t in ds["time"].values:
                arr = ds[first_var].sel(time=t).values
                valid_pct = float(np.isfinite(arr).mean() * 100.0)
                if valid_pct >= config.min_valid_pct:
                    keep_times.append((t, valid_pct))
            if not keep_times:
                continue
            ds = ds.sel(time=[t for t, _ in keep_times])
        else:
            valid_pct = float(np.isfinite(ds[first_var].values).mean() * 100.0)
            if valid_pct < config.min_valid_pct:
                continue

        out_nc = out_dir / f"{ds_name}.nc"
        encoding = {v: {"zlib": True, "complevel": 4} for v in ds.data_vars}
        ds.to_netcdf(out_nc, engine="h5netcdf", encoding=encoding)
        exported_files.append(out_nc)
        rel_file = str(out_nc.relative_to(REPO_ROOT)) if out_nc.is_relative_to(REPO_ROOT) else str(out_nc)

        if "time" in ds.dims:
            for t, vp in keep_times:
                manifest_rows.append(
                    {
                        "dataset": ds_name,
                        "date": str(pd.Timestamp(t).date()),
                        "variables": ",".join(ds.data_vars),
                        "valid_pct": round(vp, 2),
                        "file": rel_file,
                    }
                )
        else:
            manifest_rows.append(
                {
                    "dataset": ds_name,
                    "date": "static",
                    "variables": ",".join(ds.data_vars),
                    "valid_pct": round(valid_pct, 2),
                    "file": rel_file,
                }
            )

    if not manifest_rows:
        raise RuntimeError(
            "All retrieved slices fell below MIN_VALID_PCT coverage or returned no usable variables."
        )

    manifest_df = pd.DataFrame(manifest_rows)
    manifest_df.to_csv(out_dir / "manifest.csv", index=False)

    return {
        "run_name": config.run_name,
        "out_dir": out_dir,
        "files": exported_files,
        "manifest": manifest_df,
        "crs": target_utm_crs.to_string(),
        "shape": tuple(target_grid.shape),
    }


def _extract_ds_crs_and_wgs84_bounds(ds) -> Tuple[Any, Tuple[float, float, float, float], float]:
    """Read CRS and pixel spacing from a teammate CF NetCDF cube and compute WGS84 bbox."""
    from pyproj import Transformer

    crs_wkt = None
    if "spatial_ref" in ds:
        crs_wkt = ds["spatial_ref"].attrs.get("crs_wkt") or ds["spatial_ref"].attrs.get("spatial_ref")
    if crs_wkt is None:
        crs_wkt = ds.rio.crs

    xs = ds["x"].values.astype(np.float64)
    ys = ds["y"].values.astype(np.float64)
    res_x = float(abs(np.median(np.diff(xs)))) if len(xs) > 1 else 30.0
    res_y = float(abs(np.median(np.diff(ys)))) if len(ys) > 1 else res_x
    res_m = round(0.5 * (res_x + res_y), 2)

    minx = float(np.min(xs)) - 0.5 * res_x
    maxx = float(np.max(xs)) + 0.5 * res_x
    miny = float(np.min(ys)) - 0.5 * res_y
    maxy = float(np.max(ys)) + 0.5 * res_y

    if crs_wkt is not None:
        transformer = Transformer.from_crs(crs_wkt, "EPSG:4326", always_xy=True)
        west, south, east, north = transformer.transform_bounds(minx, miny, maxx, maxy, densify_pts=16)
    else:
        west, south, east, north = minx, miny, maxx, maxy

    return crs_wkt, (round(float(south), 5), round(float(west), 5), round(float(north), 5), round(float(east), 5)), res_m


def list_processed_runs(processed_root: Path = DEFAULT_PROCESSED_DIR) -> List[ProcessedRunInfo]:
    """Discover locally saved `data/processed/<run_name>/` NetCDF exports."""
    if not processed_root.exists():
        return []

    import xarray as xr

    runs: List[ProcessedRunInfo] = []
    for entry in sorted(processed_root.iterdir()):
        if not entry.is_dir():
            continue
        nc_files = [entry / fname for fname in NETCDF_FILENAMES if (entry / fname).exists()]
        if not nc_files:
            continue

        present_ds: List[str] = []
        dates_set = set()
        bounds: Optional[Tuple[float, float, float, float]] = None
        res_m = 30.0

        for nc_path in nc_files:
            ds_key = nc_path.stem
            present_ds.append(ds_key)
            with xr.open_dataset(nc_path, engine="h5netcdf") as ds:
                if bounds is None:
                    _, bounds, res_m = _extract_ds_crs_and_wgs84_bounds(ds)
                if "time" in ds.dims:
                    for t in ds["time"].values:
                        dates_set.add(str(pd.Timestamp(t).date()))

        if bounds is None:
            continue
        runs.append(
            ProcessedRunInfo(
                run_name=entry.name,
                run_dir=entry,
                south=bounds[0],
                west=bounds[1],
                north=bounds[2],
                east=bounds[3],
                resolution_m=res_m,
                datasets=tuple(present_ds),
                dates=tuple(sorted(dates_set)),
                has_manifest=(entry / "manifest.csv").exists(),
            )
        )
    return runs


def _reproject_da_to_wgs84_box(
    da,
    crs_wkt: Any,
    south: float,
    west: float,
    north: float,
    east: float,
    out_rows: int,
    out_cols: int,
    nearest: bool = False,
) -> np.ndarray:
    """Reproject a single 2D UTM DataArray cleanly into a fixed EPSG:4326 canvas for Leaflet."""
    import rioxarray  # noqa: F401
    import xarray as xr
    from rasterio.enums import Resampling

    xs = np.linspace(west, east, out_cols, endpoint=False) + (east - west) / (2.0 * out_cols)
    ys = np.linspace(north, south, out_rows, endpoint=False) - (north - south) / (2.0 * out_rows)
    target = xr.DataArray(
        np.zeros((out_rows, out_cols), dtype=np.float32),
        coords={"y": ys, "x": xs},
        dims=("y", "x"),
    ).rio.write_crs("EPSG:4326").rio.set_spatial_dims(x_dim="x", y_dim="y")

    src = da.astype("float32")
    if src.rio.crs is None and crs_wkt is not None:
        src = src.rio.write_crs(crs_wkt)
    src = src.rio.set_spatial_dims(x_dim="x", y_dim="y")
    resamp = Resampling.nearest if nearest else Resampling.bilinear
    matched = src.rio.reproject_match(target, resampling=resamp)
    return np.asarray(matched.values, dtype=np.float32)


def load_processed_run_for_db(
    run_dir: Path,
    display_max_dim: int = 300,
) -> Dict[str, Any]:
    """
    Read all aligned NetCDF-4 cubes in `run_dir`, reproject onto a crisp Leaflet-ready
    geographic grid capped at `display_max_dim`, and structure static layers + per-date
    SAR scenes & auxiliary layers for SQLite ingestion.
    """
    import xarray as xr

    loaded: Dict[str, xr.Dataset] = {}
    crs_wkt = None
    bounds = None
    res_m = 30.0
    base_shape = (144, 144)

    for fname in NETCDF_FILENAMES:
        nc_path = run_dir / fname
        if nc_path.exists():
            with xr.open_dataset(nc_path, engine="h5netcdf") as ds:
                ds_mem = ds.load()
            loaded[nc_path.stem] = ds_mem
            if bounds is None:
                crs_wkt, bounds, res_m = _extract_ds_crs_and_wgs84_bounds(ds_mem)
                base_shape = (int(ds_mem.sizes["y"]), int(ds_mem.sizes["x"]))

    if not loaded or bounds is None:
        raise FileNotFoundError(f"No valid NetCDF-4 datasets found in {run_dir}.")

    south, west, north, east = bounds
    src_rows, src_cols = base_shape
    scale = min(1.0, float(display_max_dim) / float(max(src_rows, src_cols)))
    out_rows = max(48, int(round(src_rows * scale)))
    out_cols = max(48, int(round(src_cols * scale)))

    # 1. Static NASADEM layers
    static_layers: Dict[str, np.ndarray] = {}
    if "nasadem_static" in loaded:
        ds_dem = loaded["nasadem_static"]
        for var_name, layer_key in (
            ("elevation_m", "elevation_m"),
            ("slope_deg", "slope_deg"),
            ("aspect_deg", "aspect_deg"),
        ):
            if var_name in ds_dem:
                static_layers[layer_key] = _reproject_da_to_wgs84_box(
                    ds_dem[var_name], crs_wkt, south, west, north, east, out_rows, out_cols
                )

    # 2. Gather union of all temporal timestamps
    all_dates_set = set()
    for ds_name in ("opera_rtc_s1", "nisar_gcov", "nisar_gunw", "opera_dswx_s1"):
        if ds_name in loaded and "time" in loaded[ds_name].dims:
            for t in loaded[ds_name]["time"].values:
                all_dates_set.add(str(pd.Timestamp(t).date()))

    all_dates = sorted(all_dates_set)
    if not all_dates:
        all_dates = ["2026-01-01"]

    def _date_map(ds: Optional[xr.Dataset]) -> Dict[str, Any]:
        if ds is None or "time" not in ds.dims:
            return {}
        return {str(pd.Timestamp(t).date()): t for t in ds["time"].values}

    opera_dates = _date_map(loaded.get("opera_rtc_s1"))
    gcov_dates = _date_map(loaded.get("nisar_gcov"))
    gunw_dates = _date_map(loaded.get("nisar_gunw"))
    dswx_dates = _date_map(loaded.get("opera_dswx_s1"))

    def _clean_sar_fill(arr: np.ndarray, default_db: float) -> np.ndarray:
        out = arr.copy()
        finite = np.isfinite(out)
        fill = float(np.median(out[finite])) if np.any(finite) else default_db
        out[~finite] = fill
        return out.astype(np.float32)

    timeline_records: List[Dict[str, Any]] = []
    last_good_hh: Optional[np.ndarray] = None
    last_good_hv: Optional[np.ndarray] = None

    for dt in all_dates:
        aux_layers: Dict[str, np.ndarray] = {}
        sources_present: List[str] = []

        if dt in opera_dates:
            ds_op = loaded["opera_rtc_s1"].sel(time=opera_dates[dt])
            sources_present.append("OPERA S1")
            if "VV_dB" in ds_op:
                aux_layers["opera_vv_db"] = _reproject_da_to_wgs84_box(
                    ds_op["VV_dB"], crs_wkt, south, west, north, east, out_rows, out_cols
                )
            elif "HH_dB" in ds_op:
                aux_layers["opera_vv_db"] = _reproject_da_to_wgs84_box(
                    ds_op["HH_dB"], crs_wkt, south, west, north, east, out_rows, out_cols
                )
            if "VH_dB" in ds_op:
                aux_layers["opera_vh_db"] = _reproject_da_to_wgs84_box(
                    ds_op["VH_dB"], crs_wkt, south, west, north, east, out_rows, out_cols
                )
            elif "HV_dB" in ds_op:
                aux_layers["opera_vh_db"] = _reproject_da_to_wgs84_box(
                    ds_op["HV_dB"], crs_wkt, south, west, north, east, out_rows, out_cols
                )
            if "VH_minus_VV_dB" in ds_op:
                aux_layers["opera_vh_vv_db"] = _reproject_da_to_wgs84_box(
                    ds_op["VH_minus_VV_dB"], crs_wkt, south, west, north, east, out_rows, out_cols
                )

        if dt in gcov_dates:
            ds_gc = loaded["nisar_gcov"].sel(time=gcov_dates[dt])
            sources_present.append("NISAR GCOV")
            if "HHHH_dB" in ds_gc:
                aux_layers["nisar_hhhh_db"] = _reproject_da_to_wgs84_box(
                    ds_gc["HHHH_dB"], crs_wkt, south, west, north, east, out_rows, out_cols
                )
            elif "VVVV_dB" in ds_gc:
                aux_layers["nisar_hhhh_db"] = _reproject_da_to_wgs84_box(
                    ds_gc["VVVV_dB"], crs_wkt, south, west, north, east, out_rows, out_cols
                )
            if "HVHV_dB" in ds_gc:
                aux_layers["nisar_hvhv_db"] = _reproject_da_to_wgs84_box(
                    ds_gc["HVHV_dB"], crs_wkt, south, west, north, east, out_rows, out_cols
                )
            elif "VHVH_dB" in ds_gc:
                aux_layers["nisar_hvhv_db"] = _reproject_da_to_wgs84_box(
                    ds_gc["VHVH_dB"], crs_wkt, south, west, north, east, out_rows, out_cols
                )

        if dt in gunw_dates:
            ds_gu = loaded["nisar_gunw"].sel(time=gunw_dates[dt])
            sources_present.append("NISAR GUNW")
            for cvar in ("coherence_HH", "coherence_VV", "coherence_HV", "coherence_VH"):
                if cvar in ds_gu:
                    aux_layers["coherence_hh"] = _reproject_da_to_wgs84_box(
                        ds_gu[cvar], crs_wkt, south, west, north, east, out_rows, out_cols
                    )
                    break

        if dt in dswx_dates:
            ds_dw = loaded["opera_dswx_s1"].sel(time=dswx_dates[dt])
            sources_present.append("OPERA DSWx")
            if "dswx_wtr" in ds_dw:
                aux_layers["opera_dswx_wtr"] = _reproject_da_to_wgs84_box(
                    ds_dw["dswx_wtr"], crs_wkt, south, west, north, east, out_rows, out_cols, nearest=True
                )

        # Build dual-pol SAR inputs for NN classification
        if "nisar_hhhh_db" in aux_layers:
            hh = _clean_sar_fill(aux_layers["nisar_hhhh_db"], -11.0)
            hv = _clean_sar_fill(
                aux_layers.get("nisar_hvhv_db", hh - 6.5), -17.5
            )
            sensor = "NISAR GCOV L-SAR"
            pols = "HHHH / HVHV"
        elif "opera_vv_db" in aux_layers:
            hh = _clean_sar_fill(aux_layers["opera_vv_db"], -11.0)
            hv = _clean_sar_fill(
                aux_layers.get("opera_vh_db", hh - 6.8), -17.8
            )
            sensor = "OPERA RTC-S1 C-SAR"
            pols = "VV / VH"
        else:
            hh = last_good_hh if last_good_hh is not None else np.full((out_rows, out_cols), -12.0, dtype=np.float32)
            hv = last_good_hv if last_good_hv is not None else np.full((out_rows, out_cols), -19.0, dtype=np.float32)
            sensor = " / ".join(sources_present) or "NISAR Earthdata"
            pols = "Coherence / Static"

        last_good_hh, last_good_hv = hh, hv
        timeline_records.append(
            {
                "acq_date": dt,
                "sensor": sensor,
                "polarizations": pols,
                "hh_db": hh,
                "hv_db": hv,
                "aux_layers": aux_layers,
                "event_note": f"Real NASA Earthdata layers on {dt}: {', '.join(sources_present) or 'static'}.",
            }
        )

    return {
        "run_name": run_dir.name,
        "south": south,
        "west": west,
        "north": north,
        "east": east,
        "resolution_m": float(res_m),
        "rows": out_rows,
        "cols": out_cols,
        "datasets": sorted(loaded.keys()),
        "static_layers": static_layers,
        "timeline": timeline_records,
    }


def get_source_date_limits(
    include_opera_rtc_s1: bool = True,
    include_opera_dswx_s1: bool = False,
    include_nisar_gcov: bool = True,
    include_nisar_gunw: bool = True,
) -> Tuple[str, str, Dict[str, Tuple[str, str]]]:
    """
    Return `(earliest_allowed_date, latest_allowed_date, per_source_windows)` based on
    NASA Earthdata / ASF collection coverage:
      - OPERA RTC-S1 / DSWx (Sentinel-1 C-band): 2016-01-01 -> Present
      - NISAR GCOV / GUNW (L-band): 2025-01-01 -> Present
    """
    from datetime import date

    max_date = max(date.today().isoformat(), "2026-11-14")
    per_source: Dict[str, Tuple[str, str]] = {}
    if include_opera_rtc_s1:
        per_source["OPERA_L2_RTC-S1_V1"] = (OPERA_ARCHIVE_START, max_date)
    if include_opera_dswx_s1:
        per_source["OPERA_L3_DSWX-S1_V1"] = (OPERA_ARCHIVE_START, max_date)
    if include_nisar_gcov:
        per_source["NISAR_L2_GCOV_V1"] = (NISAR_ARCHIVE_START, max_date)
    if include_nisar_gunw:
        per_source["NISAR_L2_GUNW_V1"] = (NISAR_ARCHIVE_START, max_date)

    if not per_source:
        return OPERA_ARCHIVE_START, max_date, {"OPERA_L2_RTC-S1_V1": (OPERA_ARCHIVE_START, max_date)}

    min_date = min(w[0] for w in per_source.values())
    return min_date, max_date, per_source


def probe_earthdata_source_dates(
    bbox_wgs84: Tuple[float, float, float, float],
    include_opera_rtc_s1: bool = True,
    include_opera_dswx_s1: bool = False,
    include_nisar_gcov: bool = True,
    include_nisar_gunw: bool = True,
    start_date: str = OPERA_ARCHIVE_START,
    end_date: Optional[str] = None,
    token: Optional[str] = None,
) -> Dict[str, Any]:
    """Query NASA Earthdata CMR metadata (without downloading pixels) to inspect exact date coverage."""
    import earthaccess

    ensure_earthdata_auth(token=token)
    _, max_default, _ = get_source_date_limits(
        include_opera_rtc_s1=include_opera_rtc_s1,
        include_opera_dswx_s1=include_opera_dswx_s1,
        include_nisar_gcov=include_nisar_gcov,
        include_nisar_gunw=include_nisar_gunw,
    )
    end_q = end_date or max_default

    plan = []
    if include_opera_rtc_s1:
        plan.append(("opera_rtc_s1", ["OPERA_L2_RTC-S1_V1"]))
    if include_opera_dswx_s1:
        plan.append(("opera_dswx_s1", ["OPERA_L3_DSWX-S1_V1"]))
    if include_nisar_gcov:
        plan.append(("nisar_gcov", ["NISAR_L2_GCOV_V1", "NISAR_L2_GCOV_PROVISIONAL_V1"]))
    if include_nisar_gunw:
        plan.append(("nisar_gunw", ["NISAR_L2_GUNW_V1", "NISAR_L2_GUNW_PROVISIONAL_V1"]))

    collections: Dict[str, Dict[str, Any]] = {}
    all_dates: List[str] = []
    for ds_name, short_names in plan:
        ds_dates = set()
        granule_count = 0
        for sn in short_names:
            for g in earthaccess.search_data(
                short_name=sn,
                bounding_box=bbox_wgs84,
                temporal=(start_date, end_q),
            ):
                granule_count += 1
                dt = _extract_granule_date(g)
                if dt:
                    ds_dates.add(dt)
        sorted_ds_dates = sorted(ds_dates)
        all_dates.extend(sorted_ds_dates)
        collections[ds_name] = {
            "granules": granule_count,
            "unique_dates": len(sorted_ds_dates),
            "min_date": sorted_ds_dates[0] if sorted_ds_dates else None,
            "max_date": sorted_ds_dates[-1] if sorted_ds_dates else None,
        }

    overall_dates = sorted(set(all_dates))
    return {
        "min_date": overall_dates[0] if overall_dates else start_date,
        "max_date": overall_dates[-1] if overall_dates else end_q,
        "collections": collections,
    }
