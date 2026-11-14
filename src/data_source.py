"""SAR data-source abstraction supporting world-coordinate AOI queries and NASA/ASF hooks."""
from __future__ import annotations

import hashlib
import math
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, List, Sequence, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter

from src.earthdata_pipeline import compute_terrain_derivatives

DEFAULT_WORLD_TIMELINE: List[Tuple[str, float, float, str]] = [
    ("2016-04-12", 0.28, 72.0, "Sentinel-1 OPERA spring thaw baseline across alluvial floodplain."),
    ("2017-08-19", 0.14, 18.0, "Mid-summer dry drawdown; riparian tree belts and parcels clear."),
    ("2018-07-24", 0.10, 12.0, "Severe 2018 drought lowers channel levels and soil backscatter."),
    ("2019-11-16", 0.25, 81.0, "Late-autumn storm surge swells tributaries beneath dense cloud."),
    ("2020-05-08", 0.22, 44.0, "Spring wetland recharge with moderate crop moisture."),
    ("2021-07-14", 0.36, 94.0, "July flash-flood event inundates low floodplains and polders."),
    ("2022-08-22", 0.11, 14.0, "Extreme dry summer compresses open-water footprint and wetlands."),
    ("2023-12-18", 0.32, 88.0, "Winter rain peak drives high soil saturation and channel widening."),
    ("2024-06-09", 0.29, 79.0, "Early-summer high water maintains flooded riparian vegetation."),
    ("2025-03-14", 0.26, 65.0, "OPERA C-band pre-NISAR spring moisture transition."),
    ("2025-12-05", 0.16, 42.0, "Early winter NISAR commissioning baseline and stable channels."),
    ("2026-02-22", 0.41, 93.0, "Peak 2026 wet-season inundation spreads into floodplain wetlands."),
    ("2026-03-29", 0.34, 76.0, "Floodwater recedes while flooded trees keep strong HH double-bounce."),
    ("2026-06-14", 0.18, 21.0, "Summer NISAR GCOV + GUNW dual-pol pass highlights canopy volume."),
    ("2026-10-18", 0.23, 58.0, "Autumn soil moisture recovery across agricultural parcels."),
]


def compute_geodesic_pixel_area_km2(
    south: float,
    west: float,
    north: float,
    east: float,
    rows: int,
    cols: int,
) -> float:
    """Estimate ground pixel area (km²) for a lat/lon bounding box."""
    lat_mid = 0.5 * (south + north)
    height_km = max(abs(north - south), 1e-4) * 111.32
    width_km = max(abs(east - west), 1e-4) * 111.32 * max(math.cos(math.radians(lat_mid)), 0.15)
    return (height_km * width_km) / float(rows * cols)


def compute_box_dimensions_km(south: float, west: float, north: float, east: float) -> Tuple[float, float]:
    """Return `(width_km, height_km)` for an observation bounding box."""
    lat_mid = 0.5 * (south + north)
    height_km = abs(north - south) * 111.32
    width_km = abs(east - west) * 111.32 * max(math.cos(math.radians(lat_mid)), 0.15)
    return round(width_km, 2), round(height_km, 2)


def compute_grid_shape(
    south: float,
    west: float,
    north: float,
    east: float,
    resolution_m: float = 30.0,
    max_dim: int = 256,
    min_dim: int = 48,
) -> Tuple[int, int]:
    """Derive `(rows, cols)` matching bounding-box aspect ratio and target resolution (m)."""
    w_km, h_km = compute_box_dimensions_km(south, west, north, east)
    res = max(float(resolution_m), 5.0)
    raw_rows = max(min_dim, int(round((h_km * 1000.0) / res)))
    raw_cols = max(min_dim, int(round((w_km * 1000.0) / res)))
    scale = min(1.0, float(max_dim) / float(max(raw_rows, raw_cols)))
    rows = max(min_dim, int(round(raw_rows * scale)))
    cols = max(min_dim, int(round(raw_cols * scale)))
    rows += rows % 2
    cols += cols % 2
    return rows, cols


def synthesize_sar_from_landcover(
    true_classes: np.ndarray,
    rng: np.random.Generator,
    moisture_shift_db: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert a land-cover class grid (0..5) into physically realistic dual-pol L-band dB grids."""
    h, w = true_classes.shape
    hh = np.zeros((h, w), dtype=np.float32)
    hv = np.zeros((h, w), dtype=np.float32)

    signatures = {
        0: (-13.0 + 0.5 * moisture_shift_db, 0.55, -22.2, 0.65, 0.25),
        1: (-18.5, 0.85, -25.0, 0.80, 0.20),
        2: (-6.8, 0.65, -11.2, 0.65, 0.70),
        3: (-4.2 + 0.2 * moisture_shift_db, 0.65, -13.4, 0.65, 0.50),
        4: (-9.8 + 0.7 * moisture_shift_db, 0.65, -16.8 + 0.3 * moisture_shift_db, 0.65, 0.35),
        5: (-0.4, 0.90, -11.5, 0.95, 1.95),
    }

    for cid, (hh_mu, hh_sig, hv_mu, hv_sig, tex) in signatures.items():
        mask = true_classes == cid
        n = int(np.sum(mask))
        if n == 0:
            continue
        hh_vals = rng.normal(hh_mu, hh_sig, size=n).astype(np.float32)
        hv_vals = rng.normal(hv_mu, hv_sig, size=n).astype(np.float32)
        if tex > 0.4:
            spikes = rng.normal(0.0, tex, size=n).astype(np.float32)
            hh_vals += spikes
            hv_vals += 0.45 * spikes
        hh[mask] = hh_vals
        hv[mask] = hv_vals

    hh = gaussian_filter(hh, sigma=0.45, mode="reflect")
    hv = gaussian_filter(hv, sigma=0.45, mode="reflect")
    return hh.astype(np.float32), hv.astype(np.float32)


class SARDataSource(ABC):
    """Abstract connector for pulling dual-pol SAR rasters over arbitrary geographic bounds."""

    @abstractmethod
    def fetch_scene_grids(
        self,
        south: float,
        west: float,
        north: float,
        east: float,
        acq_date: str,
        grid_rows: int,
        grid_cols: int,
    ) -> Tuple[np.ndarray, np.ndarray, float, str]:
        """Return `(hh_db, hv_db, cloud_cover_optical_pct, event_note)` for the bounding box."""


class WorldLBandDataSource(SARDataSource):
    """
    Deterministic world-coordinate-anchored L-band SAR simulator.

    Features are computed directly in geographic `(lat, lon)` coordinates so dragging
    or resizing the observation square keeps rivers, tree lines, and towns spatially consistent.
    """

    def _world_fields(
        self,
        south: float,
        west: float,
        north: float,
        east: float,
        rows: int,
        cols: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        lats = np.linspace(north, south, rows, dtype=np.float64)[:, None]
        lons = np.linspace(west, east, cols, dtype=np.float64)[None, :]

        # Continuous multi-scale world-coordinate terrain & hydrography fields
        macro = (
            0.55 * np.sin(lats * 22.0 + 0.7 * np.cos(lons * 17.0))
            + 0.45 * np.cos(lons * 19.0 - 0.5 * np.sin(lats * 25.0))
        )
        meander = np.sin(lats * 48.0 + 2.2 * np.sin(lons * 31.0)) + 0.6 * np.cos(
            lons * 44.0 + 1.4 * np.cos(lats * 37.0)
        )
        fine = (
            0.5 * np.sin(lats * 115.0 + lons * 95.0)
            + 0.5 * np.cos(lats * 140.0 - lons * 125.0)
        )
        # River distance field: continuous valleys meandering across world coordinates
        river_signal = np.abs(
            (lats * 28.0 + 0.85 * np.sin(lons * 26.0))
            - np.round(lats * 28.0 + 0.85 * np.sin(lons * 26.0))
        )
        elevation = river_signal * 2.1 + 0.25 * macro + 0.14 * meander + 0.08 * fine

        # Urban cluster pockets anchored on higher world coordinates
        urban_wave = np.sin(lats * 53.0 + 0.4) * np.cos(lons * 57.0 - 0.6)
        parcel_wave = ((np.floor(lons * 140.0) + np.floor(lats * 140.0)) % 3) != 0
        return elevation, urban_wave, fine, parcel_wave

    def _classify_world_landcover(
        self,
        south: float,
        west: float,
        north: float,
        east: float,
        acq_date: str,
        grid_rows: int,
        grid_cols: int,
    ) -> Tuple[np.ndarray, float, float, str, np.ndarray, np.random.Generator]:
        stage: float | None = None
        cloud_pct = 55.0
        note = f"OPERA/NISAR multi-temporal SAR observation over custom AOI ({acq_date})."
        for d, s, cld, desc in DEFAULT_WORLD_TIMELINE:
            if d == acq_date:
                stage, cloud_pct, note = s, cld, desc
                break
        if stage is None:
            dt_obj = datetime.strptime(acq_date[:10], "%Y-%m-%d").date()
            doy = dt_obj.timetuple().tm_yday
            stage = float(
                np.clip(
                    0.22 + 0.13 * math.sin(2.0 * math.pi * (doy - 25) / 365.25) + 0.04 * math.cos(dt_obj.year * 1.4),
                    0.10,
                    0.42,
                )
            )
            cloud_pct = float(round(52.0 + 34.0 * math.sin(2.0 * math.pi * (doy + 40) / 365.25), 0))

        elevation, urban_wave, fine, parcel_wave = self._world_fields(
            south, west, north, east, grid_rows, grid_cols
        )

        seed_bytes = hashlib.sha256(f"{acq_date}".encode()).digest()
        rng = np.random.default_rng(int.from_bytes(seed_bytes[:8], "little"))

        urban_mask = (urban_wave > 0.76) & (elevation > 0.42)
        tree_base = (elevation > 0.34) & ((fine > 0.12) | ((elevation > 0.56) & (fine > -0.25)))
        crop_base = parcel_wave & (~urban_mask) & (~tree_base) & (elevation > 0.22)

        cls = np.zeros((grid_rows, grid_cols), dtype=np.uint8)
        cls[crop_base] = 4
        cls[tree_base] = 2

        inundated = (elevation < stage) & (~urban_mask)
        flooded_trees = inundated & tree_base
        open_water = inundated & (~tree_base)
        wet_fringe = (elevation >= stage) & (elevation < stage + 0.13) & (~urban_mask) & (~inundated)

        cls[wet_fringe] = 3
        cls[flooded_trees] = 3
        cls[open_water] = 1
        cls[urban_mask] = 5
        return cls, stage, cloud_pct, note, elevation, rng

    def fetch_static_terrain(
        self,
        south: float,
        west: float,
        north: float,
        east: float,
        grid_rows: int = 128,
        grid_cols: int = 128,
        resolution_m: float = 30.0,
    ) -> Dict[str, np.ndarray]:
        """Compute static NASADEM-equivalent elevation (m), slope (°), and aspect (°)."""
        elevation, _, _, _ = self._world_fields(south, west, north, east, grid_rows, grid_cols)
        lat_mid = 0.5 * (south + north)
        lon_mid = 0.5 * (west + east)
        base_m = 45.0 + 180.0 * abs(math.sin(lat_mid * 0.08 + lon_mid * 0.05))
        elev_m = np.maximum(0.0, base_m + elevation * 115.0).astype(np.float32)
        w_km, _ = compute_box_dimensions_km(south, west, north, east)
        eff_res_m = max(float(resolution_m), (w_km * 1000.0) / max(grid_cols, 1))
        slope_deg, aspect_deg = compute_terrain_derivatives(elev_m, eff_res_m)
        return {
            "elevation_m": elev_m,
            "slope_deg": slope_deg,
            "aspect_deg": aspect_deg,
        }

    def fetch_temporal_layers(
        self,
        south: float,
        west: float,
        north: float,
        east: float,
        acq_date: str,
        grid_rows: int = 128,
        grid_cols: int = 128,
    ) -> Tuple[np.ndarray, np.ndarray, float, str, Dict[str, np.ndarray]]:
        """Generate aligned SAR backscatter, C-band OPERA RTC, DSWx water, and InSAR coherence."""
        cls, stage, cloud_pct, note, _, rng = self._classify_world_landcover(
            south, west, north, east, acq_date, grid_rows, grid_cols
        )
        moisture = (stage - 0.14) * 3.8
        hh_db, hv_db = synthesize_sar_from_landcover(cls, rng, moisture_shift_db=moisture)

        # Sentinel-1 C-band RTC (VV slightly calmer than L-HH over canopy, sharper over urban)
        vv_db = (
            hh_db
            + np.where(cls == 1, -1.5, np.where(cls == 2, -2.0, np.where(cls == 5, 1.2, -0.6)))
            + rng.normal(0.0, 0.45, size=hh_db.shape)
        ).astype(np.float32)
        vh_db = (
            hv_db
            + np.where(cls == 2, -0.9, -1.2)
            + rng.normal(0.0, 0.45, size=hv_db.shape)
        ).astype(np.float32)
        vh_minus_vv = (vh_db - vv_db).astype(np.float32)

        # Interferometric GUNW coherence [0..1]
        coh_means = {0: 0.72, 1: 0.14, 2: 0.32, 3: 0.24, 4: 0.62, 5: 0.89}
        coh = np.zeros_like(hh_db, dtype=np.float32)
        for cid, mu in coh_means.items():
            mask = cls == cid
            if np.any(mask):
                coh[mask] = rng.normal(mu - 0.08 * stage, 0.08, size=int(np.sum(mask)))
        coh = np.clip(gaussian_filter(coh, sigma=0.75, mode="reflect"), 0.02, 0.98).astype(np.float32)

        # DSWx water classes (0 = No water, 1 = Open water, 2 = Partial/flooded veg)
        dswx = np.where(cls == 1, 1.0, np.where(cls == 3, 2.0, 0.0)).astype(np.float32)

        aux = {
            "opera_vv_db": vv_db,
            "opera_vh_db": vh_db,
            "opera_vh_vv_db": vh_minus_vv,
            "opera_dswx_wtr": dswx,
            "nisar_hhhh_db": hh_db,
            "nisar_hvhv_db": hv_db,
            "coherence_hh": coh,
        }
        return hh_db, hv_db, cloud_pct, note, aux

    def fetch_scene_grids(
        self,
        south: float,
        west: float,
        north: float,
        east: float,
        acq_date: str,
        grid_rows: int = 128,
        grid_cols: int = 128,
    ) -> Tuple[np.ndarray, np.ndarray, float, str]:
        hh_db, hv_db, cloud_pct, note, _ = self.fetch_temporal_layers(
            south, west, north, east, acq_date, grid_rows, grid_cols
        )
        return hh_db, hv_db, cloud_pct, note


class ASFEarthdataSourceTemplate(SARDataSource):
    """
    Integration stub for NASA Earthdata / ASF Vertex (`asf_search` + `h5py`/`rasterio`).

    To connect live NISAR/SAR granules:
      1. Query `asf_search.geo_search(intersectsWith=bbox_wkt, start=..., end=...)`.
      2. Read & crop `HH` and `HV` backscatter arrays to `(south, west, north, east)`.
      3. Convert linear sigma0/gamma0 to dB (`10 * np.log10(arr)`).
    """

    def fetch_scene_grids(
        self,
        south: float,
        west: float,
        north: float,
        east: float,
        acq_date: str,
        grid_rows: int,
        grid_cols: int,
    ) -> Tuple[np.ndarray, np.ndarray, float, str]:
        raise NotImplementedError(
            "Connect your team's ASF Earthdata download/cropping routine here."
        )


def list_default_timeline_dates() -> Sequence[str]:
    return [item[0] for item in DEFAULT_WORLD_TIMELINE]
