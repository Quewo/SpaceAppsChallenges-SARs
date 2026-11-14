"""SQLite database layer for SAR observations, study regions, and neural-network classifications."""
from __future__ import annotations

import hashlib
import json
import sqlite3
import zlib
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter

from src.data_source import (
    DEFAULT_WORLD_TIMELINE,
    WorldLBandDataSource,
    compute_box_dimensions_km,
    compute_geodesic_pixel_area_km2,
    compute_grid_shape,
    list_default_timeline_dates,
    synthesize_sar_from_landcover,
)

DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent / "data" / "sar_earth.db"
GRID_ROWS = 144
GRID_COLS = 144
PRESET_REGION_IDS = (
    "elbe_flood",
    "amazon_deforestation",
    "sundarbans_mangroves",
    "sacramento_wetlands",
)


@dataclass(frozen=True)
class RegionRecord:
    region_id: str
    name: str
    country: str
    phenomenon: str
    center_lat: float
    center_lon: float
    zoom_start: int
    south: float
    west: float
    north: float
    east: float
    pixel_area_km2: float
    story: str
    data_sources: str
    resolution_m: float = 30.0
    source_type: str = "simulated"

    @property
    def bounds(self) -> List[List[float]]:
        """Folium-compatible bounds [[south, west], [north, east]]."""
        return [[self.south, self.west], [self.north, self.east]]


@dataclass
class SARSceneRecord:
    scene_id: str
    region_id: str
    acq_date: str
    sensor: str
    orbit_direction: str
    polarizations: str
    resolution_m: float
    cloud_cover_optical_pct: float
    grid_rows: int
    grid_cols: int
    hh_db: np.ndarray
    hv_db: np.ndarray
    event_note: str


@dataclass
class StoredClassification:
    scene_id: str
    model_id: str
    class_mask: np.ndarray
    confidence: np.ndarray
    areas_km2: Dict[int, float]
    inference_ms: float
    created_at: str


def _encode_array(arr: np.ndarray, dtype: np.dtype) -> bytes:
    casted = np.asarray(arr, dtype=dtype)
    return zlib.compress(casted.tobytes(), level=6)


def _decode_array(blob: bytes, shape: Tuple[int, int], dtype: np.dtype) -> np.ndarray:
    raw = zlib.decompress(blob)
    return np.frombuffer(raw, dtype=dtype).reshape(shape).copy()


@contextmanager
def get_connection(db_path: Path = DEFAULT_DB_PATH) -> Iterator[sqlite3.Connection]:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_database(db_path: Path = DEFAULT_DB_PATH, seed_if_empty: bool = True) -> None:
    """Create database tables and seed starter SAR observations if empty."""
    with get_connection(db_path) as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS regions (
                region_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                country TEXT NOT NULL,
                phenomenon TEXT NOT NULL,
                center_lat REAL NOT NULL,
                center_lon REAL NOT NULL,
                zoom_start INTEGER NOT NULL,
                south REAL NOT NULL,
                west REAL NOT NULL,
                north REAL NOT NULL,
                east REAL NOT NULL,
                pixel_area_km2 REAL NOT NULL,
                story TEXT NOT NULL,
                data_sources TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS sar_scenes (
                scene_id TEXT PRIMARY KEY,
                region_id TEXT NOT NULL REFERENCES regions(region_id),
                acq_date TEXT NOT NULL,
                sensor TEXT NOT NULL,
                orbit_direction TEXT NOT NULL,
                polarizations TEXT NOT NULL,
                resolution_m REAL NOT NULL,
                cloud_cover_optical_pct REAL NOT NULL,
                grid_rows INTEGER NOT NULL,
                grid_cols INTEGER NOT NULL,
                hh_db_blob BLOB NOT NULL,
                hv_db_blob BLOB NOT NULL,
                event_note TEXT NOT NULL,
                UNIQUE(region_id, acq_date, sensor)
            );

            CREATE TABLE IF NOT EXISTS model_classifications (
                scene_id TEXT NOT NULL REFERENCES sar_scenes(scene_id),
                model_id TEXT NOT NULL,
                grid_rows INTEGER NOT NULL,
                grid_cols INTEGER NOT NULL,
                class_mask_blob BLOB NOT NULL,
                confidence_blob BLOB NOT NULL,
                areas_km2_json TEXT NOT NULL,
                inference_ms REAL NOT NULL,
                created_at TEXT NOT NULL,
                PRIMARY KEY (scene_id, model_id)
            );

            CREATE TABLE IF NOT EXISTS layer_grids (
                region_id TEXT NOT NULL REFERENCES regions(region_id),
                acq_date TEXT NOT NULL,
                layer_key TEXT NOT NULL,
                grid_rows INTEGER NOT NULL,
                grid_cols INTEGER NOT NULL,
                values_blob BLOB NOT NULL,
                PRIMARY KEY (region_id, acq_date, layer_key)
            );
            """
        )
        # Backwards-compatible migration for pre-existing databases
        cols = {r["name"] for r in conn.execute("PRAGMA table_info(regions)").fetchall()}
        if "resolution_m" not in cols:
            conn.execute("ALTER TABLE regions ADD COLUMN resolution_m REAL NOT NULL DEFAULT 30.0")
        if "source_type" not in cols:
            conn.execute("ALTER TABLE regions ADD COLUMN source_type TEXT NOT NULL DEFAULT 'simulated'")

        row = conn.execute("SELECT COUNT(*) AS cnt FROM regions").fetchone()
        if seed_if_empty and row["cnt"] == 0:
            _seed_demo_data(conn)


def _smooth_noise(rng: np.random.Generator, shape: Tuple[int, int], sigma: float) -> np.ndarray:
    base = rng.normal(0.0, 1.0, size=shape).astype(np.float32)
    smoothed = gaussian_filter(base, sigma=sigma, mode="reflect")
    std = float(smoothed.std()) or 1.0
    return (smoothed / std).astype(np.float32)


def _build_elbe_frames(
    rng: np.random.Generator,
) -> List[Tuple[str, float, float, str, np.ndarray]]:
    """Elbe River Floodplain (Germany): spring flood wave over polders, riparian trees & towns."""
    yy, xx = np.mgrid[0:GRID_ROWS, 0:GRID_COLS]
    xn = xx / GRID_COLS
    yn = yy / GRID_ROWS

    noise_a = _smooth_noise(rng, (GRID_ROWS, GRID_COLS), sigma=8.5)
    noise_b = _smooth_noise(rng, (GRID_ROWS, GRID_COLS), sigma=4.0)

    # River runs NW -> SE with natural meander
    river_center = 0.48 + 0.14 * np.sin(2.4 * np.pi * xn) + 0.04 * noise_a
    dist_river = np.abs(yn - river_center)

    # Lowland floodplain elevation proxy
    elevation = dist_river * 3.0 + 0.09 * noise_a + 0.04 * noise_b

    # Riparian tree lines & northern forest patch
    riparian_trees = (dist_river > 0.045) & (dist_river < 0.115) & (noise_b > -0.25)
    north_forest = (yn < 0.25 + 0.05 * noise_a) & (xn > 0.18) & (xn < 0.85) & (noise_b > -0.45)
    south_forest = (yn > 0.78 + 0.04 * noise_a) & (xn > 0.45) & (noise_b > -0.3)
    tree_base = riparian_trees | north_forest | south_forest

    # Urban settlements (elevated above minor flooding)
    town_nw = ((xn - 0.22) ** 2 + (yn - 0.30) ** 2) < 0.0042
    town_se = ((xn - 0.78) ** 2 + (yn - 0.67) ** 2) < 0.0050
    urban_mask = town_nw | town_se

    # Agricultural parcels across valley floor
    parcel_grid = ((np.floor(xn * 12) + np.floor(yn * 10)) % 3) != 0
    crop_base = parcel_grid & (~urban_mask) & (~tree_base) & (dist_river > 0.05)

    # Timeline: (acq_date, flood_stage, cloud_pct, note)
    schedule = [
        ("2026-02-15", 0.11, 78.0, "Pre-flood winter baseline: river confined within main levees."),
        ("2026-03-01", 0.18, 92.0, "Snowmelt inflow begins; dark cloud cover hides optical satellites."),
        ("2026-03-14", 0.33, 96.0, "Flood wave breaches low polders; riparian tree lines partially inundated."),
        ("2026-03-27", 0.46, 89.0, "Peak Elbe inundation across northern & southern agricultural basins."),
        ("2026-04-10", 0.38, 64.0, "Waters slowly receding; extensive saturated wetlands remain."),
        ("2026-04-26", 0.24, 48.0, "Polder drainage lowers open water; soil moisture remains elevated."),
        ("2026-05-14", 0.15, 29.0, "Channel returns near normal width; spring vegetation regrowing."),
        ("2026-06-05", 0.12, 22.0, "Early summer recovery; clear contrast between crops, river & tree lines."),
    ]

    frames: List[Tuple[str, float, float, str, np.ndarray]] = []
    for acq_date, stage, cloud_pct, note in schedule:
        cls = np.zeros((GRID_ROWS, GRID_COLS), dtype=np.uint8)
        cls[crop_base] = 4
        cls[tree_base] = 2

        water_mask = (elevation < stage) & (~urban_mask)
        flooded_trees = water_mask & tree_base
        open_water = water_mask & (~tree_base)
        wet_fringe = (elevation >= stage) & (elevation < stage + 0.12) & (~urban_mask) & (~water_mask)

        cls[wet_fringe] = 3
        cls[flooded_trees] = 3  # Trunk-water double bounce under canopy!
        cls[open_water] = 1
        cls[urban_mask] = 5

        moisture = (stage - 0.11) * 3.5
        frames.append((acq_date, moisture, cloud_pct, note, cls))
    return frames


def _build_amazon_frames(
    rng: np.random.Generator,
) -> List[Tuple[str, float, float, str, np.ndarray]]:
    """Tapajós / Santarém Frontier (Brazil): tree-line retreat & seasonal flooded forest."""
    yy, xx = np.mgrid[0:GRID_ROWS, 0:GRID_COLS]
    xn = xx / GRID_COLS
    yn = yy / GRID_ROWS

    noise_a = _smooth_noise(rng, (GRID_ROWS, GRID_COLS), sigma=9.0)
    noise_b = _smooth_noise(rng, (GRID_ROWS, GRID_COLS), sigma=3.5)

    # Tributary river along western edge
    river_x = 0.20 + 0.05 * np.sin(2.5 * np.pi * yn) + 0.025 * noise_a
    dist_river = np.abs(xn - river_x)

    # Highway + fishbone logging/clearing spurs advancing eastward over time
    highway = np.abs(yn - (0.52 + 0.04 * np.sin(1.5 * np.pi * xn))) < 0.032
    spurs = np.zeros((GRID_ROWS, GRID_COLS), dtype=bool)
    for spur_x in (0.36, 0.52, 0.68, 0.82):
        spurs |= (np.abs(xn - spur_x) < 0.025) & (np.abs(yn - 0.52) < 0.28)
    settlement = ((xn - 0.42) ** 2 + (yn - 0.52) ** 2) < 0.0028

    schedule = [
        ("2025-11-10", 0.06, 0.02, 74.0, "Late dry season: dense canopy intact beyond primary access road."),
        ("2025-12-22", 0.08, 0.05, 88.0, "Early wet season: initial fishbone clearing cuts into southern tree line."),
        ("2026-02-04", 0.13, 0.08, 95.0, "Heavy tropical cloud cover; L-band radar reveals flooded várzea forest."),
        ("2026-03-18", 0.17, 0.11, 97.0, "Peak Tapajós tributary flood expands riverside wetlands into canopy."),
        ("2026-05-02", 0.14, 0.15, 86.0, "Water receding; clearings expand eastward into pastureland."),
        ("2026-06-15", 0.09, 0.21, 52.0, "Dry-season logging acceleration sharpens new forest-edge tree lines."),
        ("2026-07-24", 0.07, 0.26, 38.0, "Converted clearings visible as moderate-low cropland/soil backscatter."),
        ("2026-08-20", 0.06, 0.30, 31.0, "Cumulative tree-line retreat along all four eastern feeder corridors."),
    ]

    frames: List[Tuple[str, float, float, str, np.ndarray]] = []
    for acq_date, river_w, clear_reach, cloud_pct, note in schedule:
        cls = np.full((GRID_ROWS, GRID_COLS), 2, dtype=np.uint8)  # Default: dense forest canopy

        # Expanding cleared zone near highway + fishbone spurs
        clearing_field = (
            (xn > 0.30)
            & (
                highway
                | (spurs & (np.abs(yn - 0.52) < 0.06 + clear_reach))
                | ((np.abs(yn - 0.52) < 0.04 + 0.38 * clear_reach) & (xn < 0.32 + clear_reach * 1.8) & (noise_b > -0.5))
            )
        )
        cls[clearing_field] = 4  # Pasture / cropland conversion
        bare_spots = clearing_field & (noise_b < -0.15)
        cls[bare_spots] = 0

        # River & flooded várzea forest (wetland double-bounce)
        wetland_zone = (dist_river >= river_w) & (dist_river < river_w + 0.065 + 0.02 * noise_a)
        open_water = dist_river < river_w + 0.015 * noise_b
        cls[wetland_zone] = 3
        cls[open_water] = 1
        cls[settlement] = 5

        moisture = (river_w - 0.06) * 8.0
        frames.append((acq_date, moisture, cloud_pct, note, cls))
    return frames


def _build_sundarbans_frames(
    rng: np.random.Generator,
) -> List[Tuple[str, float, float, str, np.ndarray]]:
    """Sundarbans Mangrove Delta (Bangladesh): tidal channels, cyclone surge & mangrove edges."""
    yy, xx = np.mgrid[0:GRID_ROWS, 0:GRID_COLS]
    xn = xx / GRID_COLS
    yn = yy / GRID_ROWS

    noise_a = _smooth_noise(rng, (GRID_ROWS, GRID_COLS), sigma=7.5)
    noise_b = _smooth_noise(rng, (GRID_ROWS, GRID_COLS), sigma=3.0)

    # Branching estuarine channels
    ch_main = np.abs(xn - (0.50 + 0.09 * np.sin(2.2 * np.pi * yn) + 0.03 * noise_a))
    ch_west = np.abs((xn - 0.25) - 0.14 * (yn - 0.3) + 0.03 * noise_b)
    ch_east = np.abs((xn - 0.78) + 0.16 * (yn - 0.4) + 0.03 * noise_b)
    channel_dist = np.minimum(ch_main, np.minimum(ch_west, ch_east))

    bay_edge = yn > (0.84 + 0.04 * np.sin(3 * np.pi * xn) + 0.02 * noise_a)
    island_elev = channel_dist * 3.2 + 0.08 * noise_a
    northern_farms = (yn < 0.22 + 0.04 * noise_a) & (channel_dist > 0.07)
    port_cluster = ((xn - 0.35) ** 2 + (yn - 0.12) ** 2) < 0.0025

    schedule = [
        ("2026-01-12", 0.13, 18.0, "Calm dry-season tide: distinct mangrove islands and tidal channels."),
        ("2026-02-24", 0.16, 26.0, "Spring tide raises channel water levels along inner mangrove fringes."),
        ("2026-04-06", 0.20, 62.0, "Pre-monsoon tide widens estuaries; northern embankment fields saturate."),
        ("2026-05-19", 0.34, 99.0, "Cyclone storm surge (100% cloud cover): NISAR detects flooded mangroves."),
        ("2026-06-02", 0.28, 91.0, "Post-surge drainage leaves brackish wetlands across delta flats."),
        ("2026-06-28", 0.21, 84.0, "Monsoon high water persists; mangrove canopy edges show coastal erosion."),
        ("2026-07-18", 0.17, 79.0, "Tidal equilibrium stabilizes; flooded agriculture recovering inland."),
    ]

    frames: List[Tuple[str, float, float, str, np.ndarray]] = []
    for acq_date, tide, cloud_pct, note in schedule:
        cls = np.full((GRID_ROWS, GRID_COLS), 2, dtype=np.uint8)  # Mangrove tree canopy
        cls[northern_farms] = 4

        flooded_mangrove = (island_elev < tide + 0.11) & (~northern_farms)
        wet_farms = (island_elev < tide + 0.09) & northern_farms
        open_water = (island_elev < tide) | bay_edge

        cls[flooded_mangrove] = 3
        cls[wet_farms] = 3
        cls[open_water] = 1
        cls[port_cluster] = 5

        moisture = (tide - 0.13) * 4.5
        frames.append((acq_date, moisture, cloud_pct, note, cls))
    return frames


def _build_sacramento_frames(
    rng: np.random.Generator,
) -> List[Tuple[str, float, float, str, np.ndarray]]:
    """Yolo Bypass & Sacramento Valley (USA): managed flood bypass, rice fields & riparian trees."""
    yy, xx = np.mgrid[0:GRID_ROWS, 0:GRID_COLS]
    xn = xx / GRID_COLS
    yn = yy / GRID_ROWS

    noise_a = _smooth_noise(rng, (GRID_ROWS, GRID_COLS), sigma=6.0)
    bypass_corridor = np.abs(xn - (0.48 + 0.05 * yn)) < (0.14 + 0.02 * noise_a)
    sac_river = np.abs(xn - (0.78 + 0.03 * np.sin(3 * np.pi * yn))) < 0.028
    riparian_trees = (
        (np.abs(xn - (0.78 + 0.03 * np.sin(3 * np.pi * yn))) >= 0.028)
        & (np.abs(xn - (0.78 + 0.03 * np.sin(3 * np.pi * yn))) < 0.072)
    ) | (np.abs(xn - (0.31 + 0.05 * yn)) < 0.022)
    urban_east = (xn > 0.86) & (yn > 0.25) & (yn < 0.65)
    urban_west = ((xn - 0.12) ** 2 + (yn - 0.48) ** 2) < 0.0045
    urban_mask = urban_east | urban_west

    # Rectilinear rice/crop basins
    rice_west = (xn < 0.30) & (yn > 0.15) & (yn < 0.85) & (~urban_mask)
    fields_east = (xn > 0.63) & (~sac_river) & (~riparian_trees) & (~urban_mask)

    schedule = [
        ("2025-12-05", 0.05, 0.0, 35.0, "Early winter: bypass mostly dry farmland; Sacramento River steady."),
        ("2026-01-16", 0.85, 0.0, 88.0, "Atmospheric river opens weir: Yolo Bypass fills rapidly with floodwater."),
        ("2026-02-11", 0.95, 0.0, 72.0, "Peak winter wetland habitat across full bypass corridor."),
        ("2026-03-22", 0.45, 0.15, 41.0, "Bypass drains into saturated wildlife marshes; spring prep begins."),
        ("2026-04-29", 0.10, 0.78, 18.0, "Bypass dries while western rice paddies are intentionally flooded."),
        ("2026-05-24", 0.05, 0.55, 12.0, "Emergent rice stalks shift western paddies from specular water to wetland."),
        ("2026-06-15", 0.03, 0.15, 8.0, "Mid-summer green crops across valley; riparian tree lines crisp along river."),
    ]

    frames: List[Tuple[str, float, float, str, np.ndarray]] = []
    for acq_date, bypass_fill, rice_flood, cloud_pct, note in schedule:
        cls = np.zeros((GRID_ROWS, GRID_COLS), dtype=np.uint8)
        cls[rice_west | fields_east | bypass_corridor] = 4

        if bypass_fill > 0.55:
            cls[bypass_corridor] = 1
            cls[bypass_corridor & (noise_a > 0.6)] = 3
        elif bypass_fill > 0.20:
            cls[bypass_corridor] = 3
            cls[bypass_corridor & (noise_a < -0.5)] = 1

        if rice_flood > 0.65:
            cls[rice_west] = 1
        elif rice_flood > 0.30:
            cls[rice_west] = 3

        cls[riparian_trees] = 2
        cls[sac_river] = 1
        cls[urban_mask] = 5

        moisture = max(bypass_fill, rice_flood) * 1.4
        frames.append((acq_date, moisture, cloud_pct, note, cls))
    return frames


def _seed_demo_data(conn: sqlite3.Connection) -> None:
    """Populate SQLite with 4 realistic multi-date SAR regional archives."""
    rng = np.random.default_rng(2026)

    regions_meta = [
        (
            "elbe_flood",
            "Elbe River Floodplain (Lauenburg)",
            "Germany",
            "River Flooding & Polder Saturation",
            53.368,
            10.620,
            12,
            53.300,
            10.500,
            53.436,
            10.740,
            0.012,
            "Spring snowmelt and storm surge along the Elbe river corridor overflow low levees into "
            "agricultural polders. L-band SAR penetrates 90%+ storm clouds to separate specular open "
            "floodwater from double-bounce flooded riparian tree lines and bright urban towns.",
            "NISAR L-Band GCOV (simulated demo) · ASF Vertex · European Flood Context",
            _build_elbe_frames(rng),
        ),
        (
            "amazon_deforestation",
            "Tapajós / Santarém Frontier",
            "Brazil",
            "Tree-Line Retreat & Flooded Forest",
            -3.150,
            -55.020,
            12,
            -3.255,
            -55.135,
            -3.045,
            -54.905,
            0.025,
            "Persistent tropical cloud cover blinds optical sensors for months. Dual-polarization "
            "L-band radar tracks cross-polarized (HV) volume scattering to map fishbone tree-line "
            "retreat alongside seasonal várzea wetland flooding along the Tapajós tributary.",
            "NISAR L-Band · CONAE SAOCOM · NASA Earthdata Amazon Monitoring",
            _build_amazon_frames(rng),
        ),
        (
            "sundarbans_mangroves",
            "Sundarbans Mangrove Delta",
            "Bangladesh",
            "Cyclone Surge & Coastal Mangroves",
            22.150,
            89.420,
            12,
            22.055,
            89.305,
            22.245,
            89.535,
            0.021,
            "Tidal cyclones push brackish surge deep into the world's largest mangrove forest. "
            "HH backscatter brightens when water pools beneath mangrove trunks (double-bounce), "
            "while open tidal channels drop to mirror-like specular darkness.",
            "NISAR L-Band · CSA RADARSAT Constellation · ASF Vertex",
            _build_sundarbans_frames(rng),
        ),
        (
            "sacramento_wetlands",
            "Yolo Bypass & Sacramento Valley",
            "USA",
            "Managed Flood Bypass & Rice Agriculture",
            38.550,
            -121.620,
            12,
            38.465,
            -121.735,
            38.635,
            -121.505,
            0.016,
            "Tracks the seasonal transition of California's flood bypass and agricultural parcels: "
            "winter atmospheric rivers turn the bypass into open water and wildlife wetlands, followed "
            "by spring rice paddy inundation while Sacramento River tree lines remain stable.",
            "NISAR L-Band · UAVSAR California Calibration · NASA Earthdata",
            _build_sacramento_frames(rng),
        ),
    ]

    for (
        region_id,
        name,
        country,
        phenomenon,
        clat,
        clon,
        zoom,
        south,
        west,
        north,
        east,
        px_km2,
        story,
        sources,
        frames,
    ) in regions_meta:
        conn.execute(
            """
            INSERT INTO regions (
                region_id, name, country, phenomenon,
                center_lat, center_lon, zoom_start,
                south, west, north, east, pixel_area_km2,
                story, data_sources
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                region_id,
                name,
                country,
                phenomenon,
                clat,
                clon,
                zoom,
                south,
                west,
                north,
                east,
                px_km2,
                story,
                sources,
            ),
        )

        for idx, (acq_date, moisture, cloud_pct, note, true_cls) in enumerate(frames):
            hh_db, hv_db = synthesize_sar_from_landcover(true_cls, rng, moisture_shift_db=moisture)
            scene_id = f"{region_id}_{acq_date.replace('-', '')}"
            orbit = "Ascending" if idx % 2 == 0 else "Descending"
            conn.execute(
                """
                INSERT INTO sar_scenes (
                    scene_id, region_id, acq_date, sensor, orbit_direction,
                    polarizations, resolution_m, cloud_cover_optical_pct,
                    grid_rows, grid_cols, hh_db_blob, hv_db_blob, event_note
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    scene_id,
                    region_id,
                    acq_date,
                    "NISAR L-SAR",
                    orbit,
                    "HH / HV",
                    10.0,
                    cloud_pct,
                    GRID_ROWS,
                    GRID_COLS,
                    _encode_array(hh_db, np.float32),
                    _encode_array(hv_db, np.float32),
                    note,
                ),
            )


def list_regions(db_path: Path = DEFAULT_DB_PATH, presets_only: bool = False) -> List[RegionRecord]:
    init_database(db_path)
    with get_connection(db_path) as conn:
        if presets_only:
            placeholders = ",".join("?" for _ in PRESET_REGION_IDS)
            rows = conn.execute(
                f"SELECT * FROM regions WHERE region_id IN ({placeholders}) ORDER BY name",
                PRESET_REGION_IDS,
            ).fetchall()
        else:
            rows = conn.execute("SELECT * FROM regions ORDER BY name").fetchall()
    return [RegionRecord(**dict(r)) for r in rows]


def _upsert_layer_grid(
    conn: sqlite3.Connection,
    region_id: str,
    acq_date: str,
    layer_key: str,
    arr: np.ndarray,
) -> None:
    rows, cols = arr.shape
    conn.execute(
        """
        INSERT INTO layer_grids (region_id, acq_date, layer_key, grid_rows, grid_cols, values_blob)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(region_id, acq_date, layer_key) DO UPDATE SET
            grid_rows = excluded.grid_rows,
            grid_cols = excluded.grid_cols,
            values_blob = excluded.values_blob
        """,
        (region_id, acq_date, layer_key, rows, cols, _encode_array(arr, np.float32)),
    )


def ensure_auxiliary_layers_for_region(region: RegionRecord, db_path: Path = DEFAULT_DB_PATH) -> None:
    """Populate static & multi-sensor layers for curated presets if not yet materialized."""
    with get_connection(db_path) as conn:
        cnt = conn.execute(
            "SELECT COUNT(*) FROM layer_grids WHERE region_id = ?", (region.region_id,)
        ).fetchone()[0]
        if cnt > 0:
            return

        scenes = conn.execute(
            "SELECT acq_date, grid_rows, grid_cols, hh_db_blob, hv_db_blob FROM sar_scenes WHERE region_id = ? ORDER BY acq_date",
            (region.region_id,),
        ).fetchall()
        if not scenes:
            return

        r0, c0 = int(scenes[0]["grid_rows"]), int(scenes[0]["grid_cols"])
        src = WorldLBandDataSource()
        static_terrain = src.fetch_static_terrain(
            region.south, region.west, region.north, region.east, r0, c0, region.resolution_m
        )
        for key, arr in static_terrain.items():
            _upsert_layer_grid(conn, region.region_id, "static", key, arr)

        for row in scenes:
            shape = (int(row["grid_rows"]), int(row["grid_cols"]))
            hh = _decode_array(row["hh_db_blob"], shape, np.float32)
            hv = _decode_array(row["hv_db_blob"], shape, np.float32)
            seed_bytes = hashlib.sha256(f"{region.region_id}_{row['acq_date']}".encode()).digest()
            rng = np.random.default_rng(int.from_bytes(seed_bytes[:8], "little"))

            vv_db = (hh - 0.8 + rng.normal(0.0, 0.4, size=shape)).astype(np.float32)
            vh_db = (hv - 1.1 + rng.normal(0.0, 0.4, size=shape)).astype(np.float32)
            vh_vv = (vh_db - vv_db).astype(np.float32)
            # Physics-grounded coherence: low for smooth specular water (very low hh) & canopy, high for urban (bright hh)
            coh = np.clip(0.55 + 0.03 * hh - 0.015 * np.maximum(hv + 18.0, 0.0), 0.06, 0.96)
            coh = gaussian_filter(coh.astype(np.float32), sigma=0.75, mode="reflect").astype(np.float32)
            dswx = np.where(hh < -15.2, 1.0, np.where((hh > -6.0) & (hh - hv > 7.5), 2.0, 0.0)).astype(np.float32)

            dt = row["acq_date"]
            _upsert_layer_grid(conn, region.region_id, dt, "opera_vv_db", vv_db)
            _upsert_layer_grid(conn, region.region_id, dt, "opera_vh_db", vh_db)
            _upsert_layer_grid(conn, region.region_id, dt, "opera_vh_vv_db", vh_vv)
            _upsert_layer_grid(conn, region.region_id, dt, "opera_dswx_wtr", dswx)
            _upsert_layer_grid(conn, region.region_id, dt, "nisar_hhhh_db", hh)
            _upsert_layer_grid(conn, region.region_id, dt, "nisar_hvhv_db", hv)
            _upsert_layer_grid(conn, region.region_id, dt, "coherence_hh", coh)


def ensure_multiyear_dates_for_region(region: RegionRecord, db_path: Path = DEFAULT_DB_PATH) -> None:
    """Backfill full 2016–2026 Sentinel-1 OPERA + NISAR span for simulated regions on demand."""
    if region.source_type != "simulated":
        return

    with get_connection(db_path) as conn:
        rows = conn.execute(
            """
            SELECT scene_id, acq_date, grid_rows, grid_cols, hh_db_blob, hv_db_blob
            FROM sar_scenes
            WHERE region_id = ?
            ORDER BY acq_date ASC
            """,
            (region.region_id,),
        ).fetchall()
        if not rows:
            return

        existing_dates = {str(r["acq_date"]) for r in rows}
        min_existing = min(existing_dates)
        max_existing = max(existing_dates)
        if min_existing <= "2016-12-31":
            return

        r0, c0 = int(rows[0]["grid_rows"]), int(rows[0]["grid_cols"])
        shape = (r0, c0)

        # Decode dryest & wettest seeded frames for geometry-preserving seasonal interpolation
        decoded_scenes: List[Tuple[float, np.ndarray, np.ndarray]] = []
        for r in rows:
            hh_arr = _decode_array(r["hh_db_blob"], shape, np.float32)
            hv_arr = _decode_array(r["hv_db_blob"], shape, np.float32)
            water_frac = float(np.mean(hh_arr < -15.0))
            decoded_scenes.append((water_frac, hh_arr, hv_arr))
        decoded_scenes.sort(key=lambda item: item[0])
        _, hh_dry, hv_dry = decoded_scenes[0]
        _, hh_wet, hv_wet = decoded_scenes[-1]

        src = WorldLBandDataSource()
        for idx, (acq_date, stage, cloud_pct, note) in enumerate(DEFAULT_WORLD_TIMELINE):
            if acq_date in existing_dates:
                continue
            # Keep curated 2026 preset peaks intact; backfill years before & after preset window
            if min_existing <= acq_date <= max_existing:
                continue

            if region.region_id.startswith("aoi_"):
                hh_new, hv_new, cloud_pct, note, aux = src.fetch_temporal_layers(
                    region.south, region.west, region.north, region.east, acq_date, grid_rows=r0, grid_cols=c0
                )
            else:
                seed_bytes = hashlib.sha256(f"{region.region_id}_{acq_date}".encode()).digest()
                rng = np.random.default_rng(int.from_bytes(seed_bytes[:8], "little"))
                alpha = float(np.clip((stage - 0.11) / 0.30, 0.02, 0.96))
                noise_hh = gaussian_filter(rng.normal(0.0, 0.45, size=shape).astype(np.float32), sigma=0.55)
                noise_hv = gaussian_filter(rng.normal(0.0, 0.45, size=shape).astype(np.float32), sigma=0.55)
                hh_new = ((1.0 - alpha) * hh_dry + alpha * hh_wet + noise_hh).astype(np.float32)
                hv_new = ((1.0 - alpha) * hv_dry + alpha * hv_wet + noise_hv).astype(np.float32)
                vv_db = (hh_new - 0.8 + rng.normal(0.0, 0.35, size=shape)).astype(np.float32)
                vh_db = (hv_new - 1.1 + rng.normal(0.0, 0.35, size=shape)).astype(np.float32)
                vh_vv = (vh_db - vv_db).astype(np.float32)
                coh = np.clip(0.55 + 0.03 * hh_new - 0.015 * np.maximum(hv_new + 18.0, 0.0), 0.06, 0.96)
                coh = gaussian_filter(coh.astype(np.float32), sigma=0.75, mode="reflect").astype(np.float32)
                dswx = np.where(
                    hh_new < -15.2, 1.0, np.where((hh_new > -6.0) & (hh_new - hv_new > 7.5), 2.0, 0.0)
                ).astype(np.float32)
                aux = {
                    "opera_vv_db": vv_db,
                    "opera_vh_db": vh_db,
                    "opera_vh_vv_db": vh_vv,
                    "opera_dswx_wtr": dswx,
                    "nisar_hhhh_db": hh_new,
                    "nisar_hvhv_db": hv_new,
                    "coherence_hh": coh,
                }

            sensor = "NISAR L-SAR" if acq_date >= "2025-11-01" else "OPERA RTC-S1 C-SAR"
            pols = "HH / HV" if acq_date >= "2025-11-01" else "VV / VH"
            scene_id = f"{region.region_id}_{acq_date.replace('-', '')}"
            orbit = "Ascending" if idx % 2 == 0 else "Descending"
            conn.execute(
                """
                INSERT OR IGNORE INTO sar_scenes (
                    scene_id, region_id, acq_date, sensor, orbit_direction,
                    polarizations, resolution_m, cloud_cover_optical_pct,
                    grid_rows, grid_cols, hh_db_blob, hv_db_blob, event_note
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    scene_id,
                    region.region_id,
                    acq_date,
                    sensor,
                    orbit,
                    pols,
                    region.resolution_m,
                    cloud_pct,
                    r0,
                    c0,
                    _encode_array(hh_new, np.float32),
                    _encode_array(hv_new, np.float32),
                    note,
                ),
            )
            for lkey, larr in aux.items():
                _upsert_layer_grid(conn, region.region_id, acq_date, lkey, larr)


def list_region_layer_keys(region_id: str, db_path: Path = DEFAULT_DB_PATH) -> List[str]:
    """Return sorted distinct auxiliary `layer_key` names stored for `region_id`."""
    with get_connection(db_path) as conn:
        rows = conn.execute(
            "SELECT DISTINCT layer_key FROM layer_grids WHERE region_id = ? ORDER BY layer_key",
            (region_id,),
        ).fetchall()
    return [str(r["layer_key"]) for r in rows]


def get_layer_grids_for_region(
    region_id: str,
    layer_key: str,
    db_path: Path = DEFAULT_DB_PATH,
) -> Dict[str, np.ndarray]:
    """Load `{acq_date_or_static: float32_2d_array}` for a given region & `layer_key`."""
    with get_connection(db_path) as conn:
        rows = conn.execute(
            """
            SELECT acq_date, grid_rows, grid_cols, values_blob
            FROM layer_grids
            WHERE region_id = ? AND layer_key = ?
            """,
            (region_id, layer_key),
        ).fetchall()
    return {
        str(r["acq_date"]): _decode_array(r["values_blob"], (int(r["grid_rows"]), int(r["grid_cols"])), np.float32)
        for r in rows
    }


def ensure_custom_aoi_region(
    south: float,
    west: float,
    north: float,
    east: float,
    resolution_m: float = 30.0,
    label: Optional[str] = None,
    db_path: Path = DEFAULT_DB_PATH,
) -> RegionRecord:
    """
    Resolve an arbitrary dragged/resized observation box `(south, west, north, east)`
    at target `resolution_m` to an existing preset or persist a custom AOI in SQLite.
    """
    init_database(db_path)
    s_in, n_in = sorted([float(south), float(north)])
    w_in, e_in = sorted([float(west), float(east)])
    s, n = max(-84.0, round(s_in, 4)), min(84.0, round(n_in, 4))
    w, e = max(-179.9, round(w_in, 4)), min(179.9, round(e_in, 4))
    if n - s < 0.005:
        n = min(84.0, s + 0.005)
    if e - w < 0.005:
        e = min(179.9, w + 0.005)

    res_m = max(10.0, round(float(resolution_m), 1))

    # Match curated presets at default 30m resolution cleanly
    if abs(res_m - 30.0) < 0.5:
        for preset in list_regions(db_path, presets_only=True):
            if (
                abs(preset.south - s) < 2e-3
                and abs(preset.west - w) < 2e-3
                and abs(preset.north - n) < 2e-3
                and abs(preset.east - e) < 2e-3
            ):
                ensure_auxiliary_layers_for_region(preset, db_path=db_path)
                return preset

    bbox_key = f"{s:.4f},{w:.4f},{n:.4f},{e:.4f}"
    digest = hashlib.sha1(bbox_key.encode()).hexdigest()[:10]
    region_id = f"aoi_{digest}" if abs(res_m - 30.0) < 0.5 else f"aoi_{digest}_{int(round(res_m))}m"

    with get_connection(db_path) as conn:
        existing = conn.execute(
            "SELECT * FROM regions WHERE region_id = ?", (region_id,)
        ).fetchone()
        if existing is not None:
            rec = RegionRecord(**dict(existing))
            ensure_auxiliary_layers_for_region(rec, db_path=db_path)
            return rec

        center_lat = round(0.5 * (s + n), 4)
        center_lon = round(0.5 * (w + e), 4)
        clean_label = (label or "").strip() or f"Custom Observation ({center_lat:.2f}°, {center_lon:.2f}°)"
        rows, cols = compute_grid_shape(s, w, n, e, resolution_m=res_m, max_dim=224, min_dim=48)
        px_km2 = compute_geodesic_pixel_area_km2(s, w, n, e, rows, cols)

        conn.execute(
            """
            INSERT INTO regions (
                region_id, name, country, phenomenon,
                center_lat, center_lon, zoom_start,
                south, west, north, east, pixel_area_km2,
                story, data_sources, resolution_m, source_type
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                region_id,
                clean_label,
                "Custom AOI",
                "Interactive Area-of-Interest Analysis",
                center_lat,
                center_lon,
                12,
                s,
                w,
                n,
                e,
                px_km2,
                (
                    f"Dynamic multi-sensor & neural-network inference over bounding box "
                    f"[{s:.3f}, {w:.3f}] to [{n:.3f}, {e:.3f}] @ {res_m:.0f}m target spacing."
                ),
                f"Interactive AOI @ {res_m:.0f}m · Multi-Sensor Model (Ready for NASA Earthdata Fetch)",
                res_m,
                "simulated",
            ),
        )

        source = WorldLBandDataSource()
        static_terrain = source.fetch_static_terrain(s, w, n, e, rows, cols, res_m)
        for key, arr in static_terrain.items():
            _upsert_layer_grid(conn, region_id, "static", key, arr)

        for idx, acq_date in enumerate(list_default_timeline_dates()):
            hh_db, hv_db, cloud_pct, note, aux = source.fetch_temporal_layers(
                s, w, n, e, acq_date, grid_rows=rows, grid_cols=cols
            )
            scene_id = f"{region_id}_{acq_date.replace('-', '')}"
            orbit = "Ascending" if idx % 2 == 0 else "Descending"
            conn.execute(
                """
                INSERT INTO sar_scenes (
                    scene_id, region_id, acq_date, sensor, orbit_direction,
                    polarizations, resolution_m, cloud_cover_optical_pct,
                    grid_rows, grid_cols, hh_db_blob, hv_db_blob, event_note
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    scene_id,
                    region_id,
                    acq_date,
                    "NISAR L-SAR",
                    orbit,
                    "HH / HV",
                    res_m,
                    cloud_pct,
                    rows,
                    cols,
                    _encode_array(hh_db, np.float32),
                    _encode_array(hv_db, np.float32),
                    note,
                ),
            )
            for lkey, larr in aux.items():
                _upsert_layer_grid(conn, region_id, acq_date, lkey, larr)

        saved = conn.execute(
            "SELECT * FROM regions WHERE region_id = ?", (region_id,)
        ).fetchone()
    return RegionRecord(**dict(saved))


def import_processed_run_to_db(
    run_dir: Path,
    region_name: Optional[str] = None,
    db_path: Path = DEFAULT_DB_PATH,
) -> RegionRecord:
    """Ingest a real NASA Earthdata `data/processed/<run_name>/` NetCDF run into SQLite."""
    from src.earthdata_pipeline import load_processed_run_for_db

    init_database(db_path)
    bundle_data = load_processed_run_for_db(run_dir)
    run_name = bundle_data["run_name"]
    region_id = f"nasa_{run_name}"

    s, w, n, e = bundle_data["south"], bundle_data["west"], bundle_data["north"], bundle_data["east"]
    rows, cols = bundle_data["rows"], bundle_data["cols"]
    res_m = bundle_data["resolution_m"]
    center_lat = round(0.5 * (s + n), 4)
    center_lon = round(0.5 * (w + e), 4)
    px_km2 = compute_geodesic_pixel_area_km2(s, w, n, e, rows, cols)
    w_km, h_km = compute_box_dimensions_km(s, w, n, e)
    display_name = (region_name or "").strip() or f"NASA Earthdata · {run_name}"
    ds_label = ", ".join(bundle_data["datasets"])

    with get_connection(db_path) as conn:
        # Remove prior state for this run_id so re-downloads replace cleanly
        conn.execute(
            "DELETE FROM model_classifications WHERE scene_id IN (SELECT scene_id FROM sar_scenes WHERE region_id = ?)",
            (region_id,),
        )
        conn.execute("DELETE FROM layer_grids WHERE region_id = ?", (region_id,))
        conn.execute("DELETE FROM sar_scenes WHERE region_id = ?", (region_id,))
        conn.execute("DELETE FROM regions WHERE region_id = ?", (region_id,))

        conn.execute(
            """
            INSERT INTO regions (
                region_id, name, country, phenomenon,
                center_lat, center_lon, zoom_start,
                south, west, north, east, pixel_area_km2,
                story, data_sources, resolution_m, source_type
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                region_id,
                display_name,
                f"NASA Earthdata ({w_km:.1f}×{h_km:.1f} km)",
                "Aligned NASA Multi-Sensor Archive",
                center_lat,
                center_lon,
                12,
                s,
                w,
                n,
                e,
                px_km2,
                f"Imported aligned NetCDF-4 cubes ({ds_label}) @ {res_m:.0f}m UTM spacing.",
                f"NASA Earthdata ({ds_label}) @ {res_m:.0f}m",
                res_m,
                "nasa_earthdata",
            ),
        )

        for key, arr in bundle_data["static_layers"].items():
            _upsert_layer_grid(conn, region_id, "static", key, arr)

        for idx, rec in enumerate(bundle_data["timeline"]):
            dt = rec["acq_date"]
            scene_id = f"{region_id}_{dt.replace('-', '')}"
            conn.execute(
                """
                INSERT INTO sar_scenes (
                    scene_id, region_id, acq_date, sensor, orbit_direction,
                    polarizations, resolution_m, cloud_cover_optical_pct,
                    grid_rows, grid_cols, hh_db_blob, hv_db_blob, event_note
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    scene_id,
                    region_id,
                    dt,
                    rec["sensor"],
                    "Ascending" if idx % 2 == 0 else "Descending",
                    rec["polarizations"],
                    res_m,
                    0.0,
                    rows,
                    cols,
                    _encode_array(rec["hh_db"], np.float32),
                    _encode_array(rec["hv_db"], np.float32),
                    rec["event_note"],
                ),
            )
            for lkey, larr in rec["aux_layers"].items():
                _upsert_layer_grid(conn, region_id, dt, lkey, larr)

        saved = conn.execute("SELECT * FROM regions WHERE region_id = ?", (region_id,)).fetchone()
    return RegionRecord(**dict(saved))


def get_region(region_id: str, db_path: Path = DEFAULT_DB_PATH) -> RegionRecord:
    init_database(db_path)
    with get_connection(db_path) as conn:
        row = conn.execute("SELECT * FROM regions WHERE region_id = ?", (region_id,)).fetchone()
    if row is None:
        raise KeyError(f"Region '{region_id}' not found in database.")
    return RegionRecord(**dict(row))


def list_region_dates(region_id: str, db_path: Path = DEFAULT_DB_PATH) -> List[str]:
    init_database(db_path)
    with get_connection(db_path) as conn:
        rows = conn.execute(
            "SELECT acq_date FROM sar_scenes WHERE region_id = ? ORDER BY acq_date ASC",
            (region_id,),
        ).fetchall()
    return [r["acq_date"] for r in rows]


def get_scenes_in_timeframe(
    region_id: str,
    start_date: str,
    end_date: str,
    db_path: Path = DEFAULT_DB_PATH,
) -> List[SARSceneRecord]:
    init_database(db_path)
    with get_connection(db_path) as conn:
        rows = conn.execute(
            """
            SELECT * FROM sar_scenes
            WHERE region_id = ? AND acq_date >= ? AND acq_date <= ?
            ORDER BY acq_date ASC
            """,
            (region_id, start_date, end_date),
        ).fetchall()
    scenes: List[SARSceneRecord] = []
    for r in rows:
        shape = (r["grid_rows"], r["grid_cols"])
        scenes.append(
            SARSceneRecord(
                scene_id=r["scene_id"],
                region_id=r["region_id"],
                acq_date=r["acq_date"],
                sensor=r["sensor"],
                orbit_direction=r["orbit_direction"],
                polarizations=r["polarizations"],
                resolution_m=r["resolution_m"],
                cloud_cover_optical_pct=r["cloud_cover_optical_pct"],
                grid_rows=r["grid_rows"],
                grid_cols=r["grid_cols"],
                hh_db=_decode_array(r["hh_db_blob"], shape, np.float32),
                hv_db=_decode_array(r["hv_db_blob"], shape, np.float32),
                event_note=r["event_note"],
            )
        )
    return scenes


def get_classification(
    scene_id: str,
    model_id: str,
    db_path: Path = DEFAULT_DB_PATH,
) -> Optional[StoredClassification]:
    with get_connection(db_path) as conn:
        row = conn.execute(
            """
            SELECT * FROM model_classifications
            WHERE scene_id = ? AND model_id = ?
            """,
            (scene_id, model_id),
        ).fetchone()
    if row is None:
        return None
    shape = (row["grid_rows"], row["grid_cols"])
    areas = {int(k): float(v) for k, v in json.loads(row["areas_km2_json"]).items()}
    return StoredClassification(
        scene_id=row["scene_id"],
        model_id=row["model_id"],
        class_mask=_decode_array(row["class_mask_blob"], shape, np.uint8),
        confidence=_decode_array(row["confidence_blob"], shape, np.float32),
        areas_km2=areas,
        inference_ms=float(row["inference_ms"]),
        created_at=row["created_at"],
    )


def save_classification(
    scene_id: str,
    model_id: str,
    class_mask: np.ndarray,
    confidence: np.ndarray,
    areas_km2: Dict[int, float],
    inference_ms: float,
    db_path: Path = DEFAULT_DB_PATH,
) -> StoredClassification:
    created_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    rows, cols = class_mask.shape
    areas_json = json.dumps({str(k): round(float(v), 3) for k, v in areas_km2.items()})
    with get_connection(db_path) as conn:
        conn.execute(
            """
            INSERT INTO model_classifications (
                scene_id, model_id, grid_rows, grid_cols,
                class_mask_blob, confidence_blob, areas_km2_json,
                inference_ms, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(scene_id, model_id) DO UPDATE SET
                grid_rows = excluded.grid_rows,
                grid_cols = excluded.grid_cols,
                class_mask_blob = excluded.class_mask_blob,
                confidence_blob = excluded.confidence_blob,
                areas_km2_json = excluded.areas_km2_json,
                inference_ms = excluded.inference_ms,
                created_at = excluded.created_at
            """,
            (
                scene_id,
                model_id,
                rows,
                cols,
                _encode_array(class_mask, np.uint8),
                _encode_array(confidence, np.float32),
                areas_json,
                float(inference_ms),
                created_at,
            ),
        )
    return StoredClassification(
        scene_id=scene_id,
        model_id=model_id,
        class_mask=class_mask,
        confidence=confidence,
        areas_km2=areas_km2,
        inference_ms=inference_ms,
        created_at=created_at,
    )


def clear_classifications(
    region_id: Optional[str] = None,
    model_id: Optional[str] = None,
    db_path: Path = DEFAULT_DB_PATH,
) -> int:
    """Delete cached classification results so the neural network re-runs fresh."""
    clauses: List[str] = []
    params: List[Any] = []
    if region_id is not None:
        clauses.append("scene_id IN (SELECT scene_id FROM sar_scenes WHERE region_id = ?)")
        params.append(region_id)
    if model_id is not None:
        clauses.append("model_id = ?")
        params.append(model_id)

    where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    with get_connection(db_path) as conn:
        cur = conn.execute(f"DELETE FROM model_classifications {where_sql}", params)
        return int(cur.rowcount)


def insert_sar_scene(
    region_id: str,
    acq_date: str,
    hh_db: np.ndarray,
    hv_db: np.ndarray,
    sensor: str = "NISAR L-SAR",
    orbit_direction: str = "Ascending",
    polarizations: str = "HH / HV",
    resolution_m: float = 10.0,
    cloud_cover_optical_pct: float = 0.0,
    event_note: str = "Custom ingested SAR scene.",
    db_path: Path = DEFAULT_DB_PATH,
) -> str:
    """Ingest a new co-registered SAR observation into SQLite (invalidating stale model cache)."""
    init_database(db_path)
    if hh_db.shape != hv_db.shape:
        raise ValueError(f"HH shape {hh_db.shape} != HV shape {hv_db.shape}.")
    rows, cols = hh_db.shape
    scene_id = f"{region_id}_{acq_date.replace('-', '')}_{sensor.lower().replace(' ', '_')}"
    with get_connection(db_path) as conn:
        conn.execute(
            """
            INSERT INTO sar_scenes (
                scene_id, region_id, acq_date, sensor, orbit_direction,
                polarizations, resolution_m, cloud_cover_optical_pct,
                grid_rows, grid_cols, hh_db_blob, hv_db_blob, event_note
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(region_id, acq_date, sensor) DO UPDATE SET
                hh_db_blob = excluded.hh_db_blob,
                hv_db_blob = excluded.hv_db_blob,
                event_note = excluded.event_note
            """,
            (
                scene_id,
                region_id,
                acq_date,
                sensor,
                orbit_direction,
                polarizations,
                resolution_m,
                cloud_cover_optical_pct,
                rows,
                cols,
                _encode_array(hh_db, np.float32),
                _encode_array(hv_db, np.float32),
                event_note,
            ),
        )
        conn.execute("DELETE FROM model_classifications WHERE scene_id = ?", (scene_id,))
    return scene_id


def get_database_stats(db_path: Path = DEFAULT_DB_PATH) -> Dict[str, Any]:
    """Return high-level row counts and database file size for inspection."""
    init_database(db_path)
    with get_connection(db_path) as conn:
        n_regions = conn.execute("SELECT COUNT(*) FROM regions").fetchone()[0]
        n_scenes = conn.execute("SELECT COUNT(*) FROM sar_scenes").fetchone()[0]
        n_preds = conn.execute("SELECT COUNT(*) FROM model_classifications").fetchone()[0]
        scenes_rows = conn.execute(
            """
            SELECT s.scene_id, r.name AS region_name, s.acq_date, s.sensor,
                   s.polarizations, s.cloud_cover_optical_pct, s.event_note,
                   COUNT(mc.model_id) AS cached_models
            FROM sar_scenes s
            JOIN regions r ON r.region_id = s.region_id
            LEFT JOIN model_classifications mc ON mc.scene_id = s.scene_id
            GROUP BY s.scene_id
            ORDER BY r.name, s.acq_date
            """
        ).fetchall()
    size_kb = round(db_path.stat().st_size / 1024.0, 1) if db_path.exists() else 0.0
    return {
        "db_path": str(db_path),
        "size_kb": size_kb,
        "regions": int(n_regions),
        "sar_scenes": int(n_scenes),
        "classifications": int(n_preds),
        "scenes_table": [dict(r) for r in scenes_rows],
    }
