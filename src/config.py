"""Domain configuration, surface element definitions, and color palettes."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass(frozen=True)
class SurfaceClass:
    """Metadata and visual styling for a classified Earth surface element."""

    id: int
    key: str
    label: str
    short_label: str
    color_hex: str
    rgba: Tuple[int, int, int, int]
    radar_signature: str
    description: str


SURFACE_CLASSES: Dict[int, SurfaceClass] = {
    0: SurfaceClass(
        id=0,
        key="background",
        label="Bare Ground / Unclassified",
        short_label="Bare Ground",
        color_hex="#64748B",
        rgba=(100, 116, 139, 0),  # Transparent on map so basemap stays visible
        radar_signature="Low-to-moderate HH & HV depending on soil roughness",
        description="Exposed soil, dry riverbeds, or background terrain without dominant cover.",
    ),
    1: SurfaceClass(
        id=1,
        key="water",
        label="Water & Flood Extent",
        short_label="Water",
        color_hex="#0EA5E9",
        rgba=(14, 165, 233, 215),
        radar_signature="Very low HH (-15 to -23 dB) & HV (-22 to -28 dB) via specular reflection",
        description="Open rivers, lakes, tidal channels, and floodwater acting as radar mirrors.",
    ),
    2: SurfaceClass(
        id=2,
        key="treeline",
        label="Tree Lines & Forest Canopy",
        short_label="Tree Lines",
        color_hex="#22C55E",
        rgba=(34, 197, 94, 210),
        radar_signature="Strong HV (-9 to -14 dB) & moderate HH (-5 to -9 dB) via volume scattering",
        description="Riparian tree belts, dense canopy, mangroves, and forest boundaries.",
    ),
    3: SurfaceClass(
        id=3,
        key="wetland",
        label="Wetlands & Flooded Vegetation",
        short_label="Wetlands",
        color_hex="#14B8A6",
        rgba=(20, 184, 166, 205),
        radar_signature="High HH (-2 to -6 dB) with large HH-HV gap via trunk-water double-bounce",
        description="Inundated marshes, flooded forests, and saturated riparian zones.",
    ),
    4: SurfaceClass(
        id=4,
        key="cropland",
        label="Cropland & Soil Moisture",
        short_label="Cropland",
        color_hex="#EAB308",
        rgba=(234, 179, 8, 180),
        radar_signature="Moderate HH (-9 to -14 dB) rising with moisture; low-to-moderate HV",
        description="Agricultural fields, pastureland, and moisture-retaining floodplains.",
    ),
    5: SurfaceClass(
        id=5,
        key="urban",
        label="Urban & Infrastructure",
        short_label="Urban",
        color_hex="#F97316",
        rgba=(249, 115, 22, 215),
        radar_signature="Very bright HH (-3 to +3 dB) & high local texture via corner reflectors",
        description="Buildings, levees, bridges, and built surfaces showing persistent double-bounce.",
    ),
}

SELECTABLE_CLASSES = [c for cid, c in SURFACE_CLASSES.items() if cid != 0]
CLASS_BY_KEY = {c.key: c for c in SURFACE_CLASSES.values()}

# Change detection palette (relative to start of chosen timeframe)
CHANGE_COLORS = {
    "gained": (244, 63, 94, 225),      # #F43F5E — newly appeared element
    "lost": (250, 204, 21, 225),       # #FACC15 — disappeared since baseline
    "persisting": (56, 189, 248, 125), # #38BDF8 — unchanged presence
}

CHANGE_HEX = {
    "gained": "#F43F5E",
    "lost": "#FACC15",
    "persisting": "#38BDF8",
}

BASEMAPS = {
    "Esri Satellite Imagery": "Esri.WorldImagery",
    "Esri Light Gray Canvas": "Esri.WorldGrayCanvas",
    "OpenStreetMap": "OpenStreetMap",
    "Esri Topographic": "Esri.WorldTopoMap",
}


@dataclass(frozen=True)
class LayerMode:
    """Visualization mode metadata mapping frontend filter options to NASA/NN datasets."""

    mode_id: str
    label: str
    short_label: str
    category: str
    dataset_key: str
    layer_key: str
    unit: str
    colormap: str
    vmin: float
    vmax: float
    is_static: bool
    description: str


LAYER_MODES: Dict[str, LayerMode] = {
    "elements": LayerMode(
        mode_id="elements",
        label="Surface Elements (AI)",
        short_label="Elements",
        category="AI Analysis",
        dataset_key="classification",
        layer_key="class_mask",
        unit="class",
        colormap="classes",
        vmin=0.0,
        vmax=5.0,
        is_static=False,
        description="Neural network land-cover & surface element segmentation.",
    ),
    "change": LayerMode(
        mode_id="change",
        label="Temporal Changes (AI)",
        short_label="Changes",
        category="AI Analysis",
        dataset_key="classification",
        layer_key="change_mask",
        unit="change",
        colormap="change",
        vmin=0.0,
        vmax=3.0,
        is_static=False,
        description="Gained, lost, and persisting surface elements vs. start date.",
    ),
    "backscatter": LayerMode(
        mode_id="backscatter",
        label="Dual-Pol False Color (HH/HV)",
        short_label="Dual-Pol RGB",
        category="NISAR L-SAR",
        dataset_key="nisar_gcov",
        layer_key="dual_pol",
        unit="dB RGB",
        colormap="rgb",
        vmin=-25.0,
        vmax=0.0,
        is_static=False,
        description="UAVSAR-style RGB composite from co-pol (HH) & cross-pol (HV).",
    ),
    "elevation_m": LayerMode(
        mode_id="elevation_m",
        label="NASADEM Elevation (m)",
        short_label="Elevation",
        category="NASADEM Terrain",
        dataset_key="nasadem_static",
        layer_key="elevation_m",
        unit="m",
        colormap="terrain",
        vmin=0.0,
        vmax=300.0,
        is_static=True,
        description="Static digital elevation model height above sea level.",
    ),
    "slope_deg": LayerMode(
        mode_id="slope_deg",
        label="Terrain Slope (°)",
        short_label="Slope",
        category="NASADEM Terrain",
        dataset_key="nasadem_static",
        layer_key="slope_deg",
        unit="°",
        colormap="magma",
        vmin=0.0,
        vmax=35.0,
        is_static=True,
        description="Local topographic slope derived from NASADEM gradient.",
    ),
    "aspect_deg": LayerMode(
        mode_id="aspect_deg",
        label="Terrain Aspect (°)",
        short_label="Aspect",
        category="NASADEM Terrain",
        dataset_key="nasadem_static",
        layer_key="aspect_deg",
        unit="°",
        colormap="twilight",
        vmin=0.0,
        vmax=360.0,
        is_static=True,
        description="Compass orientation of terrain slope (0° North, clockwise).",
    ),
    "opera_vv_db": LayerMode(
        mode_id="opera_vv_db",
        label="OPERA S1 VV (dB)",
        short_label="OPERA VV",
        category="OPERA Sentinel-1",
        dataset_key="opera_rtc_s1",
        layer_key="opera_vv_db",
        unit="dB",
        colormap="gray",
        vmin=-25.0,
        vmax=0.0,
        is_static=False,
        description="Sentinel-1 C-band radiometrically terrain-corrected VV backscatter.",
    ),
    "opera_vh_db": LayerMode(
        mode_id="opera_vh_db",
        label="OPERA S1 VH (dB)",
        short_label="OPERA VH",
        category="OPERA Sentinel-1",
        dataset_key="opera_rtc_s1",
        layer_key="opera_vh_db",
        unit="dB",
        colormap="gray",
        vmin=-30.0,
        vmax=-5.0,
        is_static=False,
        description="Sentinel-1 C-band cross-polarized VH canopy volume backscatter.",
    ),
    "opera_vh_vv_db": LayerMode(
        mode_id="opera_vh_vv_db",
        label="OPERA S1 VH−VV (dB)",
        short_label="OPERA VH−VV",
        category="OPERA Sentinel-1",
        dataset_key="opera_rtc_s1",
        layer_key="opera_vh_vv_db",
        unit="dB",
        colormap="coolwarm",
        vmin=-14.0,
        vmax=0.0,
        is_static=False,
        description="Cross-pol minus co-pol ratio separating vegetation from open water.",
    ),
    "opera_dswx_wtr": LayerMode(
        mode_id="opera_dswx_wtr",
        label="OPERA DSWx-S1 Water",
        short_label="DSWx Water",
        category="OPERA Sentinel-1",
        dataset_key="opera_dswx_s1",
        layer_key="opera_dswx_wtr",
        unit="class",
        colormap="dswx",
        vmin=0.0,
        vmax=4.0,
        is_static=False,
        description="OPERA Dynamic Surface Water eXtent product layer.",
    ),
    "nisar_hhhh_db": LayerMode(
        mode_id="nisar_hhhh_db",
        label="NISAR GCOV HHHH (dB)",
        short_label="NISAR HHHH",
        category="NISAR L-SAR",
        dataset_key="nisar_gcov",
        layer_key="nisar_hhhh_db",
        unit="dB",
        colormap="gray",
        vmin=-25.0,
        vmax=0.0,
        is_static=False,
        description="NISAR L-band geocoded covariance co-pol backscatter intensity.",
    ),
    "nisar_hvhv_db": LayerMode(
        mode_id="nisar_hvhv_db",
        label="NISAR GCOV HVHV (dB)",
        short_label="NISAR HVHV",
        category="NISAR L-SAR",
        dataset_key="nisar_gcov",
        layer_key="nisar_hvhv_db",
        unit="dB",
        colormap="gray",
        vmin=-30.0,
        vmax=-5.0,
        is_static=False,
        description="NISAR L-band cross-pol canopy & structure sensitivity.",
    ),
    "coherence_hh": LayerMode(
        mode_id="coherence_hh",
        label="NISAR GUNW Coherence (HH)",
        short_label="Coherence",
        category="NISAR L-SAR",
        dataset_key="nisar_gunw",
        layer_key="coherence_hh",
        unit="coh",
        colormap="viridis",
        vmin=0.0,
        vmax=1.0,
        is_static=False,
        description="Interferometric phase coherence [0–1]: low over water/forest, high on stable surfaces.",
    ),
}

RESOLUTION_PRESETS_M = (15.0, 30.0, 60.0, 100.0, 150.0)

