"""Backend orchestration connecting SQLite SAR scenes, neural network inference, and map overlays."""
from __future__ import annotations

import base64
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib
import numpy as np
from scipy.ndimage import uniform_filter

from src.classifier import get_classifier
from src.config import CHANGE_COLORS, LAYER_MODES, SURFACE_CLASSES
from src.data_source import compute_box_dimensions_km
from src.database import (
    DEFAULT_DB_PATH,
    RegionRecord,
    SARSceneRecord,
    ensure_auxiliary_layers_for_region,
    ensure_custom_aoi_region,
    ensure_multiyear_dates_for_region,
    get_classification,
    get_layer_grids_for_region,
    get_region,
    get_scenes_in_timeframe,
    list_region_dates,
    list_region_layer_keys,
    save_classification,
)
from src.earthdata_pipeline import get_source_date_limits
from src.flood import compute_change_mask
from src.i18n import LAYER_DE, TRANSLATIONS


@dataclass
class ClassifiedFrame:
    scene: SARSceneRecord
    class_mask: np.ndarray
    confidence: np.ndarray
    areas_km2: Dict[int, float]
    inference_ms: float
    from_cache: bool


@dataclass
class TimeframeBundle:
    region: RegionRecord
    model_id: str
    frames: List[ClassifiedFrame]
    cache_hits: int
    newly_inferred: int
    total_inference_ms: float


def compute_class_areas_km2(class_mask: np.ndarray, pixel_area_km2: float) -> Dict[int, float]:
    """Compute covered area in km² for every class ID 0..5."""
    areas: Dict[int, float] = {}
    for cid in SURFACE_CLASSES:
        areas[cid] = round(float(np.sum(class_mask == cid) * pixel_area_km2), 3)
    return areas


def run_timeframe_pipeline(
    region_id: str,
    start_date: str,
    end_date: str,
    model_id: str = "sar_mlp_v1",
    force_recompute: bool = False,
    db_path: Path = DEFAULT_DB_PATH,
) -> TimeframeBundle:
    """
    Pull SAR scenes from SQLite for `[start_date, end_date]`, run neural network inference
    on any unclassified scenes (or all scenes if `force_recompute`), and persist outputs.
    """
    if start_date > end_date:
        start_date, end_date = end_date, start_date

    region = get_region(region_id, db_path=db_path)
    scenes = get_scenes_in_timeframe(region_id, start_date, end_date, db_path=db_path)
    classifier = get_classifier(model_id)

    frames: List[ClassifiedFrame] = []
    cache_hits = 0
    newly_inferred = 0
    total_ms = 0.0

    for scene in scenes:
        cached = None if force_recompute else get_classification(scene.scene_id, model_id, db_path=db_path)
        if cached is not None:
            cache_hits += 1
            frames.append(
                ClassifiedFrame(
                    scene=scene,
                    class_mask=cached.class_mask,
                    confidence=cached.confidence,
                    areas_km2=cached.areas_km2,
                    inference_ms=cached.inference_ms,
                    from_cache=True,
                )
            )
            total_ms += cached.inference_ms
            continue

        result = classifier.predict(scene.hh_db, scene.hv_db)
        areas_km2 = compute_class_areas_km2(result.class_mask, region.pixel_area_km2)
        save_classification(
            scene_id=scene.scene_id,
            model_id=model_id,
            class_mask=result.class_mask,
            confidence=result.confidence,
            areas_km2=areas_km2,
            inference_ms=result.inference_ms,
            db_path=db_path,
        )
        newly_inferred += 1
        total_ms += result.inference_ms
        frames.append(
            ClassifiedFrame(
                scene=scene,
                class_mask=result.class_mask,
                confidence=result.confidence,
                areas_km2=areas_km2,
                inference_ms=result.inference_ms,
                from_cache=False,
            )
        )

    return TimeframeBundle(
        region=region,
        model_id=model_id,
        frames=frames,
        cache_hits=cache_hits,
        newly_inferred=newly_inferred,
        total_inference_ms=round(total_ms, 2),
    )


def build_rgba_overlay(
    frame: ClassifiedFrame,
    selected_class_ids: Sequence[int],
    view_mode: str = "elements",
    baseline_frame: Optional[ClassifiedFrame] = None,
    sar_band: str = "False-Color Dual-Pol (HH/HV)",
    layer_grid: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Create an `(H, W, 4)` uint8 RGBA overlay for Folium/Leaflet `ImageOverlay`.

    `view_mode`:
      - "elements": renders selected surface classes in color, keeping unselected classes transparent.
      - "change": compares `frame` against `baseline_frame` for selected classes (Gained/Lost/Stable).
      - "backscatter": visualizes raw SAR radar backscatter (`HH`, `HV`, or dual-pol RGB composite).
      - any auxiliary `LAYER_MODES` id (e.g. `elevation_m`, `opera_vv_db`, `coherence_hh`) when `layer_grid` is supplied.
    """
    h, w = frame.class_mask.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    selected = [int(cid) for cid in selected_class_ids if cid != 0]

    if view_mode in LAYER_MODES and layer_grid is not None and view_mode not in ("elements", "change", "backscatter"):
        mode_def = LAYER_MODES[view_mode]
        arr = np.asarray(layer_grid, dtype=np.float32)
        valid = np.isfinite(arr)
        if not np.any(valid):
            return rgba
        vmin, vmax = mode_def.vmin, mode_def.vmax
        if view_mode == "elevation_m":
            vmin = float(np.nanpercentile(arr, 1))
            vmax = float(np.nanpercentile(arr, 99))
            if vmax - vmin < 1.0:
                vmax = vmin + 1.0
        norm = np.clip((np.nan_to_num(arr, nan=vmin) - vmin) / max(vmax - vmin, 1e-6), 0.0, 1.0)
        cmap_name = "Blues" if mode_def.colormap == "dswx" else mode_def.colormap
        rgb = (matplotlib.colormaps[cmap_name](norm)[..., :3] * 255).astype(np.uint8)
        rgba[..., :3] = rgb
        rgba[..., 3] = np.where(valid, 220, 0).astype(np.uint8)
        return rgba

    if view_mode == "backscatter":
        hh = frame.scene.hh_db
        hv = frame.scene.hv_db
        if sar_band == "HH (dB)":
            norm = np.clip((hh - (-23.0)) / 24.0, 0.0, 1.0)
            rgb = (matplotlib.colormaps["cividis"](norm)[..., :3] * 255).astype(np.uint8)
        elif sar_band == "HV (dB)":
            norm = np.clip((hv - (-27.0)) / 20.0, 0.0, 1.0)
            rgb = (matplotlib.colormaps["viridis"](norm)[..., :3] * 255).astype(np.uint8)
        else:
            # NASA/UAVSAR-style Dual-Pol False-Color: R=HH (double-bounce/roughness), G=HV (canopy), B=HH-HV
            r = np.clip((hh - (-21.0)) / 21.0, 0.0, 1.0)
            g = np.clip((hv - (-26.0)) / 18.0, 0.0, 1.0)
            b = np.clip(((hh - hv) - 1.0) / 13.0, 0.0, 1.0)
            rgb = (np.stack([r, g, b], axis=-1) * 255).astype(np.uint8)
        rgba[..., :3] = rgb
        rgba[..., 3] = 220
        return rgba

    if view_mode == "change" and baseline_frame is not None:
        chg = compute_change_mask(baseline_frame.class_mask, frame.class_mask, selected)
        rgba[chg == 1] = CHANGE_COLORS["persisting"]
        rgba[chg == 2] = CHANGE_COLORS["gained"]
        rgba[chg == 3] = CHANGE_COLORS["lost"]
        return rgba

    # Default: "elements"
    for cid in selected:
        mask = frame.class_mask == cid
        if np.any(mask):
            rgba[mask] = SURFACE_CLASSES[cid].rgba
    return rgba


def compute_change_summary(
    baseline_frame: ClassifiedFrame,
    frame: ClassifiedFrame,
    selected_class_ids: Sequence[int],
    pixel_area_km2: float,
) -> Dict[str, float]:
    """Return Gained, Lost, Persisting, and Net change area (km²) for selected elements."""
    chg = compute_change_mask(baseline_frame.class_mask, frame.class_mask, list(selected_class_ids))
    persisting = float(np.sum(chg == 1) * pixel_area_km2)
    gained = float(np.sum(chg == 2) * pixel_area_km2)
    lost = float(np.sum(chg == 3) * pixel_area_km2)
    return {
        "persisting_km2": round(persisting, 2),
        "gained_km2": round(gained, 2),
        "lost_km2": round(lost, 2),
        "net_km2": round(gained - lost, 2),
        "active_km2": round(persisting + gained, 2),
    }


def compute_transition_rows(
    baseline_frame: ClassifiedFrame,
    target_frame: ClassifiedFrame,
    pixel_area_km2: float,
    min_area_km2: float = 0.05,
) -> List[Dict[str, Any]]:
    """Compute land-cover transitions between baseline and target timestamp."""
    rows: List[Dict[str, Any]] = []
    base = baseline_frame.class_mask
    targ = target_frame.class_mask
    for from_id, from_cls in SURFACE_CLASSES.items():
        from_mask = base == from_id
        if not np.any(from_mask):
            continue
        for to_id, to_cls in SURFACE_CLASSES.items():
            if from_id == to_id:
                continue
            px = int(np.sum(from_mask & (targ == to_id)))
            area = px * pixel_area_km2
            if area >= min_area_km2:
                rows.append(
                    {
                        "From": from_cls.short_label,
                        "To": to_cls.short_label,
                        "Transition": f"{from_cls.short_label} -> {to_cls.short_label}",
                        "Area (km²)": round(area, 2),
                    }
                )
    rows.sort(key=lambda r: r["Area (km²)"], reverse=True)
    return rows


def extract_hotspot_markers(
    frame: ClassifiedFrame,
    baseline_frame: Optional[ClassifiedFrame],
    region: RegionRecord,
    selected_class_ids: Sequence[int],
) -> List[Dict[str, Any]]:
    """
    Find spatial hotspot pins (peak gain, peak loss, or highest concentration) for interactive popups.
    """
    selected = [int(cid) for cid in selected_class_ids if cid != 0]
    if not selected:
        return []

    h, w = frame.class_mask.shape
    markers: List[Dict[str, Any]] = []

    def _rc_to_latlon(r: int, c: int) -> Tuple[float, float]:
        lat = region.north - ((r + 0.5) / h) * (region.north - region.south)
        lon = region.west + ((c + 0.5) / w) * (region.east - region.west)
        return float(lat), float(lon)

    if baseline_frame is not None and frame.scene.acq_date != baseline_frame.scene.acq_date:
        chg = compute_change_mask(baseline_frame.class_mask, frame.class_mask, selected)
        for code, title, color in (
            (2, "Peak Element Gain Hotspot", "#F43F5E"),
            (3, "Peak Element Loss Hotspot", "#FACC15"),
        ):
            binary = (chg == code).astype(np.float32)
            density = uniform_filter(binary, size=19, mode="constant")
            max_val = float(density.max())
            if max_val > 0.08:
                r, c = np.unravel_index(int(np.argmax(density)), density.shape)
                lat, lon = _rc_to_latlon(r, c)
                # Local km² inside 19x19 window (~361 px)
                local_km2 = round(float(max_val * 361 * region.pixel_area_km2), 2)
                markers.append(
                    {
                        "lat": lat,
                        "lon": lon,
                        "title": title,
                        "detail": f"~{local_km2} km² local change vs {baseline_frame.scene.acq_date}",
                        "color": color,
                    }
                )

    # Also add primary center of mass for the first 2 selected classes
    for cid in selected[:2]:
        cls_info = SURFACE_CLASSES[cid]
        binary = (frame.class_mask == cid).astype(np.float32)
        density = uniform_filter(binary, size=17, mode="constant")
        if float(density.max()) > 0.20:
            r, c = np.unravel_index(int(np.argmax(density)), density.shape)
            lat, lon = _rc_to_latlon(r, c)
            total_km2 = frame.areas_km2.get(cid, 0.0)
            markers.append(
                {
                    "lat": lat,
                    "lon": lon,
                    "title": f"Dense {cls_info.short_label} Zone",
                    "detail": (
                        f"Region total: {total_km2:.2f} km² · "
                        f"HH={frame.scene.hh_db[r, c]:.1f} dB, HV={frame.scene.hv_db[r, c]:.1f} dB"
                    ),
                    "color": cls_info.color_hex,
                }
            )
    return markers


def _quantize_layer_u8(arr: np.ndarray, vmin: float, vmax: float) -> Tuple[str, int, int]:
    """Quantize a float32 grid into `uint8` base64 (`0` = NaN transparent, `1..255` = [vmin, vmax])."""
    a = np.asarray(arr, dtype=np.float32)
    valid = np.isfinite(a)
    u8 = np.zeros(a.shape, dtype=np.uint8)
    span = max(float(vmax) - float(vmin), 1e-6)
    if np.any(valid):
        scaled = np.clip((a[valid] - float(vmin)) / span, 0.0, 1.0)
        u8[valid] = (np.round(scaled * 254.0) + 1.0).astype(np.uint8)
    r, c = u8.shape
    return base64.b64encode(u8.tobytes()).decode("ascii"), int(r), int(c)


def run_aoi_timeframe_pipeline(
    south: float,
    west: float,
    north: float,
    east: float,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    model_id: str = "sar_mlp_v1",
    resolution_m: float = 30.0,
    region_id: Optional[str] = None,
    label: Optional[str] = None,
    force_recompute: bool = False,
    extend_to_source_history: bool = False,
    db_path: Path = DEFAULT_DB_PATH,
) -> Tuple[TimeframeBundle, List[str], str, str]:
    """
    Resolve an active region or dragged/resized AOI at target `resolution_m` in SQLite,
    clamp timeframe safely to its available dates, run neural network inference, and return
    `(bundle, available_dates, clamped_start, clamped_end)`.
    """
    if region_id:
        try:
            region = get_region(region_id, db_path=db_path)
            ensure_auxiliary_layers_for_region(region, db_path=db_path)
        except KeyError:
            region = ensure_custom_aoi_region(
                south=south,
                west=west,
                north=north,
                east=east,
                resolution_m=resolution_m,
                label=label,
                db_path=db_path,
            )
    else:
        region = ensure_custom_aoi_region(
            south=south,
            west=west,
            north=north,
            east=east,
            resolution_m=resolution_m,
            label=label,
            db_path=db_path,
        )

    if extend_to_source_history or (start_date and start_date < "2025-11-01" and region.source_type == "simulated"):
        ensure_multiyear_dates_for_region(region, db_path=db_path)

    avail = list_region_dates(region.region_id, db_path=db_path)
    if not avail:
        raise RuntimeError(f"No SAR scenes available for region '{region.region_id}'.")

    if not start_date or not end_date:
        s_date, e_date = avail[0], avail[-1]
    else:
        in_window = [d for d in avail if start_date <= d <= end_date]
        if len(in_window) < 1:
            s_date, e_date = avail[0], avail[-1]
        else:
            s_date = min(d for d in avail if d >= start_date) if start_date <= avail[-1] else avail[0]
            e_date = max(d for d in avail if d <= end_date) if end_date >= avail[0] else avail[-1]
            if s_date > e_date:
                s_date, e_date = avail[0], avail[-1]

    bundle = run_timeframe_pipeline(
        region_id=region.region_id,
        start_date=s_date,
        end_date=e_date,
        model_id=model_id,
        force_recompute=force_recompute,
        db_path=db_path,
    )
    return bundle, avail, s_date, e_date


def encode_bundle_for_frontend(
    bundle: TimeframeBundle,
    available_dates: Sequence[str],
    selected_elements: Sequence[int],
    view_mode: str,
    sar_band: str,
    opacity: float,
    show_graph: bool,
    presets: Sequence[RegionRecord],
    start_date: str,
    end_date: str,
    db_path: Path = DEFAULT_DB_PATH,
    processed_runs: Sequence[Dict[str, Any]] = (),
    has_earthdata_token: bool = False,
    lang: str = "en",
) -> Dict[str, Any]:
    """Serialize classified frames, auxiliary NASA layers, and map state for the interactive component."""
    lang = lang if lang in TRANSLATIONS else "en"
    region = bundle.region
    w_km, h_km = compute_box_dimensions_km(region.south, region.west, region.north, region.east)

    ensure_auxiliary_layers_for_region(region, db_path=db_path)
    present_keys = set(list_region_layer_keys(region.region_id, db_path=db_path))

    mode_def = LAYER_MODES.get(view_mode, LAYER_MODES["elements"])
    is_aux_mode = mode_def.mode_id not in ("elements", "change", "backscatter")
    layer_grids = (
        get_layer_grids_for_region(region.region_id, mode_def.layer_key, db_path=db_path)
        if is_aux_mode
        else {}
    )

    vmin, vmax = float(mode_def.vmin), float(mode_def.vmax)
    static_layer_b64: Optional[str] = None
    static_rows, static_cols = 0, 0
    if is_aux_mode and layer_grids:
        all_finite = [np.asarray(arr)[np.isfinite(arr)] for arr in layer_grids.values()]
        flat_vals = np.concatenate([v for v in all_finite if v.size > 0]) if all_finite else np.array([], dtype=np.float32)
        if mode_def.mode_id == "elevation_m" and flat_vals.size > 0:
            vmin = round(float(np.percentile(flat_vals, 1)), 1)
            vmax = round(float(np.percentile(flat_vals, 99)), 1)
            if vmax - vmin < 2.0:
                vmax = vmin + 2.0
        if mode_def.is_static and "static" in layer_grids:
            static_layer_b64, static_rows, static_cols = _quantize_layer_u8(
                layer_grids["static"], vmin, vmax
            )

    encoded_frames: List[Dict[str, Any]] = []
    for f in bundle.frames:
        rows, cols = f.class_mask.shape
        hh_u8 = np.clip((f.scene.hh_db + 24.0) / 26.0 * 255.0, 0.0, 255.0).astype(np.uint8)
        hv_u8 = np.clip((f.scene.hv_db + 28.0) / 22.0 * 255.0, 0.0, 255.0).astype(np.uint8)

        layer_b64: Optional[str] = None
        has_layer = True
        if is_aux_mode:
            if mode_def.is_static:
                has_layer = static_layer_b64 is not None
            else:
                arr = layer_grids.get(f.scene.acq_date)
                if arr is not None and np.any(np.isfinite(arr)):
                    layer_b64, _, _ = _quantize_layer_u8(arr, vmin, vmax)
                    has_layer = True
                else:
                    has_layer = False

        encoded_frames.append(
            {
                "date": f.scene.acq_date,
                "rows": int(rows),
                "cols": int(cols),
                "mask_b64": base64.b64encode(f.class_mask.astype(np.uint8).tobytes()).decode("ascii"),
                "hh_b64": base64.b64encode(hh_u8.tobytes()).decode("ascii"),
                "hv_b64": base64.b64encode(hv_u8.tobytes()).decode("ascii"),
                "layer_b64": layer_b64,
                "has_layer": has_layer,
                "areas_km2": {str(k): round(float(v), 2) for k, v in f.areas_km2.items()},
                "mean_conf_pct": round(float(f.confidence.mean()) * 100.0, 1),
                "cloud_pct": round(float(f.scene.cloud_cover_optical_pct), 0),
                "inference_ms": round(float(f.inference_ms), 1),
                "from_cache": bool(f.from_cache),
                "note": f.scene.event_note,
            }
        )

    grid_r = bundle.frames[0].class_mask.shape[0] if bundle.frames else static_rows or 128
    grid_c = bundle.frames[0].class_mask.shape[1] if bundle.frames else static_cols or 128

    def _loc_layer(m) -> Dict[str, str]:
        if lang == "de" and m.mode_id in LAYER_DE:
            d = LAYER_DE[m.mode_id]
            return {
                "label": d["label"],
                "short_label": d["short_label"],
                "category": d["category"],
                "description": d["description"],
            }
        return {
            "label": m.label,
            "short_label": m.short_label,
            "category": m.category,
            "description": m.description,
        }

    available_layers = []
    for mid, m in LAYER_MODES.items():
        is_avail = (
            mid in ("elements", "change", "backscatter")
            or (m.layer_key in present_keys)
        )
        loc = _loc_layer(m)
        available_layers.append(
            {
                "mode_id": m.mode_id,
                "label": loc["label"],
                "short_label": loc["short_label"],
                "category": loc["category"],
                "unit": m.unit,
                "colormap": m.colormap,
                "vmin": m.vmin,
                "vmax": m.vmax,
                "is_static": m.is_static,
                "description": loc["description"],
                "available": bool(is_avail),
            }
        )

    act_loc = _loc_layer(mode_def)
    src_min_d, src_max_d, _ = get_source_date_limits()

    return {
        "lang": lang,
        "i18n": TRANSLATIONS[lang],
        "source_limits": {
            "min_date": available_dates[0] if (region.source_type == "nasa_earthdata" and available_dates) else min(src_min_d, available_dates[0] if available_dates else src_min_d),
            "max_date": available_dates[-1] if (region.source_type == "nasa_earthdata" and available_dates) else max(src_max_d, available_dates[-1] if available_dates else src_max_d),
        },
        "region": {
            "id": region.region_id,
            "name": region.name,
            "country": region.country,
            "south": region.south,
            "west": region.west,
            "north": region.north,
            "east": region.east,
            "center_lat": region.center_lat,
            "center_lon": region.center_lon,
            "width_km": w_km,
            "height_km": h_km,
            "area_km2": round(w_km * h_km, 1),
            "resolution_m": float(region.resolution_m),
            "source_type": region.source_type,
            "grid_rows": int(grid_r),
            "grid_cols": int(grid_c),
        },
        "available_dates": list(available_dates),
        "start_date": start_date,
        "end_date": end_date,
        "selected_elements": [int(cid) for cid in selected_elements],
        "view_mode": mode_def.mode_id,
        "sar_band": sar_band,
        "opacity": float(opacity),
        "show_graph": bool(show_graph),
        "model_id": bundle.model_id,
        "newly_inferred": bundle.newly_inferred,
        "cache_hits": bundle.cache_hits,
        "total_inference_ms": bundle.total_inference_ms,
        "active_layer": {
            "mode_id": mode_def.mode_id,
            "label": act_loc["label"],
            "short_label": act_loc["short_label"],
            "category": act_loc["category"],
            "unit": mode_def.unit,
            "colormap": mode_def.colormap,
            "vmin": round(vmin, 2),
            "vmax": round(vmax, 2),
            "is_static": mode_def.is_static,
            "description": act_loc["description"],
        },
        "static_layer_b64": static_layer_b64,
        "available_layers": available_layers,
        "processed_runs": list(processed_runs),
        "has_earthdata_token": bool(has_earthdata_token),
        "frames": encoded_frames,
        "presets": [
            {
                "id": p.region_id,
                "name": p.name,
                "country": p.country,
                "south": p.south,
                "west": p.west,
                "north": p.north,
                "east": p.east,
            }
            for p in presets
        ],
    }
