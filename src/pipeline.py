"""Backend orchestration connecting SQLite SAR scenes, neural network inference, and map overlays."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib
import numpy as np
from scipy.ndimage import uniform_filter

from src.classifier import get_classifier
from src.config import CHANGE_COLORS, SURFACE_CLASSES
from src.database import (
    DEFAULT_DB_PATH,
    RegionRecord,
    SARSceneRecord,
    get_classification,
    get_region,
    get_scenes_in_timeframe,
    save_classification,
)
from src.flood import compute_change_mask


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
) -> np.ndarray:
    """
    Create an `(H, W, 4)` uint8 RGBA overlay for Folium `ImageOverlay`.

    `view_mode`:
      - "elements": renders selected surface classes in color, keeping unselected classes transparent.
      - "change": compares `frame` against `baseline_frame` for selected classes (Gained/Lost/Stable).
      - "backscatter": visualizes raw SAR radar backscatter (`HH`, `HV`, or dual-pol RGB composite).
    """
    h, w = frame.class_mask.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    selected = [int(cid) for cid in selected_class_ids if cid != 0]

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
