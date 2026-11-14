"""Classical SAR backscatter processing & physics-based surface classification baseline."""
from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy.ndimage import uniform_filter
from skimage.filters import threshold_otsu
from skimage.morphology import closing, disk, opening, remove_small_objects


def speckle_lee_filter(img_db: np.ndarray, size: int = 3) -> np.ndarray:
    """Apply an adaptive Lee-style speckle filter in the dB domain."""
    arr = np.asarray(img_db, dtype=np.float32)
    local_mean = uniform_filter(arr, size=size, mode="reflect")
    local_sq_mean = uniform_filter(arr * arr, size=size, mode="reflect")
    local_var = np.maximum(local_sq_mean - local_mean * local_mean, 0.0)
    noise_var = float(np.median(local_var)) + 1e-6
    weight = local_var / (local_var + noise_var)
    return (local_mean + weight * (arr - local_mean)).astype(np.float32)


def otsu_water_mask(hh_db: np.ndarray, min_pixels: int = 12) -> np.ndarray:
    """Detect open water via speckle filtering, Otsu thresholding, and morphology."""
    filtered = speckle_lee_filter(hh_db, size=3)
    thresh = min(float(threshold_otsu(filtered)), -13.5)
    mask = filtered <= thresh
    mask = opening(mask, footprint=disk(1))
    mask = closing(mask, footprint=disk(1))
    if np.any(mask):
        mask = remove_small_objects(mask, max_size=max(min_pixels - 1, 1))
    return mask.astype(bool)


def classify_physics_baseline(
    hh_db: np.ndarray,
    hv_db: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Deterministic multi-class SAR baseline using dual-pol scattering rules.

    Class IDs match `src.config.SURFACE_CLASSES`:
      0 = Bare Ground, 1 = Water, 2 = Tree Lines, 3 = Wetlands, 4 = Cropland, 5 = Urban.
    """
    hh = speckle_lee_filter(hh_db, size=3)
    hv = speckle_lee_filter(hv_db, size=3)
    pol_diff = hh - hv

    hh_mean = uniform_filter(hh, size=3, mode="reflect")
    hh_std = np.sqrt(
        np.maximum(uniform_filter(hh * hh, size=3, mode="reflect") - hh_mean * hh_mean, 0.0)
    )

    mask = np.zeros(hh.shape, dtype=np.uint8)
    confidence = np.full(hh.shape, 0.72, dtype=np.float32)

    # 1. Open water: specular reflection (very low HH & HV)
    water = (hh < -14.8) & (hv < -20.5)
    mask[water] = 1
    confidence[water] = np.clip(0.75 + (-14.8 - hh[water]) * 0.04, 0.65, 0.99)

    # 2. Urban / Infrastructure: bright corner reflectors with strong heterogeneity
    urban = (~water) & (hh > -3.8) & ((hh_std > 1.35) | (hh > -1.8))
    mask[urban] = 5
    confidence[urban] = np.clip(0.74 + (hh[urban] + 3.8) * 0.04, 0.65, 0.98)

    # 3. Wetlands / Flooded Vegetation: trunk-water double-bounce (high HH, large HH-HV)
    wetland = (~water) & (~urban) & (hh > -6.5) & (pol_diff > 6.8)
    mask[wetland] = 3
    confidence[wetland] = np.clip(0.70 + (pol_diff[wetland] - 6.8) * 0.04, 0.60, 0.96)

    # 4. Tree Lines / Forest Canopy: strong volume scattering (high HV, small HH-HV)
    treeline = (~water) & (~urban) & (~wetland) & (hv > -14.8) & (pol_diff < 6.8)
    mask[treeline] = 2
    confidence[treeline] = np.clip(0.72 + (hv[treeline] + 14.8) * 0.045, 0.62, 0.97)

    # 5. Cropland / Soil Moisture: intermediate backscatter with vegetated HV
    cropland = (
        (~water)
        & (~urban)
        & (~wetland)
        & (~treeline)
        & (hh > -11.8)
        & (hh <= -7.0)
        & (hv > -19.2)
    )
    mask[cropland] = 4
    confidence[cropland] = 0.74

    return mask, confidence


def compute_change_mask(
    baseline_mask: np.ndarray,
    current_mask: np.ndarray,
    selected_class_ids: Tuple[int, ...] | list[int],
) -> np.ndarray:
    """
    Compute categorical change map for the union of `selected_class_ids`.

    Returns uint8 array:
      0 = Absent in both
      1 = Persisting (present in both)
      2 = Gained (absent in baseline, present now)
      3 = Lost (present in baseline, absent now)
    """
    selected = set(int(cid) for cid in selected_class_ids if cid != 0)
    if not selected:
        return np.zeros_like(current_mask, dtype=np.uint8)

    base_active = np.isin(baseline_mask, list(selected))
    curr_active = np.isin(current_mask, list(selected))

    out = np.zeros_like(current_mask, dtype=np.uint8)
    out[base_active & curr_active] = 1
    out[(~base_active) & curr_active] = 2
    out[base_active & (~curr_active)] = 3
    return out
