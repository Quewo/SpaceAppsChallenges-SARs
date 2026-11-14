"""Neural network and physics-based SAR surface classification backends."""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.ndimage import uniform_filter

from src.flood import classify_physics_baseline

WEIGHTS_DIR = Path(__file__).resolve().parent / "weights"
DEFAULT_MLP_WEIGHTS_PATH = WEIGHTS_DIR / "sar_mlp_v1.npz"


@dataclass
class ClassificationResult:
    """Output contract returned by any surface classification backend."""

    class_mask: np.ndarray        # uint8 (H, W), values in 0..5
    confidence: np.ndarray        # float32 (H, W), values in [0, 1]
    inference_ms: float
    model_id: str


def extract_sar_features(hh_db: np.ndarray, hv_db: np.ndarray) -> np.ndarray:
    """
    Build a 7-channel spatial/polarimetric feature tensor `(H*W, 7)` from HH and HV dB grids.

    Features:
      0: HH (dB)
      1: HV (dB)
      2: HH - HV polarization difference (dB)
      3: 3x3 spatial mean of HH
      4: 3x3 spatial std of HH (texture / urban corner heterogeneity)
      5: 3x3 spatial mean of HV
      6: 3x3 spatial std of HV (canopy / edge texture)
    """
    hh = np.asarray(hh_db, dtype=np.float32)
    hv = np.asarray(hv_db, dtype=np.float32)

    hh_mean = uniform_filter(hh, size=3, mode="reflect")
    hh_sq_mean = uniform_filter(hh * hh, size=3, mode="reflect")
    hh_std = np.sqrt(np.maximum(hh_sq_mean - hh_mean * hh_mean, 0.0))

    hv_mean = uniform_filter(hv, size=3, mode="reflect")
    hv_sq_mean = uniform_filter(hv * hv, size=3, mode="reflect")
    hv_std = np.sqrt(np.maximum(hv_sq_mean - hv_mean * hv_mean, 0.0))

    pol_diff = hh - hv
    stacked = np.stack(
        [hh, hv, pol_diff, hh_mean, hh_std, hv_mean, hv_std],
        axis=-1,
    )
    return stacked.reshape(-1, 7).astype(np.float32)


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp_vals = np.exp(shifted)
    return exp_vals / np.sum(exp_vals, axis=1, keepdims=True)


def train_reference_sar_mlp(
    save_path: Path = DEFAULT_MLP_WEIGHTS_PATH,
    seed: int = 42,
    epochs: int = 280,
) -> Dict[str, np.ndarray]:
    """
    Train a compact 2-hidden-layer NumPy MLP (7 -> 28 -> 18 -> 6) on physical L-band SAR patches.

    Patch-level generation preserves realistic 3x3 neighborhood mean/std texture for each class.
    """
    rng = np.random.default_rng(seed)

    # Canonical physical L-band SAR signatures: (hh_mu, hh_sig, hv_mu, hv_sig, tex_extra)
    class_specs = {
        0: (-13.0, 0.55, -22.2, 0.65, 0.25),  # Bare Ground
        1: (-18.5, 0.85, -25.0, 0.80, 0.20),  # Water (specular)
        2: (-6.8, 0.65, -11.2, 0.65, 0.70),   # Tree Lines (volume scattering)
        3: (-4.2, 0.65, -13.4, 0.65, 0.50),   # Wetlands (trunk-water double-bounce)
        4: (-9.8, 0.65, -16.8, 0.65, 0.35),   # Cropland / Soil Moisture
        5: (-0.4, 0.90, -11.5, 0.95, 1.95),   # Urban (strong corner reflectors + high texture)
    }

    features_list: List[np.ndarray] = []
    labels_list: List[np.ndarray] = []

    patch_size = 22
    for cls_id, (hh_mu, hh_sig, hv_mu, hv_sig, tex_extra) in class_specs.items():
        for _ in range(4):
            hh_patch = rng.normal(hh_mu, hh_sig, size=(patch_size, patch_size)).astype(np.float32)
            hv_patch = rng.normal(hv_mu, hv_sig, size=(patch_size, patch_size)).astype(np.float32)
            if tex_extra > 0.4:
                # Simulate structured corner-reflector / canopy roughness spikes
                spikes = rng.normal(0.0, tex_extra, size=(patch_size, patch_size)).astype(np.float32)
                hh_patch += spikes
                hv_patch += 0.45 * spikes
            feat = extract_sar_features(hh_patch, hv_patch)
            # Drop 1-pixel border to avoid reflection edge bias
            inner_mask = np.ones((patch_size, patch_size), dtype=bool)
            inner_mask[0, :] = inner_mask[-1, :] = inner_mask[:, 0] = inner_mask[:, -1] = False
            inner_feat = feat[inner_mask.ravel()]
            features_list.append(inner_feat)
            labels_list.append(np.full(inner_feat.shape[0], cls_id, dtype=np.int64))

    X = np.vstack(features_list)
    y = np.concatenate(labels_list)

    mean = X.mean(axis=0, keepdims=True).astype(np.float32)
    std = (X.std(axis=0, keepdims=True) + 1e-6).astype(np.float32)
    Xn = (X - mean) / std

    n_samples, in_dim = Xn.shape
    h1, h2, out_dim = 28, 18, 6

    w1 = (rng.normal(0, np.sqrt(2.0 / in_dim), size=(in_dim, h1))).astype(np.float32)
    b1 = np.zeros((1, h1), dtype=np.float32)
    w2 = (rng.normal(0, np.sqrt(2.0 / h1), size=(h1, h2))).astype(np.float32)
    b2 = np.zeros((1, h2), dtype=np.float32)
    w3 = (rng.normal(0, np.sqrt(2.0 / h2), size=(h2, out_dim))).astype(np.float32)
    b3 = np.zeros((1, out_dim), dtype=np.float32)

    y_onehot = np.zeros((n_samples, out_dim), dtype=np.float32)
    y_onehot[np.arange(n_samples), y] = 1.0

    # Adam optimizer in pure NumPy
    params = [w1, b1, w2, b2, w3, b3]
    m_state = [np.zeros_like(p) for p in params]
    v_state = [np.zeros_like(p) for p in params]
    lr = 0.025
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    l2 = 1e-4

    for step in range(1, epochs + 1):
        z1 = Xn @ params[0] + params[1]
        a1 = np.maximum(z1, 0.0)
        z2 = a1 @ params[2] + params[3]
        a2 = np.maximum(z2, 0.0)
        logits = a2 @ params[4] + params[5]
        probs = _softmax(logits)

        dlogits = (probs - y_onehot) / n_samples
        dw3 = a2.T @ dlogits + l2 * params[4]
        db3 = dlogits.sum(axis=0, keepdims=True)

        da2 = dlogits @ params[4].T
        dz2 = da2 * (z2 > 0.0)
        dw2 = a1.T @ dz2 + l2 * params[2]
        db2 = dz2.sum(axis=0, keepdims=True)

        da1 = dz2 @ params[2].T
        dz1 = da1 * (z1 > 0.0)
        dw1 = Xn.T @ dz1 + l2 * params[0]
        db1 = dz1.sum(axis=0, keepdims=True)

        grads = [dw1, db1, dw2, db2, dw3, db3]
        for idx, g in enumerate(grads):
            m_state[idx] = beta1 * m_state[idx] + (1.0 - beta1) * g
            v_state[idx] = beta2 * v_state[idx] + (1.0 - beta2) * (g * g)
            m_hat = m_state[idx] / (1.0 - beta1 ** step)
            v_hat = v_state[idx] / (1.0 - beta2 ** step)
            params[idx] -= lr * m_hat / (np.sqrt(v_hat) + eps)

    weights = {
        "mean": mean,
        "std": std,
        "w1": params[0],
        "b1": params[1],
        "w2": params[2],
        "b2": params[3],
        "w3": params[4],
        "b3": params[5],
    }
    save_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(save_path, **weights)
    return weights


class BaseSARClassifier(ABC):
    """Interface for plugging any SAR surface classification model into the backend."""

    model_id: str
    display_name: str
    description: str

    @abstractmethod
    def predict(self, hh_db: np.ndarray, hv_db: np.ndarray) -> ClassificationResult:
        """Classify a co-registered SAR scene (HH and HV in dB) into class IDs 0..5."""


class SARNeuralNetClassifier(BaseSARClassifier):
    """
    Built-in spatial-polarimetric Neural Network (MLP 7 -> 28 -> 18 -> 6).

    Replace `weights_path` or subclass `BaseSARClassifier` to connect your team's
    trained PyTorch / ONNX checkpoint without touching UI or database code.
    """

    def __init__(self, weights_path: Path = DEFAULT_MLP_WEIGHTS_PATH) -> None:
        self.model_id = "sar_mlp_v1"
        self.display_name = "SAR-MLP Neural Network (v1)"
        self.description = (
            "2-hidden-layer neural network trained on 7 dual-pol & 3x3 spatial texture features "
            "(HH, HV, HH-HV, local means, and local standard deviations)."
        )
        self.weights_path = weights_path
        if weights_path.exists():
            loaded = np.load(weights_path)
            self.weights = {k: loaded[k] for k in loaded.files}
        else:
            self.weights = train_reference_sar_mlp(weights_path)

    def predict(self, hh_db: np.ndarray, hv_db: np.ndarray) -> ClassificationResult:
        t0 = time.perf_counter()
        h, w = hh_db.shape
        X = extract_sar_features(hh_db, hv_db)
        Xn = (X - self.weights["mean"]) / self.weights["std"]

        a1 = np.maximum(Xn @ self.weights["w1"] + self.weights["b1"], 0.0)
        a2 = np.maximum(a1 @ self.weights["w2"] + self.weights["b2"], 0.0)
        logits = a2 @ self.weights["w3"] + self.weights["b3"]
        probs = _softmax(logits)

        class_mask = np.argmax(probs, axis=1).reshape(h, w).astype(np.uint8)
        confidence = np.max(probs, axis=1).reshape(h, w).astype(np.float32)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return ClassificationResult(
            class_mask=class_mask,
            confidence=confidence,
            inference_ms=round(elapsed_ms, 2),
            model_id=self.model_id,
        )


class ClassicalPhysicsClassifier(BaseSARClassifier):
    """Deterministic SAR physics & Otsu baseline from `src.flood` for validation."""

    def __init__(self) -> None:
        self.model_id = "physics_otsu_v1"
        self.display_name = "SAR Physics Baseline (Lee + Dual-Pol Rules)"
        self.description = (
            "Classical Lee speckle filtering combined with dual-pol scattering rules "
            "(specular water, volume canopy, double-bounce wetlands/urban)."
        )

    def predict(self, hh_db: np.ndarray, hv_db: np.ndarray) -> ClassificationResult:
        t0 = time.perf_counter()
        mask, conf = classify_physics_baseline(hh_db, hv_db)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return ClassificationResult(
            class_mask=mask,
            confidence=conf,
            inference_ms=round(elapsed_ms, 2),
            model_id=self.model_id,
        )


_CLASSIFIER_CACHE: Dict[str, BaseSARClassifier] = {}


def list_models() -> List[BaseSARClassifier]:
    """Return all registered classification backends."""
    return [
        get_classifier("sar_mlp_v1"),
        get_classifier("physics_otsu_v1"),
    ]


def get_classifier(model_id: str) -> BaseSARClassifier:
    """Retrieve a cached classifier backend instance by ID."""
    if model_id not in _CLASSIFIER_CACHE:
        if model_id == "sar_mlp_v1":
            _CLASSIFIER_CACHE[model_id] = SARNeuralNetClassifier()
        elif model_id == "physics_otsu_v1":
            _CLASSIFIER_CACHE[model_id] = ClassicalPhysicsClassifier()
        else:
            raise ValueError(f"Unknown model_id '{model_id}'.")
    return _CLASSIFIER_CACHE[model_id]
