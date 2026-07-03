from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


def select_xai_candidates(detections: pd.DataFrame, max_per_frame: int = 5, c_low: float = 0.5, sigma_k: float = 0.3, sigma_s: float = 0.3, tau_pre: float = 0.3) -> pd.DataFrame:
    frame = detections.copy()
    frame["Q_pre"] = frame[["c_i", "k_i", "s_i"]].min(axis=1)
    frame["risk"] = (1.0 - frame["c_i"]) + (1.0 - frame["k_i"]) + (1.0 - frame["s_i"])
    mask = (frame["confidence"] < c_low) | (frame["k_i"] < sigma_k) | (frame["s_i"] < sigma_s) | (frame["Q_pre"] < tau_pre)
    return frame[mask].sort_values("risk", ascending=False).head(max_per_frame)


def compute_cam(image_path: str | Path, method: str = "eigencam") -> tuple[np.ndarray | None, float]:
    start = time.perf_counter()
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None, 0.0
    if method not in {"eigencam", "gradcam"}:
        raise ValueError(f"Unsupported XAI method: {method}")
    blur = cv2.GaussianBlur(image, (5, 5), 0)
    grad_x = cv2.Sobel(blur, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(blur, cv2.CV_32F, 0, 1, ksize=3)
    cam = normalize_cam(cv2.magnitude(grad_x, grad_y))
    return cam, (time.perf_counter() - start) * 1000.0


def compute_xai_score(cam: np.ndarray | None, bbox: tuple[float, float, float, float]) -> tuple[float | None, float, float]:
    if cam is None:
        return None, 0.0, 0.0
    total = float(cam.sum())
    if total <= 0:
        return None, 0.0, total
    h, w = cam.shape[:2]
    x1, y1, x2, y2 = bbox
    ix1, iy1 = max(0, int(round(x1))), max(0, int(round(y1)))
    ix2, iy2 = min(w, int(round(x2))), min(h, int(round(y2)))
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0, 0.0, total
    inside = float(cam[iy1:iy2, ix1:ix2].sum())
    return inside / total, inside, total


def normalize_cam(cam: np.ndarray) -> np.ndarray:
    cam = cam.astype(np.float32)
    lo, hi = float(cam.min()), float(cam.max())
    if hi <= lo:
        return np.zeros_like(cam, dtype=np.float32)
    return (cam - lo) / (hi - lo)
