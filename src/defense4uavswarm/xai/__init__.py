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


def compute_xai_score(cam: np.ndarray | None, bbox: tuple[float, float, float, float]) -> dict:
    if cam is None:
        return _empty_scores()
    total = float(cam.sum())
    if total <= 0:
        out = _empty_scores()
        out["xai_total_energy"] = total
        return out
    h, w = cam.shape[:2]
    x1, y1, x2, y2 = bbox
    ix1, iy1 = max(0, int(round(x1))), max(0, int(round(y1)))
    ix2, iy2 = min(w, int(round(x2))), min(h, int(round(y2)))
    if ix2 <= ix1 or iy2 <= iy1:
        out = _empty_scores()
        out.update({"x_raw": 0.0, "xai_total_energy": total, "image_area": float(w * h)})
        return out
    inside = float(cam[iy1:iy2, ix1:ix2].sum())
    x_raw = inside / total
    bbox_area = float((ix2 - ix1) * (iy2 - iy1))
    image_area = float(w * h)
    area_fraction = bbox_area / max(1.0, image_area)
    density = x_raw / max(area_fraction, 1e-6)
    peak_y, peak_x = np.unravel_index(int(np.argmax(cam)), cam.shape)
    top_scores = {}
    for pct in (1, 5, 10):
        threshold = np.quantile(cam, 1.0 - pct / 100.0)
        top_mask = cam >= threshold
        denom = int(top_mask.sum())
        inside_top = int(top_mask[iy1:iy2, ix1:ix2].sum())
        top_scores[f"x_top{pct}_inside"] = inside_top / max(1, denom)
    return {
        "x_raw": x_raw,
        "x_i": min(max(density / 5.0, 0.0), 1.0),
        "xai_energy_inside_bbox": inside,
        "xai_total_energy": total,
        "bbox_area": bbox_area,
        "image_area": image_area,
        "bbox_area_fraction": area_fraction,
        "x_density": density,
        "x_i_density_norm_cap2": min(max(density / 2.0, 0.0), 1.0),
        "x_i_density_norm_cap5": min(max(density / 5.0, 0.0), 1.0),
        "x_i_density_norm_cap10": min(max(density / 10.0, 0.0), 1.0),
        "x_top1_inside": top_scores["x_top1_inside"],
        "x_top5_inside": top_scores["x_top5_inside"],
        "x_top10_inside": top_scores["x_top10_inside"],
        "x_peak_inside": float(ix1 <= peak_x < ix2 and iy1 <= peak_y < iy2),
        "x_i_selected": min(max(density / 5.0, 0.0), 1.0),
        "x_i_selected_method": "density_norm_cap5",
    }


def _empty_scores() -> dict:
    return {
        "x_raw": None,
        "x_i": None,
        "xai_energy_inside_bbox": 0.0,
        "xai_total_energy": 0.0,
        "bbox_area": None,
        "image_area": None,
        "bbox_area_fraction": None,
        "x_density": None,
        "x_i_density_norm_cap2": None,
        "x_i_density_norm_cap5": None,
        "x_i_density_norm_cap10": None,
        "x_top1_inside": None,
        "x_top5_inside": None,
        "x_top10_inside": None,
        "x_peak_inside": None,
        "x_i_selected": None,
        "x_i_selected_method": "unavailable",
    }


def normalize_cam(cam: np.ndarray) -> np.ndarray:
    cam = cam.astype(np.float32)
    lo, hi = float(cam.min()), float(cam.max())
    if hi <= lo:
        return np.zeros_like(cam, dtype=np.float32)
    return (cam - lo) / (hi - lo)
