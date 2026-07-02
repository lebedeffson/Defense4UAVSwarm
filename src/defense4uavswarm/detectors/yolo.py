from __future__ import annotations

from pathlib import Path
import math
from time import perf_counter

import pandas as pd
from ultralytics import YOLO

from defense4uavswarm.schema import to_frame
from defense4uavswarm.tracking.simple import iou


class YoloRunner:
    def __init__(self, weights: str, device: str = "auto") -> None:
        self.model = YOLO(weights)
        self.device = None if device == "auto" else device

    def reset_tracker(self) -> None:
        self.model.predictor = None

    def track_frame(self, image_path: Path, cfg: dict, scenario: str, eps: float | None, sequence_id: str, frame_id: int, state: dict) -> tuple[list[dict], float]:
        start = perf_counter()
        res = self.model.track(
            source=str(image_path),
            imgsz=cfg["model"]["imgsz"],
            conf=cfg["model"]["conf"],
            iou=cfg["model"]["iou"],
            max_det=cfg["model"]["max_det"],
            tracker=cfg["model"]["tracker"],
            persist=True,
            device=self.device,
            verbose=False,
        )[0]
        latency_ms = (perf_counter() - start) * 1000
        rows = []
        boxes = res.boxes
        if boxes is None:
            return rows, latency_ms
        for b in boxes:
            x1, y1, x2, y2 = [float(v) for v in b.xyxy[0].tolist()]
            conf = float(b.conf[0])
            cls = int(b.cls[0])
            class_name = res.names.get(cls, str(cls)) if hasattr(res, "names") else str(cls)
            pred_id = int(b.id[0]) if b.id is not None else None
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            k_center = 1.0
            k_iou = 1.0
            d_center = 0.0
            alpha_base = max(1.0, math.sqrt(max(1.0, (x2 - x1) * (y2 - y1))))
            track_age = 1
            if pred_id is not None and pred_id in state:
                px, py, vx, vy, prev_box, vbox, size, prev_age = state[pred_id]
                track_age = int(prev_age) + 1
                if track_age >= int(cfg["filtering"].get("new_track_k_neutral_age", 3)):
                    predicted_center = (px + vx, py + vy)
                    predicted_box = tuple(prev_box[i] + vbox[i] for i in range(4))
                    d_center = math.hypot(cx - predicted_center[0], cy - predicted_center[1])
                    alpha_base = max(1.0, size)
                    k_center = math.exp(-d_center / alpha_base)
                    k_iou = iou(predicted_box, (x1, y1, x2, y2))
            k_combined = math.sqrt(max(0.0, min(1.0, k_center)) * max(0.0, min(1.0, k_iou)))
            k_variant = cfg.get("filtering", {}).get("k_variant", "center")
            if k_variant == "iou":
                k_i = k_iou
            elif k_variant == "combined":
                k_i = k_combined
            else:
                k_i = k_center
            if pred_id is not None:
                prev = state.get(pred_id)
                if prev is None:
                    vx, vy = 0.0, 0.0
                    vbox = (0.0, 0.0, 0.0, 0.0)
                else:
                    prev_box = prev[4]
                    vx, vy = cx - prev[0], cy - prev[1]
                    vbox = (x1 - prev_box[0], y1 - prev_box[1], x2 - prev_box[2], y2 - prev_box[3])
                size = math.sqrt(max(1.0, (x2 - x1) * (y2 - y1)))
                state[pred_id] = (cx, cy, vx, vy, (x1, y1, x2, y2), vbox, size, track_age)
            rows.append(
                {
                    "scenario": scenario,
                    "model_name": cfg["model"].get("name", Path(cfg["model"]["weights"]).stem),
                    "fgsm_loss": cfg.get("fgsm", {}).get("loss", "class_only"),
                    "eps": eps,
                    "frame_id": frame_id,
                    "sequence_id": sequence_id,
                    "gt_track_id": None,
                    "pred_track_id": pred_id,
                    "class_id": cls,
                    "class_name": class_name,
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "confidence": conf,
                    "c_i": conf,
                    "k_i": k_i,
                    "k_variant": k_variant,
                    "k_center": k_center,
                    "k_iou": k_iou,
                    "k_combined": k_combined,
                    "d_center": d_center,
                    "alpha_base": alpha_base,
                    "alpha_scale": cfg.get("filtering", {}).get("alpha_scale", 1.0),
                    "track_age": track_age,
                    "s_i": 1.0,
                    "x_i": 1.0,
                    "Q_raw": None,
                    "Q_i": None,
                    "Q_smooth": None,
                    "beta": None,
                    "accepted": True,
                    "t_norm": None,
                    "tau": None,
                    "image_path": str(image_path),
                    "latency_ms": latency_ms,
                    "xai_triggered": False,
                }
            )
        return rows, latency_ms
