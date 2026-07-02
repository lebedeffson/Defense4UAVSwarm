from __future__ import annotations

from pathlib import Path
import math
from time import perf_counter

import pandas as pd
from ultralytics import YOLO

from defense4uavswarm.schema import to_frame


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
            k_i = 1.0
            track_age = 1
            if pred_id is not None and pred_id in state:
                px, py, vx, vy, size, prev_age = state[pred_id]
                track_age = int(prev_age) + 1
                if track_age >= int(cfg["filtering"].get("new_track_k_neutral_age", 3)):
                    d = math.hypot(cx - (px + vx), cy - (py + vy))
                    k_i = math.exp(-d / max(1.0, size))
            if pred_id is not None:
                prev = state.get(pred_id)
                vx, vy = (0.0, 0.0) if prev is None else (cx - prev[0], cy - prev[1])
                size = math.sqrt(max(1.0, (x2 - x1) * (y2 - y1)))
                state[pred_id] = (cx, cy, vx, vy, size, track_age)
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
                    "track_age": track_age,
                    "s_i": 1.0,
                    "x_i": 1.0,
                    "Q_i": None,
                    "accepted": True,
                    "t_norm": None,
                    "tau": None,
                    "image_path": str(image_path),
                    "latency_ms": latency_ms,
                    "xai_triggered": False,
                }
            )
        return rows, latency_ms
