from __future__ import annotations

import math

import pandas as pd


def iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter
    return inter / denom if denom > 0 else 0.0


def add_simple_tracks(df: pd.DataFrame, iou_threshold: float = 0.3, max_age: int = 3) -> pd.DataFrame:
    df = df.sort_values(["sequence_id", "frame_id", "confidence"], ascending=[True, True, False]).copy()
    next_id = 1
    outputs = []
    for _, seq_df in df.groupby("sequence_id", sort=False):
        tracks: dict[int, dict] = {}
        for frame_id, frame_df in seq_df.groupby("frame_id", sort=True):
            used_tracks = set()
            for idx, row in frame_df.iterrows():
                box = (row.x1, row.y1, row.x2, row.y2)
                best_tid, best_iou = None, 0.0
                for tid, tr in tracks.items():
                    if tid in used_tracks or frame_id - tr["frame_id"] > max_age:
                        continue
                    score = iou(box, tr["box"])
                    if score > best_iou:
                        best_tid, best_iou = tid, score
                if best_tid is None or best_iou < iou_threshold:
                    best_tid = next_id
                    next_id += 1
                used_tracks.add(best_tid)
                cx = (row.x1 + row.x2) / 2
                cy = (row.y1 + row.y2) / 2
                prev = tracks.get(best_tid)
                if prev is None:
                    k_i = 1.0
                else:
                    px, py = prev["center"]
                    vx, vy = prev.get("velocity", (0.0, 0.0))
                    d = math.hypot(cx - (px + vx), cy - (py + vy))
                    size = max(1.0, math.sqrt(max(1.0, (row.x2 - row.x1) * (row.y2 - row.y1))))
                    k_i = math.exp(-d / size)
                velocity = (0.0, 0.0) if prev is None else (cx - prev["center"][0], cy - prev["center"][1])
                tracks[best_tid] = {"box": box, "frame_id": frame_id, "center": (cx, cy), "velocity": velocity}
                out = row.to_dict()
                out["pred_track_id"] = best_tid
                out["k_i"] = k_i
                outputs.append(out)
    return pd.DataFrame(outputs)
