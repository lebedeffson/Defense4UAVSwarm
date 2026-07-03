from __future__ import annotations

import json
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from defense4uavswarm.tracking.simple import iou


def load_frame_index(root: str | Path) -> pd.DataFrame:
    root = Path(root)
    frame = pd.read_csv(root / "metadata" / "frame_index.csv")
    frame["transform_to_reference_matrix"] = frame["transform_to_reference"].map(lambda x: np.array(json.loads(x), dtype=float))
    frame["transform_from_reference_matrix"] = frame["transform_from_reference"].map(lambda x: np.array(json.loads(x), dtype=float))
    transforms_path = root / "metadata" / "transforms.json"
    if transforms_path.exists():
        payload = json.loads(transforms_path.read_text(encoding="utf-8"))
        jitter = {agent: spec.get("transform", {}).get("bbox_jitter", 0.0) for agent, spec in payload.get("agents", {}).items()}
        frame["bbox_jitter"] = frame["agent_id"].map(jitter).fillna(0.0).astype(float)
    else:
        frame["bbox_jitter"] = 0.0
    return frame


def transform_bbox(box: tuple[float, float, float, float], matrix: np.ndarray) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = box
    pts = np.array([[x1, y1, 1.0], [x2, y1, 1.0], [x2, y2, 1.0], [x1, y2, 1.0]], dtype=float).T
    out = matrix @ pts
    out = out[:2] / np.maximum(out[2:], 1e-9)
    return (float(out[0].min()), float(out[1].min()), float(out[0].max()), float(out[1].max()))


def jitter_bbox(box: tuple[float, float, float, float], jitter: float, *seed_parts: object) -> tuple[float, float, float, float]:
    if jitter <= 0:
        return box
    seed_text = "|".join(str(p) for p in seed_parts).encode("utf-8")
    seed = int.from_bytes(hashlib.sha256(seed_text).digest()[:8], "little", signed=False)
    rng = np.random.default_rng(seed)
    dx, dy = rng.normal(0.0, jitter, size=2)
    dw, dh = rng.normal(0.0, jitter * 0.25, size=2)
    x1, y1, x2, y2 = box
    return (float(x1 + dx - dw), float(y1 + dy - dh), float(x2 + dx + dw), float(y2 + dy + dh))


def compute_inter_agent_consistency(
    detections: pd.DataFrame,
    frame_index: pd.DataFrame,
    iou_min: float = 0.3,
    s_missing_policy: str = "neutral",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if detections.empty:
        return detections.copy(), pd.DataFrame()
    transforms = frame_index.set_index(["sequence_id", "frame_id", "agent_id"])["transform_to_reference_matrix"].to_dict()
    det = detections.copy().reset_index(drop=True)
    det["det_id"] = det.get("det_id", pd.Series(range(len(det)))).fillna(pd.Series(range(len(det)))).astype(int)
    ref_boxes = []
    for row in det.itertuples(index=False):
        matrix = transforms[(row.sequence_id, row.frame_id, row.agent_id)]
        ref_boxes.append(transform_bbox((row.x1, row.y1, row.x2, row.y2), matrix))
    det[["bbox_ref_x1", "bbox_ref_y1", "bbox_ref_x2", "bbox_ref_y2"]] = pd.DataFrame(ref_boxes, index=det.index)
    feature_rows, match_rows = [], []
    for _, group in det.groupby(["sequence_id", "frame_id"], sort=False):
        records = group.to_dict("records")
        for a in records:
            best_iou = 0.0
            best_agent = None
            match_count = 0
            best_row = None
            accepted_rows = []
            for b in records:
                if a["agent_id"] == b["agent_id"]:
                    continue
                compatible = _class_compatible(a, b)
                val = iou(_ref_box(a), _ref_box(b)) if compatible else 0.0
                accepted = compatible and val >= iou_min
                match_count += int(accepted)
                if val > best_iou:
                    best_iou, best_agent = val, b["agent_id"]
                    best_row = _match_row(a, b, val, compatible, accepted)
                if accepted:
                    accepted_rows.append(_match_row(a, b, val, compatible, accepted))
            match_rows.extend(accepted_rows)
            if best_row and not best_row["match_accepted"]:
                match_rows.append(best_row)
            has_neighbors = any(a["agent_id"] != b["agent_id"] for b in records)
            s_i = best_iou if has_neighbors else _missing_value(s_missing_policy)
            row = dict(a)
            row.update(
                {
                    "c_i": float(a.get("confidence", 1.0)),
                    "s_i": float(s_i),
                    "best_neighbor_agent": best_agent,
                    "best_neighbor_iou": float(best_iou),
                    "num_neighbor_matches": match_count,
                    "s_missing_policy": s_missing_policy,
                }
            )
            feature_rows.append(row)
    return pd.DataFrame(feature_rows), pd.DataFrame(match_rows)


def _ref_box(row: dict) -> tuple[float, float, float, float]:
    return (row["bbox_ref_x1"], row["bbox_ref_y1"], row["bbox_ref_x2"], row["bbox_ref_y2"])


def _class_compatible(a: dict, b: dict) -> bool:
    if "class_id" in a and "class_id" in b and pd.notna(a["class_id"]) and pd.notna(b["class_id"]):
        return int(a["class_id"]) == int(b["class_id"])
    if a.get("class_name") and b.get("class_name"):
        return a["class_name"] == b["class_name"]
    return True


def _match_row(a: dict, b: dict, val: float, compatible: bool, accepted: bool) -> dict:
    return {
        "sequence_id": a["sequence_id"],
        "frame_id": a["frame_id"],
        "agent_a": a["agent_id"],
        "agent_b": b["agent_id"],
        "det_id_a": a["det_id"],
        "det_id_b": b["det_id"],
        "bbox_a_ref": json.dumps(list(_ref_box(a))),
        "bbox_b_ref": json.dumps(list(_ref_box(b))),
        "iou_ref": val,
        "class_a": a.get("class_name"),
        "class_b": b.get("class_name"),
        "class_compatible": compatible,
        "match_accepted": accepted,
    }


def _missing_value(policy: str) -> float:
    return {"neutral": 1.0, "penalize": 0.5, "strict": 0.0}.get(policy, 1.0)
