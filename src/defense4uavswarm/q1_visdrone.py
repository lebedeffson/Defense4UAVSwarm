from __future__ import annotations

import json
import math
import pickle
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from defense4uavswarm.datasets.visdrone import VISDRONE_CLASSES, VisDroneDataset
from defense4uavswarm.v8_sim import vector_iou


EPS = 1e-9
COCO_TO_VISDRONE = {
    "person": {"pedestrian", "people"},
    "bicycle": {"bicycle"},
    "car": {"car", "van"},
    "truck": {"truck"},
    "bus": {"bus"},
    "motorcycle": {"motor"},
}


@dataclass(frozen=True)
class Q1Params:
    naive_conf: float = 0.30
    low_conf: float = 0.10
    high_conf: float = 0.50
    persistence_min_age: int = 1
    ema_lambda: float = 0.7
    ema_threshold: float = 0.45
    bayes_threshold: float = 0.55
    logodds_confirm: float = 1.0
    logodds_strong_reward: float = 0.8
    logodds_weak_reward: float = 0.35
    logodds_start: float = 0.25
    geom_q_floor: float = 0.6
    geom_threshold: float = 0.45
    rf_threshold: float = 0.55


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, payload: Any) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_gt(dataset_root: str | Path, sequence_ids: list[str] | None = None) -> pd.DataFrame:
    ds = VisDroneDataset(dataset_root, "val", subset_hint="VID")
    gt = ds.all_annotations(sequence_ids)
    if gt.empty:
        return gt
    gt = gt.rename(columns={"gt_track_id": "object_id"}).copy()
    gt["object_id"] = gt["object_id"].astype(str)
    return gt


def load_detections(path: str | Path) -> pd.DataFrame:
    payload = read_json(path)
    rows: list[dict[str, Any]] = []
    for rec in payload.get("frames", []):
        for det in rec.get("detections", []):
            box = det.get("bbox") or det.get("bbox_2d")
            rows.append(
                {
                    "det_id": det.get("det_id", f"{rec['sequence_id']}_{rec['frame_id']}_{len(rows)}"),
                    "sequence_id": rec["sequence_id"],
                    "frame_id": int(rec["frame_id"]),
                    "image_path": rec.get("image_path", ""),
                    "class_id": det.get("class_id", -1),
                    "class_name": str(det.get("class_name", det.get("class_id", ""))),
                    "bbox": [float(v) for v in box],
                    "confidence": float(det.get("confidence", det.get("score", 0.0))),
                    "detector": det.get("detector", payload.get("detector", "")),
                }
            )
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    boxes = pd.DataFrame(df["bbox"].tolist(), columns=["x1", "y1", "x2", "y2"])
    return pd.concat([df, boxes], axis=1)


def class_compatible(det: pd.Series, gt: pd.Series) -> bool:
    det_name = str(det.get("class_name", "")).lower()
    gt_name = str(gt.get("class_name", "")).lower()
    if det_name == gt_name:
        return True
    if det_name in COCO_TO_VISDRONE and gt_name in COCO_TO_VISDRONE[det_name]:
        return True
    try:
        det_id = int(det.get("class_id", -999))
        gt_id = int(gt.get("class_id", -999))
    except Exception:
        return False
    if det_id == gt_id:
        return True
    return VISDRONE_CLASSES.get(det_id, "").lower() == gt_name


def label_detections(det: pd.DataFrame, gt: pd.DataFrame, iou_threshold: float = 0.5) -> pd.DataFrame:
    if det.empty:
        return det.copy()
    d = det.copy().reset_index(drop=True)
    eval_is_tp = np.zeros(len(d), dtype=bool)
    matched_gt_id = np.full(len(d), "", dtype=object)
    matched_gt_iou = np.zeros(len(d), dtype=float)
    gt_by_key = {key: group.reset_index(drop=True) for key, group in gt.groupby(["sequence_id", "frame_id"], sort=False)}
    for key, group in d.groupby(["sequence_id", "frame_id"], sort=False):
        g = gt_by_key.get(key)
        if g is None or g.empty:
            continue
        gt_boxes = g[["x1", "y1", "x2", "y2"]].to_numpy(dtype=float)
        gt_names = g["class_name"].astype(str).str.lower().to_numpy()
        gt_ids = g["object_id"].astype(str).to_numpy()
        used = np.zeros(len(g), dtype=bool)
        for row in group.sort_values("confidence", ascending=False).itertuples():
            idx = int(row.Index)
            allowed_names = compatible_gt_names(str(row.class_name).lower(), int(row.class_id) if str(row.class_id).lstrip("-").isdigit() else None)
            compatible = np.isin(gt_names, list(allowed_names)) & (~used)
            candidates = np.flatnonzero(compatible)
            if len(candidates) == 0:
                continue
            det_box = np.asarray([row.x1, row.y1, row.x2, row.y2], dtype=float)
            ious = vector_iou(det_box, gt_boxes[candidates])
            best_pos = int(np.argmax(ious))
            if float(ious[best_pos]) >= iou_threshold:
                gt_idx = candidates[best_pos]
                used[gt_idx] = True
                eval_is_tp[idx] = True
                matched_gt_id[idx] = gt_ids[gt_idx]
                matched_gt_iou[idx] = float(ious[best_pos])
    d["eval_is_tp"] = eval_is_tp
    d["matched_gt_id"] = matched_gt_id
    d["matched_gt_iou"] = matched_gt_iou
    return d


def compatible_gt_names(det_name: str, det_class_id: int | None = None) -> set[str]:
    if det_name in COCO_TO_VISDRONE:
        return set(COCO_TO_VISDRONE[det_name])
    names = {det_name}
    if det_class_id is not None and det_class_id in VISDRONE_CLASSES:
        names.add(VISDRONE_CLASSES[det_class_id].lower())
    return names


def add_single_camera_features(det: pd.DataFrame, max_gap: int = 2, link_iou: float = 0.25) -> pd.DataFrame:
    if det.empty:
        return det.copy()
    d = det.copy().sort_values(["sequence_id", "class_name", "frame_id", "confidence"], ascending=[True, True, True, False]).reset_index(drop=True)
    d["bbox_area"] = (d["x2"] - d["x1"]).clip(lower=1) * (d["y2"] - d["y1"]).clip(lower=1)
    d["bbox_aspect_ratio"] = (d["x2"] - d["x1"]).clip(lower=1) / (d["y2"] - d["y1"]).clip(lower=1)
    d["num_detections_in_frame"] = d.groupby(["sequence_id", "frame_id"])["det_id"].transform("count")
    d["c_i"] = d["confidence"].clip(0, 1)
    tracklet_ids = np.full(len(d), "", dtype=object)
    temporal_age = np.zeros(len(d), dtype=int)
    k_values = np.full(len(d), 0.35, dtype=float)
    next_id = 0
    for (seq, cls), group in d.groupby(["sequence_id", "class_name"], sort=False):
        active: dict[str, dict[str, Any]] = {}
        for frame_id, frame in group.groupby("frame_id", sort=True):
            assigned: set[str] = set()
            for row in frame.sort_values("confidence", ascending=False).itertuples():
                idx = int(row.Index)
                row_box = np.asarray([row.x1, row.y1, row.x2, row.y2], dtype=float)
                best_track = ""
                best_iou = 0.0
                for tid, state in active.items():
                    if tid in assigned or int(frame_id) - int(state["frame_id"]) > max_gap:
                        continue
                    score = float(vector_iou(row_box, np.asarray([state["bbox"]], dtype=float))[0])
                    if score > best_iou:
                        best_iou, best_track = score, tid
                if best_track and best_iou >= link_iou:
                    tid = best_track
                    age = int(active[tid]["age"]) + 1
                    tracklet_ids[idx] = tid
                    temporal_age[idx] = age
                    k_values[idx] = max(0.05, min(1.0, best_iou))
                    active[tid] = {"bbox": row_box, "frame_id": int(frame_id), "age": age}
                    assigned.add(tid)
                else:
                    next_id += 1
                    tid = f"{seq}_{cls}_trk_{next_id:08d}"
                    tracklet_ids[idx] = tid
                    active[tid] = {"bbox": row_box, "frame_id": int(frame_id), "age": 0}
            active = {tid: st for tid, st in active.items() if int(frame_id) - int(st["frame_id"]) <= max_gap}
    d["tracklet_id"] = tracklet_ids
    d["temporal_age"] = temporal_age
    d["k_i"] = k_values
    d["s_i"] = 1.0
    d["Q_i"] = np.minimum(d["c_i"], d["k_i"])
    d["confidence_new"] = d["confidence"] * (0.6 + 0.4 * d["Q_i"])
    d["track_status"] = np.where(d["temporal_age"] <= 0, "new_candidate", "existing_track")
    return d


def method_acceptance(det: pd.DataFrame, method: str, params: Q1Params, rf_model: Any | None = None) -> pd.Series:
    m = method.lower()
    if det.empty:
        return pd.Series([], dtype=bool)
    if m in {"s_naive", "naive"}:
        return det["confidence"] >= params.naive_conf
    if m == "persistence_gate":
        return (det["confidence"] >= params.low_conf) & (det["temporal_age"] >= params.persistence_min_age)
    if m == "ema_confidence_gate":
        return ema_acceptance(det, params)
    if m == "bayesian_existence_filter":
        return bayesian_acceptance(det, params)
    if m in {"bytetrack", "sort"}:
        return bytetrack_acceptance(det, params)
    if m == "s2_logodds_temporal":
        return logodds_acceptance(det, params)
    if m == "geometry_dynamic_no_multiagent":
        conf_rw = det["confidence"] * (params.geom_q_floor + (1.0 - params.geom_q_floor) * np.maximum(det["k_i"], det["temporal_age"].clip(0, 3) / 3.0))
        return (conf_rw >= params.geom_threshold) | ((det["temporal_age"] >= 2) & (det["confidence"] >= params.low_conf))
    if m == "rf_learned_gate" and rf_model is not None:
        fp_prob = rf_model.predict_proba(feature_matrix(det))[:, 1]
        return pd.Series(fp_prob < params.rf_threshold, index=det.index)
    return pd.Series(False, index=det.index)


def ema_acceptance(det: pd.DataFrame, params: Q1Params) -> pd.Series:
    out = pd.Series(False, index=det.index)
    for _, group in det.sort_values(["sequence_id", "frame_id"]).groupby("tracklet_id", sort=False):
        trust = 0.0
        for idx, row in group.sort_values("frame_id").iterrows():
            trust = params.ema_lambda * trust + (1.0 - params.ema_lambda) * float(row.confidence)
            out.loc[idx] = trust >= params.ema_threshold
    return out


def bayesian_acceptance(det: pd.DataFrame, params: Q1Params) -> pd.Series:
    # Beta-Bernoulli persistence proxy: repeated observations increase existence probability.
    age = det["temporal_age"].astype(float)
    trust = (1.0 + age) / (2.0 + age)
    return (trust * det["confidence"].astype(float)) >= params.bayes_threshold


def bytetrack_acceptance(det: pd.DataFrame, params: Q1Params) -> pd.Series:
    high = det["confidence"] >= params.high_conf
    recovered = (det["confidence"] >= params.low_conf) & (det["temporal_age"] >= 1) & (det["k_i"] >= 0.25)
    return high | recovered


def logodds_acceptance(det: pd.DataFrame, params: Q1Params) -> pd.Series:
    out = pd.Series(False, index=det.index)
    for _, group in det.sort_values(["sequence_id", "frame_id"]).groupby("tracklet_id", sort=False):
        logodds = params.logodds_start
        for idx, row in group.sort_values("frame_id").iterrows():
            if float(row.confidence) >= 0.40 and float(row.k_i) >= 0.30:
                logodds += params.logodds_strong_reward
            elif float(row.confidence) >= 0.25 or float(row.k_i) >= 0.15:
                logodds += params.logodds_weak_reward
            out.loc[idx] = logodds >= params.logodds_confirm
    return out


def feature_matrix(det: pd.DataFrame) -> np.ndarray:
    cols = ["c_i", "k_i", "confidence", "bbox_area", "bbox_aspect_ratio", "num_detections_in_frame", "temporal_age"]
    return det[cols].fillna(0).to_numpy(dtype=float)


def train_rf(det: pd.DataFrame, threshold_grid: list[float] | None = None) -> tuple[Any, float]:
    from sklearn.ensemble import RandomForestClassifier

    if det.empty:
        raise ValueError("Cannot train RF on empty calibration set")
    y = (~det["eval_is_tp"].astype(bool)).astype(int).to_numpy()
    model = RandomForestClassifier(n_estimators=80, max_depth=8, min_samples_leaf=4, class_weight="balanced", random_state=42)
    model.fit(feature_matrix(det), y)
    grid = threshold_grid or [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]
    score = model.predict_proba(feature_matrix(det))[:, 1]
    best_tau = 0.55
    best_score = -1e9
    for tau in grid:
        accepted = score < tau
        row = compute_metrics(det, accepted, expected_gt=int(det["eval_is_tp"].sum()), num_frames=max(1, det[["sequence_id", "frame_id"]].drop_duplicates().shape[0]), method="rf_calibration")
        objective = -row["FP"] + 0.25 * row["TP"] - 2.0 * row["FN"]
        if objective > best_score:
            best_score = objective
            best_tau = float(tau)
    return model, best_tau


def compute_metrics(det: pd.DataFrame, accepted: pd.Series | np.ndarray, expected_gt: int, num_frames: int, method: str) -> dict[str, Any]:
    acc = det[pd.Series(accepted, index=det.index).astype(bool)]
    tp = int(acc["eval_is_tp"].astype(bool).sum()) if len(acc) else 0
    fp = int((~acc["eval_is_tp"].astype(bool)).sum()) if len(acc) else 0
    fn = max(0, int(expected_gt) - tp)
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(EPS, precision + recall)
    false_new = count_false_new_tracks(acc)
    return {
        "method": method,
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "precision": precision,
        "recall": recall,
        "F1": f1,
        "IDF1": f1,
        "false_new_tracks": false_new,
        "false_new_tracks_per_100_frames": false_new / max(1, num_frames) * 100.0,
        "track_breaks": count_track_breaks(acc),
    }


def count_false_new_tracks(acc: pd.DataFrame) -> int:
    if acc.empty:
        return 0
    fp = acc[~acc["eval_is_tp"].astype(bool)]
    if fp.empty:
        return 0
    first = fp.sort_values("frame_id").groupby("tracklet_id", sort=False).head(1)
    return int(len(first))


def count_track_breaks(acc: pd.DataFrame) -> int:
    if acc.empty or "matched_gt_id" not in acc:
        return 0
    hits = acc[acc["eval_is_tp"].astype(bool) & acc["matched_gt_id"].astype(str).str.len().gt(0)]
    breaks = 0
    for _, group in hits.groupby(["sequence_id", "matched_gt_id"], sort=False):
        frames = sorted(set(int(x) for x in group["frame_id"]))
        for a, b in zip(frames, frames[1:]):
            if b - a > 1:
                breaks += 1
    return breaks


def evaluate_methods(det: pd.DataFrame, gt: pd.DataFrame, methods: list[str], params: Q1Params | None = None, rf_model: Any | None = None) -> pd.DataFrame:
    p = params or Q1Params()
    expected_gt = len(gt)
    num_frames = max(1, gt[["sequence_id", "frame_id"]].drop_duplicates().shape[0])
    rows = []
    for method in methods:
        start = time.perf_counter()
        model = rf_model
        if method.lower() == "rf_learned_gate" and model is None:
            rows.append({"method": method, "available": False, "TP": np.nan, "FP": np.nan, "FN": np.nan, "precision": np.nan, "recall": np.nan, "F1": np.nan, "IDF1": np.nan, "false_new_tracks": np.nan, "false_new_tracks_per_100_frames": np.nan, "track_breaks": np.nan, "runtime_ms_per_frame": np.nan})
            continue
        accepted = method_acceptance(det, method, p, model)
        row = compute_metrics(det, accepted, expected_gt, num_frames, method)
        row["available"] = True
        row["runtime_ms_per_frame"] = (time.perf_counter() - start) * 1000.0 / max(1, num_frames)
        rows.append(row)
    return pd.DataFrame(rows)


def make_chunk_split(gt: pd.DataFrame, chunk_size: int = 100) -> pd.DataFrame:
    rows = []
    for seq, group in gt.groupby("sequence_id", sort=True):
        frames = sorted(set(int(x) for x in group["frame_id"]))
        for frame in frames:
            chunk = (frame - min(frames)) // chunk_size
            rows.append({"sequence_id": seq, "frame_id": frame, "chunk_id": f"{seq}_chunk_{chunk:04d}", "chunk_index": int(chunk)})
    chunks = pd.DataFrame(rows).drop_duplicates()
    max_chunk = chunks.groupby("sequence_id")["chunk_index"].transform("max")
    chunks["split"] = np.where(chunks["chunk_index"] <= np.floor(max_chunk * 0.6), "calibration", "holdout")
    return chunks


def select_budget_chunks(chunks: pd.DataFrame, budget: float, seed: int) -> set[str]:
    cal = sorted(chunks[chunks["split"].eq("calibration")]["chunk_id"].unique())
    if budget <= 0 or not cal:
        return set()
    rng = random.Random(seed)
    n = max(1, int(round(len(cal) * min(1.0, budget))))
    return set(rng.sample(cal, min(n, len(cal))))


def save_model(path: str | Path, model: Any, threshold: float) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("wb") as f:
        pickle.dump({"model": model, "threshold": threshold}, f)


def load_model(path: str | Path) -> tuple[Any, float]:
    with Path(path).open("rb") as f:
        payload = pickle.load(f)
    return payload["model"], float(payload.get("threshold", 0.55))


def ensure_area_norm(det: pd.DataFrame) -> pd.DataFrame:
    d = det.copy()
    if "bbox_area_norm" in d:
        return d
    if "image_area" not in d:
        if "image_path" in d and d["image_path"].astype(str).str.len().gt(0).any():
            try:
                import cv2
                sizes: dict[str, float] = {}
                for path in sorted(set(d["image_path"].astype(str))):
                    img = cv2.imread(path)
                    if img is not None:
                        h, w = img.shape[:2]
                        sizes[path] = float(w * h)
                d["image_area"] = d["image_path"].astype(str).map(sizes).fillna(1920.0 * 1080.0)
            except Exception:
                d["image_area"] = 1920.0 * 1080.0
        else:
            d["image_area"] = 1920.0 * 1080.0
    d["bbox_area_norm"] = (d["bbox_area"].astype(float) / d["image_area"].astype(float).clip(lower=1)).clip(0, 1)
    return d


def calibration_adaptive_stats(det_cal: pd.DataFrame) -> dict[str, float]:
    d = ensure_area_norm(det_cal)
    per_frame = d.groupby(["sequence_id", "frame_id"])["det_id"].count()
    density_ref = float(np.percentile(per_frame.to_numpy(dtype=float), 95)) if len(per_frame) else 1.0
    return {
        "density_ref": max(1.0, density_ref),
        "small_area_q": float(d["bbox_area_norm"].quantile(0.25)) if len(d) else 0.001,
        "medium_area_q": float(d["bbox_area_norm"].quantile(0.60)) if len(d) else 0.01,
    }


def density_adaptive_threshold(num_detections: pd.Series, cfg: dict[str, Any]) -> pd.Series:
    density_ref = max(EPS, float(cfg.get("density_ref", 1.0)))
    factor = (num_detections.astype(float) / density_ref).clip(0, 1)
    threshold = float(cfg.get("base_threshold", 0.50)) - float(cfg.get("density_gain", 0.08)) * factor
    return threshold.clip(float(cfg.get("min_threshold", 0.40)), float(cfg.get("max_threshold", 0.60)))


def area_q_floor(area_norm: pd.Series, cfg: dict[str, Any]) -> pd.Series:
    small_q = float(cfg.get("small_area_q", area_norm.quantile(0.25) if len(area_norm) else 0.001))
    medium_q = float(cfg.get("medium_area_q", area_norm.quantile(0.60) if len(area_norm) else 0.01))
    small_floor = float(cfg.get("small_q_floor", 0.75))
    medium_floor = float(cfg.get("medium_q_floor", 0.65))
    large_floor = float(cfg.get("large_q_floor", 0.55))
    values = np.where(area_norm < small_q, small_floor, np.where(area_norm < medium_q, medium_floor, large_floor))
    return pd.Series(values, index=area_norm.index)


def adaptive_geometry_acceptance(det: pd.DataFrame, cfg: dict[str, Any]) -> pd.Series:
    if det.empty:
        return pd.Series([], dtype=bool)
    d = ensure_area_norm(det)
    c = d["c_i"].astype(float).clip(0, 1)
    k = d["k_i"].astype(float).clip(0, 1)
    threshold_iou = max(EPS, float(cfg.get("temporal_support_threshold", 0.30)))
    temporal_available = d["temporal_age"].astype(float) > 0
    temporal_support = (k / threshold_iou).clip(0, 1)
    k_eff = np.maximum(k, temporal_support.where(temporal_available, k))
    aggregator = str(cfg.get("aggregator", "min"))
    if aggregator == "geometric_mean":
        alpha = float(cfg.get("alpha", 0.35))
        beta = float(cfg.get("beta", 0.35))
        delta = float(cfg.get("delta", 0.30))
        t_eff = temporal_support.where(temporal_available, 1.0).astype(float).clip(EPS, 1)
        q = ((c.clip(EPS, 1) ** alpha) * (pd.Series(k_eff, index=d.index).clip(EPS, 1) ** beta) * (t_eff ** delta)) ** (1.0 / max(EPS, alpha + beta + delta))
    else:
        q = np.minimum(c, k_eff)
    q_floor = area_q_floor(d["bbox_area_norm"].astype(float), cfg)
    conf_rw = d["confidence"].astype(float) * (q_floor + (1.0 - q_floor) * pd.Series(q, index=d.index))
    threshold = density_adaptive_threshold(d["num_detections_in_frame"], cfg)
    accepted = conf_rw >= threshold
    if bool(cfg.get("use_recovery", True)):
        recovery = (d["temporal_age"].astype(float) >= float(cfg.get("recovery_min_age", 2))) & (d["confidence"].astype(float) >= float(cfg.get("recovery_conf", 0.10)))
        accepted = accepted | recovery
    return pd.Series(accepted, index=d.index)


def add_adaptive_metrics(det: pd.DataFrame, gt: pd.DataFrame, cfg: dict[str, Any], method: str) -> dict[str, Any]:
    accepted = adaptive_geometry_acceptance(det, cfg)
    return compute_metrics(det, accepted, len(gt), max(1, gt[["sequence_id", "frame_id"]].drop_duplicates().shape[0]), method)


def pareto_flags(frame: pd.DataFrame, f1_col: str = "F1", false_col: str = "false_new_tracks") -> pd.Series:
    flags = []
    values = frame[[f1_col, false_col]].to_numpy(dtype=float)
    for i, (f1, false_new) in enumerate(values):
        dominated = False
        for j, (other_f1, other_false) in enumerate(values):
            if i == j:
                continue
            if other_f1 >= f1 and other_false <= false_new and (other_f1 > f1 or other_false < false_new):
                dominated = True
                break
        flags.append(not dominated)
    return pd.Series(flags, index=frame.index)
