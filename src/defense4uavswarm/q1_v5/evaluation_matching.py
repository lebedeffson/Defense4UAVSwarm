from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from defense4uavswarm.q1_visdrone import annotate_ignored_detections, compatible_by_mode
from defense4uavswarm.v8_sim import vector_iou


@dataclass(frozen=True)
class MatchingConfig:
    iou_threshold: float = 0.5
    class_matching: str = "coarse_class"
    ignored_policy: str = "exclude_ignored"
    matcher_id: str = "q1_v54_recomputed_greedy_confidence_v1"
    confirmation_window_frames: int = 3


def build_frame_index(gt: pd.DataFrame, candidates: pd.DataFrame | None = None) -> pd.DataFrame:
    frames = []
    if gt is not None and not gt.empty:
        frames.append(gt[["sequence_id", "frame_id"]])
    if candidates is not None and not candidates.empty:
        frames.append(candidates[["sequence_id", "frame_id"]])
    if not frames:
        return pd.DataFrame(columns=["sequence_id", "frame_id"])
    return pd.concat(frames, ignore_index=True).drop_duplicates().sort_values(["sequence_id", "frame_id"]).reset_index(drop=True)


def evaluate_acceptance_mask(
    candidates: pd.DataFrame,
    accepted: pd.Series | np.ndarray,
    gt: pd.DataFrame,
    frame_index: pd.DataFrame,
    config: MatchingConfig,
    ignored: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    accepted_series = pd.Series(accepted, index=candidates.index).astype(bool)
    keep = [
        "det_id",
        "sequence_id",
        "frame_id",
        "tracklet_id",
        "class_id",
        "class_name",
        "confidence",
        "x1",
        "y1",
        "x2",
        "y2",
    ]
    optional = [c for c in ["image_path", "detector"] if c in candidates]
    det = candidates.loc[accepted_series, [c for c in keep + optional if c in candidates]].copy()
    det = _prepare_accepted(det, config, ignored)
    matched = _recompute_one_to_one_matches(det, gt, config)
    track_events = track_event_table(matched, config)
    metrics = corrected_metrics(matched, gt, frame_index, track_events)
    metrics["matcher_id"] = config.matcher_id
    metrics["metric_source"] = "recomputed_after_acceptance"
    metrics["occupancy_scope"] = "observed_detection_rows_proxy"
    return matched, track_events, metrics


def _prepare_accepted(det: pd.DataFrame, config: MatchingConfig, ignored: pd.DataFrame | None) -> pd.DataFrame:
    if det.empty:
        out = det.copy()
        for col in ["eval_is_tp", "matched_gt_id", "matched_gt_iou", "inside_ignored_region", "ignored_iou"]:
            if col not in out:
                out[col] = []
        return out
    d = det.copy().reset_index(drop=False).rename(columns={"index": "candidate_row_index"})
    d = annotate_ignored_detections(d, ignored if ignored is not None else pd.DataFrame())
    if config.ignored_policy == "exclude_ignored":
        d = d[~d["inside_ignored_region"].astype(bool)].copy()
    elif config.ignored_policy not in {"count_ignored", "report_ignored"}:
        raise ValueError(f"Unknown ignored_policy: {config.ignored_policy}")
    return d.reset_index(drop=True)


def _recompute_one_to_one_matches(det: pd.DataFrame, gt: pd.DataFrame, config: MatchingConfig) -> pd.DataFrame:
    d = det.copy().reset_index(drop=True)
    d["eval_is_tp"] = False
    d["matched_gt_id"] = ""
    d["matched_gt_iou"] = 0.0
    if d.empty or gt is None or gt.empty:
        return d
    gt_by_key = {key: group.reset_index(drop=True) for key, group in gt.groupby(["sequence_id", "frame_id"], sort=False)}
    for key, group in d.groupby(["sequence_id", "frame_id"], sort=False):
        g = gt_by_key.get(key)
        if g is None or g.empty:
            continue
        gt_boxes = g[["x1", "y1", "x2", "y2"]].to_numpy(dtype=float)
        gt_names = g["class_name"].astype(str).str.lower().to_numpy()
        gt_class_ids = g["class_id"].astype(int).to_numpy() if "class_id" in g else np.full(len(g), -999)
        gt_ids = (g["object_id"] if "object_id" in g else g.get("gt_track_id", pd.Series(range(len(g))))).astype(str).to_numpy()
        used = np.zeros(len(g), dtype=bool)
        for row in group.sort_values(["confidence", "det_id"], ascending=[False, True]).itertuples():
            idx = int(row.Index)
            try:
                det_cls = int(row.class_id)
            except Exception:
                det_cls = None
            compatible = np.array(
                [
                    compatible_by_mode(str(row.class_name), det_cls, gt_names[i], int(gt_class_ids[i]), config.class_matching)
                    for i in range(len(g))
                ],
                dtype=bool,
            ) & (~used)
            candidates = np.flatnonzero(compatible)
            if len(candidates) == 0:
                continue
            box = np.asarray([row.x1, row.y1, row.x2, row.y2], dtype=float)
            ious = vector_iou(box, gt_boxes[candidates])
            best_pos = int(np.argmax(ious))
            if float(ious[best_pos]) >= config.iou_threshold:
                gt_idx = int(candidates[best_pos])
                used[gt_idx] = True
                d.loc[idx, "eval_is_tp"] = True
                d.loc[idx, "matched_gt_id"] = str(gt_ids[gt_idx])
                d.loc[idx, "matched_gt_iou"] = float(ious[best_pos])
    return d


def track_event_table(matched: pd.DataFrame, config: MatchingConfig) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if matched.empty or "tracklet_id" not in matched:
        return pd.DataFrame(
            columns=[
                "sequence_id",
                "tracklet_id",
                "confirmation_frame_id",
                "first_candidate_frame_id",
                "is_false_initialization",
                "assigned_gt_key",
                "num_distinct_gt_matches_in_window",
                "identity_switch_in_window",
                "active_frame_count",
            ]
        )
    for (seq, tid), group in matched.sort_values(["sequence_id", "tracklet_id", "frame_id"]).groupby(["sequence_id", "tracklet_id"], sort=False):
        first = int(group["frame_id"].min())
        confirmation = first
        window = group[(group["frame_id"].astype(int) >= confirmation) & (group["frame_id"].astype(int) < confirmation + config.confirmation_window_frames)]
        gt_ids = sorted(set(str(x) for x in window.loc[window["eval_is_tp"].astype(bool), "matched_gt_id"] if str(x)))
        rows.append(
            {
                "sequence_id": seq,
                "tracklet_id": tid,
                "first_candidate_frame_id": first,
                "confirmation_frame_id": confirmation,
                "is_false_initialization": len(gt_ids) == 0,
                "assigned_gt_key": gt_ids[0] if gt_ids else "",
                "num_distinct_gt_matches_in_window": len(gt_ids),
                "identity_switch_in_window": len(gt_ids) > 1,
                "active_frame_count": int(group["frame_id"].nunique()),
                "last_observed_frame_id": int(group["frame_id"].max()),
            }
        )
    return pd.DataFrame(rows)


def corrected_metrics(matched: pd.DataFrame, gt: pd.DataFrame, frame_index: pd.DataFrame, track_events: pd.DataFrame) -> dict[str, float]:
    gt_eval = _gt_in_frame_universe(gt, frame_index)
    tp = int(matched["eval_is_tp"].astype(bool).sum()) if not matched.empty else 0
    fp = int((~matched["eval_is_tp"].astype(bool)).sum()) if not matched.empty else 0
    fn = max(0, int(len(gt_eval)) - tp)
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    false_events = track_events[track_events["is_false_initialization"].astype(bool)] if not track_events.empty else pd.DataFrame()
    true_events = track_events[~track_events["is_false_initialization"].astype(bool)] if not track_events.empty else pd.DataFrame()
    num_frames = max(1, len(frame_index.drop_duplicates(["sequence_id", "frame_id"]))) if frame_index is not None and not frame_index.empty else 1
    observed_false_occ = int(false_events["active_frame_count"].sum()) if not false_events.empty else 0
    return {
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "precision": precision,
        "recall": recall,
        "F1": f1,
        "IDF1": f1,
        "false_initializations": int(len(false_events)),
        "false_new_tracks": int(len(false_events)),
        "false_new_tracks_per_100_frames": int(len(false_events)) / num_frames * 100.0,
        "true_initializations": int(len(true_events)),
        "track_initiation_precision": int(len(true_events)) / max(1, len(track_events)),
        "observed_false_track_occupancy_frames": observed_false_occ,
        "false_track_occupancy_frames": observed_false_occ,
        "observed_false_track_share": observed_false_occ / max(1, int(track_events["active_frame_count"].sum()) if not track_events.empty else 0),
        "mean_observed_false_lifetime": float(false_events["active_frame_count"].mean()) if not false_events.empty else 0.0,
        "median_observed_false_lifetime": float(false_events["active_frame_count"].median()) if not false_events.empty else 0.0,
        "num_eval_frames": num_frames,
        "num_gt": int(len(gt_eval)),
        "track_breaks": _count_track_breaks(matched),
    }


def _gt_in_frame_universe(gt: pd.DataFrame, frame_index: pd.DataFrame) -> pd.DataFrame:
    if gt is None or gt.empty:
        return pd.DataFrame()
    if frame_index is None or frame_index.empty:
        return gt.copy()
    return gt.merge(frame_index[["sequence_id", "frame_id"]].drop_duplicates(), on=["sequence_id", "frame_id"], how="inner")


def _count_track_breaks(matched: pd.DataFrame) -> int:
    if matched.empty or "matched_gt_id" not in matched:
        return 0
    hits = matched[matched["eval_is_tp"].astype(bool) & matched["matched_gt_id"].astype(str).str.len().gt(0)]
    breaks = 0
    for _, group in hits.groupby(["sequence_id", "matched_gt_id"], sort=False):
        frames = sorted(set(int(x) for x in group["frame_id"]))
        breaks += sum(1 for a, b in zip(frames, frames[1:]) if b - a > 1)
    return int(breaks)
