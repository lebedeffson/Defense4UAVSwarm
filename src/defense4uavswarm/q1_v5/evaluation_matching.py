from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from defense4uavswarm.q1_visdrone import annotate_ignored_detections, compatible_by_mode
from defense4uavswarm.v8_sim import vector_iou


@dataclass(frozen=True)
class MatchingConfig:
    iou_threshold: float = 0.5
    class_matching: str = "coarse_class"
    ignored_policy: str = "exclude_ignored"
    matcher_id: str = "q1_v542_max_cardinality_iou_v1"
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
    det = _prepare_accepted(det)
    matched = _recompute_one_to_one_matches(det, gt, config, ignored)
    track_events = track_event_table(matched, config)
    metrics = corrected_metrics(matched, gt, frame_index, track_events)
    metrics["matcher_id"] = config.matcher_id
    metrics["metric_source"] = "recomputed_after_acceptance"
    metrics["occupancy_scope"] = "observed_detection_rows_proxy"
    return matched, track_events, metrics


def _prepare_accepted(det: pd.DataFrame) -> pd.DataFrame:
    if det.empty:
        out = det.copy()
        for col in ["eval_is_tp", "matched_gt_id", "matched_gt_iou", "inside_ignored_region", "ignored_iou", "ignored_unmatched_suppressed", "eval_counts_as_fp"]:
            if col not in out:
                out[col] = []
        return out
    d = det.copy().reset_index(drop=False).rename(columns={"index": "candidate_row_index"})
    d["ignored_unmatched_suppressed"] = False
    d["eval_counts_as_fp"] = False
    return d.reset_index(drop=True)


def _recompute_one_to_one_matches(det: pd.DataFrame, gt: pd.DataFrame, config: MatchingConfig, ignored: pd.DataFrame | None = None) -> pd.DataFrame:
    d = det.copy().reset_index(drop=True)
    d["eval_is_tp"] = False
    d["matched_gt_id"] = ""
    d["matched_gt_iou"] = 0.0
    d["ignored_unmatched_suppressed"] = False
    d["eval_counts_as_fp"] = False
    if config.ignored_policy not in {"exclude_ignored", "count_ignored", "report_ignored"}:
        raise ValueError(f"Unknown ignored_policy: {config.ignored_policy}")
    if d.empty:
        return d
    if gt is not None and not gt.empty:
        gt_by_key = {key: group.reset_index(drop=True) for key, group in gt.groupby(["sequence_id", "frame_id"], sort=False)}
    else:
        gt_by_key = {}
    for key, group in d.groupby(["sequence_id", "frame_id"], sort=False):
        g = gt_by_key.get(key)
        if g is not None and not g.empty:
            _match_frame_max_cardinality(d, group, g, config)
    d = _apply_ignore_to_unmatched(d, ignored, config)
    d["eval_counts_as_fp"] = (~d["eval_is_tp"].astype(bool)) & (~d["ignored_unmatched_suppressed"].astype(bool))
    return d


def _match_frame_max_cardinality(d: pd.DataFrame, group: pd.DataFrame, gt_frame: pd.DataFrame, config: MatchingConfig) -> None:
    det_indices = list(group.index)
    if not det_indices or gt_frame.empty:
        return
    det_boxes = group[["x1", "y1", "x2", "y2"]].to_numpy(dtype=float)
    gt_boxes = gt_frame[["x1", "y1", "x2", "y2"]].to_numpy(dtype=float)
    iou = np.vstack([vector_iou(box, gt_boxes) for box in det_boxes])
    compat = np.zeros_like(iou, dtype=bool)
    gt_names = gt_frame["class_name"].astype(str).str.lower().to_numpy()
    gt_class_ids = gt_frame["class_id"].astype(int).to_numpy() if "class_id" in gt_frame else np.full(len(gt_frame), -999)
    for r, row in enumerate(group.itertuples()):
        try:
            det_cls = int(row.class_id)
        except Exception:
            det_cls = None
        for c in range(len(gt_frame)):
            compat[r, c] = compatible_by_mode(str(row.class_name), det_cls, gt_names[c], int(gt_class_ids[c]), config.class_matching)
    valid = compat & (iou >= float(config.iou_threshold))
    if not valid.any():
        return
    large = float(max(len(det_indices), len(gt_frame)) + 1)
    score = np.where(valid, large + iou, 0.0)
    row_ind, col_ind = linear_sum_assignment(-score)
    gt_ids = (gt_frame["object_id"] if "object_id" in gt_frame else gt_frame.get("gt_track_id", pd.Series(range(len(gt_frame))))).astype(str).to_numpy()
    seq_value = str(group.iloc[0]["sequence_id"]) if "sequence_id" in group else ""
    for r, c in zip(row_ind, col_ind):
        if not valid[int(r), int(c)]:
            continue
        idx = int(det_indices[int(r)])
        d.loc[idx, "eval_is_tp"] = True
        d.loc[idx, "matched_gt_id"] = f"{seq_value}:{gt_ids[int(c)]}"
        d.loc[idx, "matched_gt_iou"] = float(iou[int(r), int(c)])


def _apply_ignore_to_unmatched(d: pd.DataFrame, ignored: pd.DataFrame | None, config: MatchingConfig) -> pd.DataFrame:
    if d.empty:
        return d
    annotated = annotate_ignored_detections(d, ignored if ignored is not None else pd.DataFrame())
    if config.ignored_policy == "exclude_ignored":
        annotated["ignored_unmatched_suppressed"] = (~annotated["eval_is_tp"].astype(bool)) & annotated["inside_ignored_region"].astype(bool)
    elif config.ignored_policy in {"count_ignored", "report_ignored"}:
        annotated["ignored_unmatched_suppressed"] = False
    return annotated


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
    group_cols = ["sequence_id", "tracklet_id"] + (["episode_id"] if "episode_id" in matched else [])
    for key, group in matched.sort_values(group_cols + ["frame_id"]).groupby(group_cols, sort=False):
        if not isinstance(key, tuple):
            key = (key,)
        seq = key[0]
        tid = key[1]
        episode_id = key[2] if len(key) > 2 else 0
        first = int(group["frame_id"].min())
        confirmation = first
        window = group[(group["frame_id"].astype(int) >= confirmation) & (group["frame_id"].astype(int) < confirmation + config.confirmation_window_frames)]
        gt_ids = sorted(set(str(x) for x in window.loc[window["eval_is_tp"].astype(bool), "matched_gt_id"] if str(x)))
        rows.append(
            {
                "sequence_id": seq,
                "tracklet_id": tid,
                "episode_id": episode_id,
                "first_candidate_frame_id": first,
                "confirmation_frame_id": confirmation,
                "gate_delay": int(confirmation - first),
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
    fp = int(matched["eval_counts_as_fp"].astype(bool).sum()) if not matched.empty and "eval_counts_as_fp" in matched else int((~matched["eval_is_tp"].astype(bool)).sum()) if not matched.empty else 0
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
        "false_initializations": int(len(false_events)),
        "false_new_tracks": int(len(false_events)),
        "false_new_tracks_per_100_frames": int(len(false_events)) / num_frames * 100.0,
        "true_initializations": int(len(true_events)),
        "track_initiation_precision": int(len(true_events)) / max(1, len(track_events)),
        "observed_false_track_occupancy_rows": observed_false_occ,
        "ignored_unmatched_detections": int(matched["ignored_unmatched_suppressed"].astype(bool).sum()) if not matched.empty and "ignored_unmatched_suppressed" in matched else 0,
        "observed_false_track_share": observed_false_occ / max(1, int(track_events["active_frame_count"].sum()) if not track_events.empty else 0),
        "mean_observed_false_lifetime": float(false_events["active_frame_count"].mean()) if not false_events.empty else 0.0,
        "median_observed_false_lifetime": float(false_events["active_frame_count"].median()) if not false_events.empty else 0.0,
        "num_eval_frames": num_frames,
        "num_gt": int(len(gt_eval)),
        "track_breaks": _count_track_breaks(matched),
        "median_gate_delay": float(track_events["gate_delay"].median()) if not track_events.empty and "gate_delay" in track_events else 0.0,
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
