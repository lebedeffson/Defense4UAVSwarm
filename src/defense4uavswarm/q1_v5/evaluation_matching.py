from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from defense4uavswarm.q1_visdrone import annotate_ignored_detections, compatible_by_mode
from defense4uavswarm.q1_v5.initiation_gate import GateResult
from defense4uavswarm.v8_sim import vector_iou


@dataclass(frozen=True)
class MatchingConfig:
    iou_threshold: float = 0.5
    class_matching: str = "coarse_class"
    ignored_policy: str = "exclude_ignored"
    matcher_id: str = "q1_v542_max_cardinality_iou_v1"
    confirmation_window_frames: int = 3


@dataclass(frozen=True)
class EvaluationResult:
    matched_detections: pd.DataFrame
    episode_events: pd.DataFrame
    gt_confirmation_events: pd.DataFrame
    summary: dict[str, float | int | str]


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


def evaluate_gate_result(
    candidates: pd.DataFrame,
    gate_result: GateResult,
    gt: pd.DataFrame,
    ignored: pd.DataFrame | None,
    frame_manifest: pd.DataFrame,
    config: MatchingConfig,
) -> EvaluationResult:
    if not gate_result.acceptance_mask.index.equals(candidates.index):
        raise ValueError("GateResult acceptance_mask index must match candidates index")
    if "episode_id" not in candidates.columns:
        raise ValueError("candidates must include episode_id before evaluation")
    _assert_candidate_contract(candidates, frame_manifest)
    accepted_series = gate_result.acceptance_mask.astype(bool)
    keep = [
        "det_id",
        "sequence_id",
        "frame_id",
        "tracklet_id",
        "episode_id",
        "class_id",
        "class_name",
        "confidence",
        "x1",
        "y1",
        "x2",
        "y2",
        "image_width",
        "image_height",
    ]
    det = candidates.loc[accepted_series, keep].copy()
    det = _prepare_accepted(det)
    matched = _recompute_one_to_one_matches(det, gt, config, ignored)
    episode_events = episode_event_table_from_gate(matched, gate_result.episode_metadata, config)
    gt_events = gt_confirmation_table(gt, episode_events)
    metrics = corrected_metrics(matched, gt, frame_manifest, episode_events)
    if gate_result.online_acceptance_mask is not None:
        online_series = gate_result.online_acceptance_mask.astype(bool)
        if not online_series.index.equals(candidates.index):
            raise ValueError("GateResult online_acceptance_mask index must match candidates index")
        online_det = candidates.loc[online_series, keep].copy()
        online_det = _prepare_accepted(online_det)
        online_matched = _recompute_one_to_one_matches(online_det, gt, config, ignored)
        online_events = episode_event_table_from_gate(online_matched, gate_result.episode_metadata, config)
        online_metrics = corrected_metrics(online_matched, gt, frame_manifest, online_events)
        for key, value in online_metrics.items():
            if isinstance(value, (int, float, np.integer, np.floating)):
                metrics[f"online_current_frame_{key}"] = value
    for key in ["TP", "FP", "FN", "precision", "recall", "F1"]:
        metrics[f"backfilled_map_{key}"] = metrics[key]
    _add_gt_confirmation_metrics(metrics, gt_events)
    _add_quarantine_budget_metrics(metrics, gate_result.episode_metadata)
    metrics["matcher_id"] = config.matcher_id
    metrics["metric_source"] = "recomputed_after_acceptance"
    metrics["occupancy_scope"] = "observed_detection_rows_proxy"
    metrics["method_id"] = gate_result.method_id
    metrics["parameter_json"] = gate_result.parameter_json
    return EvaluationResult(matched, episode_events, gt_events, metrics)


def _assert_candidate_contract(candidates: pd.DataFrame, frame_manifest: pd.DataFrame) -> None:
    required = {"det_id", "sequence_id", "frame_id", "tracklet_id", "episode_id", "image_width", "image_height"}
    missing = required - set(candidates.columns)
    if missing:
        raise ValueError(f"candidates missing required columns: {sorted(missing)}")
    key = ["sequence_id", "tracklet_id", "episode_id", "frame_id"]
    if candidates.duplicated(key).any():
        bad = candidates.loc[candidates.duplicated(key, keep=False), key].head(5)
        raise ValueError(f"Duplicate episode observation key: {bad.to_dict(orient='records')}")
    manifest_key = frame_manifest[["sequence_id", "frame_id", "image_width", "image_height"]].drop_duplicates()
    joined = candidates[["sequence_id", "frame_id", "image_width", "image_height"]].merge(
        manifest_key,
        on=["sequence_id", "frame_id"],
        how="left",
        suffixes=("", "_manifest"),
    )
    if joined[["image_width_manifest", "image_height_manifest"]].isna().any().any():
        raise ValueError("candidate frame missing from frame manifest")
    if not (
        joined["image_width"].astype(int).eq(joined["image_width_manifest"].astype(int))
        & joined["image_height"].astype(int).eq(joined["image_height_manifest"].astype(int))
    ).all():
        raise ValueError("candidate image dimensions do not match frame manifest")


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


def episode_event_table_from_gate(matched: pd.DataFrame, episode_metadata: pd.DataFrame, config: MatchingConfig) -> pd.DataFrame:
    cols = [
        "sequence_id",
        "tracklet_id",
        "episode_id",
        "first_candidate_frame_id",
        "confirmation_frame_id",
        "rejection_frame_id",
        "terminal_status",
        "gate_delay_frames",
        "num_observed_frames_before_terminal",
        "last_observed_frame_id",
        "accepted_active_frame_count",
        "is_false_initialization",
        "assigned_gt_key",
        "num_distinct_gt_matches_in_window",
        "identity_switch_in_window",
    ]
    if episode_metadata.empty:
        return pd.DataFrame(columns=cols)
    rows: list[dict[str, Any]] = []
    matched_by_key = {
        key: group
        for key, group in matched.groupby(["sequence_id", "tracklet_id", "episode_id"], sort=False)
    } if not matched.empty else {}
    for row in episode_metadata.itertuples(index=False):
        seq = str(row.sequence_id)
        tid = str(row.tracklet_id)
        episode_id = int(row.episode_id)
        confirmation = getattr(row, "confirmation_frame_id")
        confirmation = None if pd.isna(confirmation) else int(confirmation)
        group = matched_by_key.get((seq, tid, episode_id), pd.DataFrame())
        if confirmation is not None and not group.empty:
            window = group[
                (group["frame_id"].astype(int) >= confirmation)
                & (group["frame_id"].astype(int) < confirmation + config.confirmation_window_frames)
            ]
            gt_ids = sorted(set(str(x) for x in window.loc[window["eval_is_tp"].astype(bool), "matched_gt_id"] if str(x)))
        else:
            gt_ids = []
        rows.append(
            {
                "sequence_id": seq,
                "tracklet_id": tid,
                "episode_id": episode_id,
                "first_candidate_frame_id": int(row.first_candidate_frame_id),
                "confirmation_frame_id": confirmation,
                "rejection_frame_id": None if pd.isna(getattr(row, "rejection_frame_id")) else int(getattr(row, "rejection_frame_id")),
                "terminal_status": str(row.terminal_status),
                "gate_delay_frames": None if pd.isna(getattr(row, "gate_delay_frames")) else int(getattr(row, "gate_delay_frames")),
                "num_observed_frames_before_terminal": int(row.num_observed_frames_before_terminal),
                "last_observed_frame_id": int(row.last_observed_frame_id),
                "accepted_active_frame_count": int(group["frame_id"].nunique()) if not group.empty else 0,
                "is_false_initialization": bool(str(row.terminal_status) == "confirmed" and len(gt_ids) == 0),
                "assigned_gt_key": gt_ids[0] if gt_ids else "",
                "num_distinct_gt_matches_in_window": len(gt_ids),
                "identity_switch_in_window": len(gt_ids) > 1,
            }
        )
    return pd.DataFrame(rows, columns=cols)


def gt_confirmation_table(gt: pd.DataFrame, episode_events: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "sequence_id",
        "gt_track_id",
        "first_visible_frame_id",
        "first_confirmed_frame_id",
        "confirmation_delay_from_gt_frames",
        "was_ever_confirmed",
        "confirming_episode_key",
    ]
    if gt is None or gt.empty:
        return pd.DataFrame(columns=cols)
    gt_tracks = (
        gt.assign(gt_track_id=gt["object_id"].astype(str) if "object_id" in gt else gt["gt_track_id"].astype(str))
        .groupby(["sequence_id", "gt_track_id"], as_index=False)["frame_id"]
        .min()
        .rename(columns={"frame_id": "first_visible_frame_id"})
    )
    confirmed = episode_events[
        episode_events["terminal_status"].eq("confirmed")
        & episode_events["assigned_gt_key"].astype(str).str.len().gt(0)
    ].copy() if not episode_events.empty else pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for row in gt_tracks.itertuples(index=False):
        key = f"{row.sequence_id}:{row.gt_track_id}"
        hit = confirmed[confirmed["assigned_gt_key"].eq(key)] if not confirmed.empty else pd.DataFrame()
        if hit.empty:
            rows.append(
                {
                    "sequence_id": row.sequence_id,
                    "gt_track_id": row.gt_track_id,
                    "first_visible_frame_id": int(row.first_visible_frame_id),
                    "first_confirmed_frame_id": None,
                    "confirmation_delay_from_gt_frames": None,
                    "was_ever_confirmed": False,
                    "confirming_episode_key": "",
                }
            )
        else:
            first = hit.sort_values("confirmation_frame_id").iloc[0]
            delay = int(first["confirmation_frame_id"]) - int(row.first_visible_frame_id)
            rows.append(
                {
                    "sequence_id": row.sequence_id,
                    "gt_track_id": row.gt_track_id,
                    "first_visible_frame_id": int(row.first_visible_frame_id),
                    "first_confirmed_frame_id": int(first["confirmation_frame_id"]),
                    "confirmation_delay_from_gt_frames": int(delay),
                    "was_ever_confirmed": True,
                    "confirming_episode_key": f"{first['sequence_id']}:{first['tracklet_id']}:{int(first['episode_id'])}",
                }
            )
    return pd.DataFrame(rows, columns=cols)


def corrected_metrics(matched: pd.DataFrame, gt: pd.DataFrame, frame_index: pd.DataFrame, track_events: pd.DataFrame) -> dict[str, float]:
    gt_eval = _gt_in_frame_universe(gt, frame_index)
    tp = int(matched["eval_is_tp"].astype(bool).sum()) if not matched.empty else 0
    fp = int(matched["eval_counts_as_fp"].astype(bool).sum()) if not matched.empty and "eval_counts_as_fp" in matched else int((~matched["eval_is_tp"].astype(bool)).sum()) if not matched.empty else 0
    fn = max(0, int(len(gt_eval)) - tp)
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    if not track_events.empty and "terminal_status" in track_events:
        confirmed_events = track_events[track_events["terminal_status"].eq("confirmed")].copy()
    else:
        confirmed_events = track_events.copy()
    false_events = confirmed_events[confirmed_events["is_false_initialization"].astype(bool)] if not confirmed_events.empty else pd.DataFrame()
    true_events = confirmed_events[~confirmed_events["is_false_initialization"].astype(bool)] if not confirmed_events.empty else pd.DataFrame()
    if len(true_events) + len(false_events) != len(confirmed_events):
        raise AssertionError("confirmed initiation event partition is inconsistent")
    num_frames = max(1, len(frame_index.drop_duplicates(["sequence_id", "frame_id"]))) if frame_index is not None and not frame_index.empty else 1
    occ_col = "accepted_active_frame_count" if "accepted_active_frame_count" in track_events else "active_frame_count"
    observed_false_occ = int(false_events[occ_col].sum()) if not false_events.empty and occ_col in false_events else 0
    total_occ = int(track_events[occ_col].sum()) if not track_events.empty and occ_col in track_events else 0
    gate_delay_col = "gate_delay_frames" if "gate_delay_frames" in track_events else "gate_delay"
    track_initiation_precision = int(len(true_events)) / max(1, int(len(confirmed_events)))
    if not 0.0 <= track_initiation_precision <= 1.0:
        raise AssertionError("track_initiation_precision outside [0, 1]")
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
        "confirmed_episodes": int(len(confirmed_events)) if not confirmed_events.empty else 0,
        "track_initiation_precision": track_initiation_precision,
        "observed_false_track_occupancy_rows": observed_false_occ,
        "observed_false_track_occupancy_per_100_frames": observed_false_occ / num_frames * 100.0,
        "ignored_unmatched_detections": int(matched["ignored_unmatched_suppressed"].astype(bool).sum()) if not matched.empty and "ignored_unmatched_suppressed" in matched else 0,
        "observed_false_track_share": observed_false_occ / max(1, total_occ),
        "mean_observed_false_lifetime": float(false_events[occ_col].mean()) if not false_events.empty and occ_col in false_events else 0.0,
        "median_observed_false_lifetime": float(false_events[occ_col].median()) if not false_events.empty and occ_col in false_events else 0.0,
        "num_eval_frames": num_frames,
        "num_gt": int(len(gt_eval)),
        "track_breaks": _count_track_breaks(matched),
        "median_gate_delay_from_candidate": float(confirmed_events[gate_delay_col].dropna().median()) if not confirmed_events.empty and gate_delay_col in confirmed_events and confirmed_events[gate_delay_col].notna().any() else 0.0,
    }


def _add_quarantine_budget_metrics(metrics: dict[str, float | int | str], episode_metadata: pd.DataFrame) -> None:
    if episode_metadata.empty or "configured_quarantine_fraction" not in episode_metadata:
        return
    starts = int(len(episode_metadata))
    quarantined = int(episode_metadata.get("was_quarantined", pd.Series(False, index=episode_metadata.index)).fillna(False).astype(bool).sum())
    configured = float(episode_metadata["configured_quarantine_fraction"].dropna().max()) if episode_metadata["configured_quarantine_fraction"].notna().any() else 0.0
    realized = quarantined / max(1, starts)
    tolerance = 1.0 / max(1, starts)
    metrics["episode_starts"] = starts
    metrics["quarantined_starts"] = quarantined
    metrics["configured_quarantine_fraction"] = configured
    metrics["realized_quarantine_fraction"] = realized
    metrics["timeout_releases"] = int(episode_metadata.get("timeout_release_count", pd.Series(0, index=episode_metadata.index)).fillna(0).astype(int).sum())
    metrics["hard_veto_rejections"] = int(episode_metadata.get("hard_veto_rejection_count", pd.Series(0, index=episode_metadata.index)).fillna(0).astype(int).sum())
    metrics["budget_invariant_pass"] = bool(realized <= configured + tolerance)


def _add_gt_confirmation_metrics(metrics: dict[str, float | int | str], gt_events: pd.DataFrame) -> None:
    if gt_events.empty:
        metrics.update(
            {
                "true_track_confirmation_rate": 0.0,
                "mean_confirmation_delay_from_gt": 0.0,
                "median_confirmation_delay_from_gt": 0.0,
                "p95_confirmation_delay_from_gt": 0.0,
            }
        )
        return
    confirmed = gt_events[gt_events["was_ever_confirmed"].astype(bool)]
    delays = confirmed["confirmation_delay_from_gt_frames"].dropna().astype(float)
    metrics["true_track_confirmation_rate"] = float(len(confirmed) / max(1, len(gt_events)))
    metrics["mean_confirmation_delay_from_gt"] = float(delays.mean()) if not delays.empty else 0.0
    metrics["median_confirmation_delay_from_gt"] = float(delays.median()) if not delays.empty else 0.0
    metrics["p95_confirmation_delay_from_gt"] = float(delays.quantile(0.95)) if not delays.empty else 0.0


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
