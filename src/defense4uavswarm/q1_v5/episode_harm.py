from __future__ import annotations

import hashlib
import math
import subprocess
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from defense4uavswarm.q1_v5.evaluation_matching import MatchingConfig, evaluate_gate_result
from defense4uavswarm.q1_v5.initiation_gate import GateConfig, tracker_baseline_gate_result


PROTOCOL_ID = "q1_selective_quarantine_v22_risk_prioritized_two_stage"


@dataclass(frozen=True)
class EpisodeHarmInputs:
    candidates: pd.DataFrame
    gt: pd.DataFrame
    ignored: pd.DataFrame
    frame_manifest: pd.DataFrame
    matching: MatchingConfig
    gate: GateConfig
    tracker: str = "bytetrack"


def build_episode_harm_table(inputs: EpisodeHarmInputs) -> pd.DataFrame:
    candidates = _ensure_episode_features(inputs.candidates)
    baseline = tracker_baseline_gate_result(candidates, inputs.gate)
    evaluated = evaluate_gate_result(candidates, baseline, inputs.gt, inputs.ignored, inputs.frame_manifest, inputs.matching)
    matched = evaluated.matched_detections.copy()
    if "candidate_row_index" not in matched:
        raise ValueError("corrected evaluator must preserve candidate_row_index")
    false_by_index = {
        int(row.candidate_row_index): bool((not bool(row.eval_is_tp)) and bool(row.eval_counts_as_fp))
        for row in matched.itertuples(index=False)
    }
    tp_by_index = {int(row.candidate_row_index): bool(row.eval_is_tp) for row in matched.itertuples(index=False)}
    rows: list[dict[str, Any]] = []
    feature_rows = _past_only_start_features(candidates, inputs.tracker)
    feature_by_key = {
        (str(r.sequence_id), str(r.tracklet_id), int(r.episode_id)): r._asdict()
        for r in feature_rows.itertuples(index=False)
    }
    for key, group in candidates.sort_values(["sequence_id", "tracklet_id", "episode_id", "frame_id"]).groupby(["sequence_id", "tracklet_id", "episode_id"], sort=False):
        seq, tid, episode_id = str(key[0]), str(key[1]), int(key[2])
        g = group.sort_values("frame_id").copy()
        first = g.iloc[0]
        second = g.iloc[1] if len(g) > 1 else None
        idxs = [int(i) for i in g.index]
        matched_tp_rows = sum(1 for idx in idxs if tp_by_index.get(idx, False))
        false_rows = [int(g.loc[idx, "frame_id"]) for idx in idxs if false_by_index.get(idx, False)]
        first_frame = int(first.frame_id)
        false_1 = sum(1 for f in false_rows if f < first_frame + 1)
        false_3 = sum(1 for f in false_rows if f < first_frame + 3)
        false_5 = sum(1 for f in false_rows if f < first_frame + 5)
        false_10 = sum(1 for f in false_rows if f < first_frame + 10)
        is_false = matched_tp_rows == 0
        row = {
            "sequence_id": seq,
            "tracklet_id": tid,
            "episode_id": episode_id,
            "first_frame_id": first_frame,
            "second_frame_id": int(second.frame_id) if second is not None else np.nan,
            "last_frame_id": int(g["frame_id"].max()),
            "source_commit": git(["rev-parse", "HEAD"]),
            "matcher_id": inputs.matching.matcher_id,
            "protocol_id": PROTOCOL_ID,
            "tracker_profile": inputs.tracker,
            "has_any_valid_gt_match": bool(matched_tp_rows > 0),
            "matched_tp_rows": int(matched_tp_rows),
            "unmatched_false_rows": int(sum(1 for idx in idxs if false_by_index.get(idx, False))),
            "false_rows_first_1_frame": int(false_1),
            "false_rows_first_3_frames": int(false_3),
            "false_rows_first_5_frames": int(false_5),
            "false_rows_first_10_frames": int(false_10),
            "episode_lifetime_frames": int(g["frame_id"].max() - g["frame_id"].min() + 1),
            "episode_row_count": int(len(g)),
            "is_false_episode": bool(is_false),
            "harm_target_h3": int(false_3 if is_false else 0),
            "harm_target_h5": int(false_5 if is_false else 0),
            "harm_target_h10": int(false_10 if is_false else 0),
            **feature_by_key.get((seq, tid, episode_id), {}),
            **_stage2_features(g),
        }
        rows.append(row)
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["sequence_id", "first_frame_id", "tracklet_id", "episode_id"]).reset_index(drop=True)
    return out


def _ensure_episode_features(candidates: pd.DataFrame) -> pd.DataFrame:
    out = candidates.copy()
    if "bbox_area_norm" not in out:
        area = ((out["x2"].astype(float) - out["x1"].astype(float)).clip(lower=1) * (out["y2"].astype(float) - out["y1"].astype(float)).clip(lower=1))
        out["bbox_area_norm"] = area / (out["image_width"].astype(float) * out["image_height"].astype(float)).clip(lower=1)
    if "bbox_aspect_ratio" not in out:
        out["bbox_aspect_ratio"] = (out["x2"].astype(float) - out["x1"].astype(float)).clip(lower=1) / (out["y2"].astype(float) - out["y1"].astype(float)).clip(lower=1)
    if "num_detections_in_frame" not in out:
        out["num_detections_in_frame"] = out.groupby(["sequence_id", "frame_id"])["tracklet_id"].transform("count")
    return out


def _past_only_start_features(candidates: pd.DataFrame, tracker: str) -> pd.DataFrame:
    starts = candidates.sort_values(["sequence_id", "tracklet_id", "episode_id", "frame_id"]).groupby(["sequence_id", "tracklet_id", "episode_id"], as_index=False, sort=False).first()
    counts = candidates.groupby(["sequence_id", "tracklet_id", "episode_id"], as_index=False).size().rename(columns={"size": "episode_row_count"})
    starts = starts.merge(counts, on=["sequence_id", "tracklet_id", "episode_id"], how="left")
    rows: list[dict[str, Any]] = []
    for seq, group in starts.sort_values(["sequence_id", "frame_id", "tracklet_id", "episode_id"]).groupby("sequence_id", sort=False):
        past_conf: list[float] = []
        past_support: list[float] = []
        past_survival: list[int] = []
        by_frame = {int(f): fr.copy() for f, fr in group.groupby("frame_id", sort=True)}
        for frame_id in sorted(by_frame):
            frame = by_frame[frame_id].copy().sort_values(["tracklet_id", "episode_id"])
            supports = [_absolute_support(float(r.confidence)) for r in frame.itertuples(index=False)]
            ranks = [_rank(float(r.confidence), past_conf) for r in frame.itertuples(index=False)]
            threshold = float(np.quantile(past_support, 0.10)) if past_support else 0.0
            survival_rate = float(np.mean(past_survival[-200:])) if past_survival else 0.25
            median_support = float(np.median(supports)) if supports else 0.0
            for support, rank, row in zip(supports, ranks, frame.itertuples(index=False)):
                area = float(row.bbox_area_norm)
                rows.append(
                    {
                        "sequence_id": str(row.sequence_id),
                        "tracklet_id": str(row.tracklet_id),
                        "episode_id": int(row.episode_id),
                        "confidence_start": float(row.confidence),
                        "absolute_support_start": support,
                        "relative_confidence_start": rank,
                        "episode_start_rank": _rank(support, past_support),
                        "bbox_area_norm_start": area,
                        "log_bbox_area_norm_start": math.log(max(area, 1e-12)),
                        "aspect_ratio_start": float(row.bbox_aspect_ratio),
                        "border_distance_norm_start": _border_distance(row),
                        "class_name_start": str(getattr(row, "class_name", "unknown")),
                        "simultaneous_episode_starts": int(len(frame)),
                        "frame_candidate_count": int(getattr(row, "num_detections_in_frame", len(frame))),
                        "frame_median_support": median_support,
                        "past_survival_rate": survival_rate,
                        "past_quarantine_threshold": threshold,
                        "tracker_profile": tracker,
                    }
                )
            past_conf.extend(float(r.confidence) for r in frame.itertuples(index=False))
            past_support.extend(supports)
            # survival is known from the frozen episode rows, but only after the current
            # frame; append after emitting features to avoid same-frame leakage.
            for row in frame.itertuples(index=False):
                past_survival.append(1 if int(getattr(row, "episode_row_count", 1)) >= 2 else 0)
    return pd.DataFrame(rows)


def _stage2_features(group: pd.DataFrame) -> dict[str, Any]:
    g = group.sort_values("frame_id").copy()
    first = g.iloc[0]
    if len(g) < 2:
        return {
            "second_observation_gap_frames": np.nan,
            "confidence_second": np.nan,
            "support_second": np.nan,
            "confidence_delta": np.nan,
            "support_delta": np.nan,
            "bbox_iou_1_2": np.nan,
            "center_displacement_norm_1_2": np.nan,
            "scale_ratio_1_2": np.nan,
            "aspect_ratio_delta_1_2": np.nan,
            "class_consistent_1_2": False,
            "border_distance_delta_1_2": np.nan,
            "local_new_start_density_second_frame": np.nan,
        }
    second = g.iloc[1]
    s1 = _absolute_support(float(first.confidence))
    s2 = _absolute_support(float(second.confidence))
    return {
        "second_observation_gap_frames": int(second.frame_id) - int(first.frame_id),
        "confidence_second": float(second.confidence),
        "support_second": s2,
        "confidence_delta": float(second.confidence) - float(first.confidence),
        "support_delta": s2 - s1,
        "bbox_iou_1_2": _iou(first, second),
        "center_displacement_norm_1_2": _center_displacement(first, second),
        "scale_ratio_1_2": _area(second) / max(1.0, _area(first)),
        "aspect_ratio_delta_1_2": float(second.bbox_aspect_ratio) - float(first.bbox_aspect_ratio),
        "class_consistent_1_2": str(second.class_name) == str(first.class_name),
        "border_distance_delta_1_2": _border_distance(second) - _border_distance(first),
        "local_new_start_density_second_frame": int(second.num_detections_in_frame),
    }


def _absolute_support(confidence: float, floor: float = 0.05) -> float:
    return max(0.0, min(1.0, (float(confidence) - floor) / max(1e-12, 1.0 - floor)))


def _rank(value: float, history: list[float]) -> float:
    if not history:
        return 0.5
    return float(sum(1 for x in history if x <= float(value)) / len(history))


def _area(row: Any) -> float:
    return max(1.0, float(row.x2) - float(row.x1)) * max(1.0, float(row.y2) - float(row.y1))


def _iou(a: Any, b: Any) -> float:
    x1 = max(float(a.x1), float(b.x1))
    y1 = max(float(a.y1), float(b.y1))
    x2 = min(float(a.x2), float(b.x2))
    y2 = min(float(a.y2), float(b.y2))
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    return float(inter / max(1e-12, _area(a) + _area(b) - inter))


def _center_displacement(a: Any, b: Any) -> float:
    ax, ay = (float(a.x1) + float(a.x2)) / 2.0, (float(a.y1) + float(a.y2)) / 2.0
    bx, by = (float(b.x1) + float(b.x2)) / 2.0, (float(b.y1) + float(b.y2)) / 2.0
    return float(math.hypot(ax - bx, ay - by) / max(1.0, math.sqrt(_area(a))))


def _border_distance(row: Any) -> float:
    w, h = max(1.0, float(row.image_width)), max(1.0, float(row.image_height))
    return float(min(float(row.x1), float(row.y1), w - float(row.x2), h - float(row.y2)) / max(w, h))


def git(args: list[str]) -> str:
    try:
        return subprocess.check_output(["git", *args], text=True).strip()
    except Exception:
        return "unavailable"


def sha256_frame(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
