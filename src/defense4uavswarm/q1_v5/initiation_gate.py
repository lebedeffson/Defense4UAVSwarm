from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class GateConfig:
    maximum_pending_age_frames: int = 6
    max_track_gap: int = 1


@dataclass(frozen=True)
class GateEpisodeRecord:
    sequence_id: str
    tracklet_id: str
    episode_id: int
    first_candidate_frame_id: int
    confirmation_frame_id: int | None
    rejection_frame_id: int | None
    terminal_status: str
    gate_delay_frames: int | None
    num_observed_frames_before_terminal: int
    last_observed_frame_id: int


@dataclass(frozen=True)
class GateResult:
    acceptance_mask: pd.Series
    episode_metadata: pd.DataFrame
    method_id: str
    parameter_json: str


def assign_episode_ids(candidates: pd.DataFrame, max_track_gap: int = 1) -> pd.DataFrame:
    if candidates.empty:
        out = candidates.copy()
        out["episode_id"] = pd.Series(dtype=int)
        return out
    out = candidates.copy()
    out["episode_id"] = 0
    sort_cols = ["sequence_id", "tracklet_id", "frame_id"] + (["det_id"] if "det_id" in out else [])
    for _, group in out.sort_values(sort_cols).groupby(["sequence_id", "tracklet_id"], sort=False):
        episode_id = 0
        previous_frame: int | None = None
        for idx, row in group.iterrows():
            frame = int(row.frame_id)
            if previous_frame is not None and frame - previous_frame > int(max_track_gap) + 1:
                episode_id += 1
            out.loc[idx, "episode_id"] = episode_id
            previous_frame = frame
    return out


def split_duplicate_observation_tracklets(candidates: pd.DataFrame) -> pd.DataFrame:
    key = ["sequence_id", "tracklet_id", "frame_id"]
    if candidates.empty or not all(c in candidates.columns for c in key):
        return candidates.copy()
    out = candidates.copy()
    dup = out.duplicated(key, keep=False)
    if not dup.any():
        return out
    det_col = out["det_id"].astype(str) if "det_id" in out else out.index.astype(str)
    order = out.groupby(key, sort=False).cumcount()
    mask = dup & order.gt(0)
    out.loc[mask, "tracklet_id"] = out.loc[mask, "tracklet_id"].astype(str) + "__obs_" + det_col.loc[mask].str.replace(r"[^A-Za-z0-9_.-]", "_", regex=True)
    return out


def episode_group_columns(candidates: pd.DataFrame) -> list[str]:
    return ["sequence_id", "tracklet_id"] + (["episode_id"] if "episode_id" in candidates else [])


def gate_result_from_trigger(
    candidates: pd.DataFrame,
    trigger: pd.Series | np.ndarray,
    cfg: GateConfig | None = None,
    *,
    method_id: str = "terminal_gate",
    parameters: dict[str, Any] | None = None,
) -> GateResult:
    cfg = cfg or GateConfig()
    candidates = assign_episode_ids(candidates, cfg.max_track_gap) if "episode_id" not in candidates else candidates
    assert_unique_episode_observations(candidates)
    trigger_series = pd.Series(trigger, index=candidates.index).astype(bool)
    out = pd.Series(False, index=candidates.index)
    records: list[GateEpisodeRecord] = []
    group_cols = episode_group_columns(candidates)
    for _, group in candidates.sort_values(group_cols + ["frame_id"]).groupby(group_cols, sort=False):
        confirmed = False
        rejected = False
        first_frame = None
        last_frame = None
        confirmation_frame: int | None = None
        rejection_frame: int | None = None
        terminal_observed = 0
        sequence_id = str(group.iloc[0]["sequence_id"])
        tracklet_id = str(group.iloc[0]["tracklet_id"])
        episode_id = int(group.iloc[0].get("episode_id", 0))
        for idx, row in group.sort_values("frame_id").iterrows():
            frame = int(row.frame_id)
            if first_frame is None:
                first_frame = frame
            last_frame = frame
            age = frame - int(first_frame) + 1
            if confirmed:
                out.loc[idx] = True
                continue
            if rejected:
                out.loc[idx] = False
                continue
            if bool(trigger_series.loc[idx]):
                confirmed = True
                confirmation_frame = frame
                terminal_observed = int(group[group["frame_id"].astype(int) <= frame].shape[0])
                out.loc[idx] = True
            elif age > cfg.maximum_pending_age_frames:
                rejected = True
                rejection_frame = frame
                terminal_observed = int(group[group["frame_id"].astype(int) <= frame].shape[0])
                out.loc[idx] = False
        if first_frame is None or last_frame is None:
            continue
        terminal_status = "confirmed" if confirmed else "rejected" if rejected else "pending_end"
        if terminal_observed == 0:
            terminal_observed = int(len(group))
        records.append(
            GateEpisodeRecord(
                sequence_id=sequence_id,
                tracklet_id=tracklet_id,
                episode_id=episode_id,
                first_candidate_frame_id=int(first_frame),
                confirmation_frame_id=confirmation_frame,
                rejection_frame_id=rejection_frame,
                terminal_status=terminal_status,
                gate_delay_frames=(int(confirmation_frame) - int(first_frame)) if confirmation_frame is not None else None,
                num_observed_frames_before_terminal=int(terminal_observed),
                last_observed_frame_id=int(last_frame),
            )
        )
    metadata = pd.DataFrame([r.__dict__ for r in records])
    return GateResult(out, metadata, method_id, json.dumps(parameters or {}, sort_keys=True, separators=(",", ":")))


def terminalize_initiation(
    candidates: pd.DataFrame,
    trigger: pd.Series | np.ndarray,
    cfg: GateConfig | None = None,
) -> pd.Series:
    return gate_result_from_trigger(candidates, trigger, cfg).acceptance_mask


def confidence_initiation_gate(candidates: pd.DataFrame, threshold: float, cfg: GateConfig | None = None) -> pd.Series:
    return confidence_initiation_gate_result(candidates, threshold, cfg).acceptance_mask


def confidence_initiation_gate_result(candidates: pd.DataFrame, threshold: float, cfg: GateConfig | None = None) -> GateResult:
    trigger = candidates["confidence"].astype(float) >= float(threshold)
    return gate_result_from_trigger(candidates, trigger, cfg, method_id="confidence_initiation_gate", parameters={"threshold": float(threshold)})


def m_of_n_confirmation(candidates: pd.DataFrame, m: int, n: int, confidence_threshold: float, cfg: GateConfig | None = None) -> pd.Series:
    return m_of_n_confirmation_result(candidates, m, n, confidence_threshold, cfg).acceptance_mask


def m_of_n_confirmation_result(candidates: pd.DataFrame, m: int, n: int, confidence_threshold: float, cfg: GateConfig | None = None) -> GateResult:
    cfg = cfg or GateConfig()
    candidates = assign_episode_ids(candidates, cfg.max_track_gap) if "episode_id" not in candidates else candidates
    trigger = pd.Series(False, index=candidates.index)
    group_cols = episode_group_columns(candidates)
    for _, group in candidates.sort_values(group_cols + ["frame_id"]).groupby(group_cols, sort=False):
        hits_by_frame = {int(row.frame_id): float(row.confidence) >= float(confidence_threshold) for row in group.itertuples()}
        for idx, row in group.iterrows():
            frame = int(row.frame_id)
            count = sum(bool(hits_by_frame.get(f, False)) for f in range(frame - int(n) + 1, frame + 1))
            trigger.loc[idx] = count >= int(m)
    return gate_result_from_trigger(
        candidates,
        trigger,
        cfg,
        method_id="m_of_n_confirmation",
        parameters={"M": int(m), "N": int(n), "confidence_threshold": float(confidence_threshold)},
    )


def bayesian_terminal_gate(
    candidates: pd.DataFrame,
    high_update: float = 0.90,
    mid_update: float = 0.35,
    negative_update: float = -0.25,
    threshold: float = 1.10,
    cfg: GateConfig | None = None,
) -> pd.Series:
    return bayesian_terminal_gate_result(candidates, high_update, mid_update, negative_update, threshold, cfg).acceptance_mask


def bayesian_terminal_gate_result(
    candidates: pd.DataFrame,
    high_update: float = 0.90,
    mid_update: float = 0.35,
    negative_update: float = -0.25,
    threshold: float = 1.10,
    cfg: GateConfig | None = None,
) -> GateResult:
    cfg = cfg or GateConfig()
    candidates = assign_episode_ids(candidates, cfg.max_track_gap) if "episode_id" not in candidates else candidates
    trigger = pd.Series(False, index=candidates.index)
    group_cols = episode_group_columns(candidates)
    for _, group in candidates.sort_values(group_cols + ["frame_id"]).groupby(group_cols, sort=False):
        score = 0.0
        last_frame = None
        for idx, row in group.sort_values("frame_id").iterrows():
            frame = int(row.frame_id)
            if last_frame is not None and frame - last_frame > 1:
                score += negative_update * (frame - last_frame - 1)
            conf = float(row.confidence)
            score += high_update if conf >= 0.50 else mid_update if conf >= 0.30 else negative_update
            trigger.loc[idx] = score >= threshold
            last_frame = frame
    return gate_result_from_trigger(
        candidates,
        trigger,
        cfg,
        method_id="bayesian_fixed_terminal",
        parameters={"high_update": float(high_update), "mid_update": float(mid_update), "negative_update": float(negative_update), "threshold": float(threshold)},
    )


def rf_terminal_gate(candidates: pd.DataFrame, fp_probability: np.ndarray, threshold: float, cfg: GateConfig | None = None) -> pd.Series:
    return rf_terminal_gate_result(candidates, fp_probability, threshold, cfg).acceptance_mask


def rf_terminal_gate_result(candidates: pd.DataFrame, fp_probability: np.ndarray, threshold: float, cfg: GateConfig | None = None) -> GateResult:
    trigger = pd.Series(np.asarray(fp_probability, dtype=float) < float(threshold), index=candidates.index)
    return gate_result_from_trigger(candidates, trigger, cfg, method_id="rf_terminal_gate", parameters={"threshold": float(threshold)})


def tracker_baseline_gate_result(candidates: pd.DataFrame, cfg: GateConfig | None = None) -> GateResult:
    trigger = pd.Series(True, index=candidates.index)
    return gate_result_from_trigger(candidates, trigger, cfg, method_id="tracker_baseline", parameters={})


def legacy_trust_gate_result(candidates: pd.DataFrame, trigger: pd.Series | np.ndarray, cfg: GateConfig | None = None) -> GateResult:
    return gate_result_from_trigger(candidates, trigger, cfg, method_id="legacy_trust_terminalized", parameters={"source": "geometry_dynamic_no_multiagent"})


def assert_unique_episode_observations(candidates: pd.DataFrame) -> None:
    key = ["sequence_id", "tracklet_id", "episode_id", "frame_id"]
    if all(c in candidates.columns for c in key) and candidates.duplicated(key).any():
        bad = candidates.loc[candidates.duplicated(key, keep=False), key].head(5)
        raise ValueError(f"Duplicate episode observation key: {bad.to_dict(orient='records')}")
