from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class GateConfig:
    maximum_pending_age_frames: int = 6
    max_track_gap: int = 1


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


def episode_group_columns(candidates: pd.DataFrame) -> list[str]:
    return ["sequence_id", "tracklet_id"] + (["episode_id"] if "episode_id" in candidates else [])


def terminalize_initiation(
    candidates: pd.DataFrame,
    trigger: pd.Series | np.ndarray,
    cfg: GateConfig | None = None,
) -> pd.Series:
    cfg = cfg or GateConfig()
    candidates = assign_episode_ids(candidates, cfg.max_track_gap) if "episode_id" not in candidates else candidates
    trigger_series = pd.Series(trigger, index=candidates.index).astype(bool)
    out = pd.Series(False, index=candidates.index)
    group_cols = episode_group_columns(candidates)
    for _, group in candidates.sort_values(group_cols + ["frame_id"]).groupby(group_cols, sort=False):
        confirmed = False
        rejected = False
        first_frame = None
        last_frame = None
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
                out.loc[idx] = True
            elif age > cfg.maximum_pending_age_frames:
                rejected = True
                out.loc[idx] = False
    return out


def confidence_initiation_gate(candidates: pd.DataFrame, threshold: float, cfg: GateConfig | None = None) -> pd.Series:
    return terminalize_initiation(candidates, candidates["confidence"].astype(float) >= float(threshold), cfg)


def m_of_n_confirmation(candidates: pd.DataFrame, m: int, n: int, confidence_threshold: float, cfg: GateConfig | None = None) -> pd.Series:
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
    return terminalize_initiation(candidates, trigger, cfg)


def bayesian_terminal_gate(
    candidates: pd.DataFrame,
    high_update: float = 0.90,
    mid_update: float = 0.35,
    negative_update: float = -0.25,
    threshold: float = 1.10,
    cfg: GateConfig | None = None,
) -> pd.Series:
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
    return terminalize_initiation(candidates, trigger, cfg)


def rf_terminal_gate(candidates: pd.DataFrame, fp_probability: np.ndarray, threshold: float, cfg: GateConfig | None = None) -> pd.Series:
    trigger = pd.Series(np.asarray(fp_probability, dtype=float) < float(threshold), index=candidates.index)
    return terminalize_initiation(candidates, trigger, cfg)
