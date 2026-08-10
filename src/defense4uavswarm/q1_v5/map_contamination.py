from __future__ import annotations

import numpy as np
import pandas as pd


def track_level_table(det: pd.DataFrame, accepted: pd.Series | np.ndarray) -> pd.DataFrame:
    acc = det[pd.Series(accepted, index=det.index).astype(bool)].copy()
    if acc.empty:
        return pd.DataFrame(columns=["tracklet_id", "is_false_track", "init_frame", "last_active_frame", "lifetime_frames", "active_frame_count"])
    acc["eval_is_tp_bool"] = acc["eval_is_tp"].astype(bool)
    grouped = acc.groupby("tracklet_id", sort=False)
    tracks = grouped.agg(
        sequence_id=("sequence_id", "first"),
        init_frame=("frame_id", "min"),
        last_active_frame=("frame_id", "max"),
        active_frame_count=("frame_id", "count"),
        has_tp=("eval_is_tp_bool", "max"),
        max_confidence=("confidence", "max"),
        mean_confidence=("confidence", "mean"),
    ).reset_index()
    tracks["is_false_track"] = ~tracks["has_tp"].astype(bool)
    tracks["lifetime_frames"] = tracks["last_active_frame"].astype(int) - tracks["init_frame"].astype(int) + 1
    tracks["termination_reason"] = "end_or_filtered"
    return tracks.drop(columns=["has_tp"])


def contamination_metrics(det: pd.DataFrame, accepted: pd.Series | np.ndarray, num_agents: int = 1) -> dict[str, float]:
    tracks = track_level_table(det, accepted)
    if tracks.empty:
        return _empty()
    false_tracks = tracks[tracks["is_false_track"].astype(bool)]
    acc = det[pd.Series(accepted, index=det.index).astype(bool)].copy()
    false_ids = set(false_tracks["tracklet_id"].astype(str))
    false_rows = acc[acc["tracklet_id"].astype(str).isin(false_ids)]
    concurrency = false_rows.groupby(["sequence_id", "frame_id"])["tracklet_id"].nunique() if not false_rows.empty else pd.Series(dtype=float)
    total_occupancy = int(acc.groupby("tracklet_id").size().sum()) if not acc.empty else 0
    false_occupancy = int(false_tracks["active_frame_count"].sum()) if not false_tracks.empty else 0
    return {
        "false_new_tracks": int(len(false_tracks)),
        "false_track_occupancy_frames": false_occupancy,
        "mean_false_track_lifetime": float(false_tracks["lifetime_frames"].mean()) if len(false_tracks) else 0.0,
        "median_false_track_lifetime": float(false_tracks["lifetime_frames"].median()) if len(false_tracks) else 0.0,
        "p90_false_track_lifetime": float(false_tracks["lifetime_frames"].quantile(0.90)) if len(false_tracks) else 0.0,
        "p95_false_track_lifetime": float(false_tracks["lifetime_frames"].quantile(0.95)) if len(false_tracks) else 0.0,
        "peak_concurrent_false_tracks": int(concurrency.max()) if len(concurrency) else 0,
        "mean_concurrent_false_tracks": float(concurrency.mean()) if len(concurrency) else 0.0,
        "p95_concurrent_false_tracks": float(concurrency.quantile(0.95)) if len(concurrency) else 0.0,
        "false_track_share": false_occupancy / max(1, total_occupancy),
        "proxy_redundant_track_broadcasts": false_occupancy * max(0, int(num_agents) - 1),
    }


def _empty() -> dict[str, float]:
    return {
        "false_new_tracks": 0,
        "false_track_occupancy_frames": 0,
        "mean_false_track_lifetime": 0.0,
        "median_false_track_lifetime": 0.0,
        "p90_false_track_lifetime": 0.0,
        "p95_false_track_lifetime": 0.0,
        "peak_concurrent_false_tracks": 0,
        "mean_concurrent_false_tracks": 0.0,
        "p95_concurrent_false_tracks": 0.0,
        "false_track_share": 0.0,
        "proxy_redundant_track_broadcasts": 0,
    }
