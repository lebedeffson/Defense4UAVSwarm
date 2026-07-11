from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any

import pandas as pd

from defense4uavswarm.q1_v5.initiation_gate import GateEpisodeRecord, GateResult, assert_unique_episode_observations
from defense4uavswarm.q1_v5.token_bucket import FrameTokenBucket


METHOD_ID = "risk_prioritized_two_stage_quarantine"


@dataclass(frozen=True)
class RiskPrioritizedTwoStageConfig:
    stage1_fraction_max: float = 0.05
    stage1_budget_capacity: float = 2.0
    stage1_probe_frames: int = 1
    stage2_fraction_max: float = 0.03
    stage2_budget_capacity: float = 2.0
    stage2_horizon_frames: int = 5
    alpha_stage1_true: float = 0.10
    alpha_stage2_true: float = 0.05
    confirmation_support: float = 0.70
    evidence_decay: float = 0.90
    missing_frame_penalty: float = 0.15
    critical_confidence_floor: float = 0.02
    buffer_backfill: bool = True
    diagnostic_logging: bool = False
    tracker_profile: str = "bytetrack"

    @classmethod
    def from_mapping(cls, data: dict[str, Any] | None) -> "RiskPrioritizedTwoStageConfig":
        if not data:
            return cls()
        allowed = set(cls.__dataclass_fields__)  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in data.items() if k in allowed})

    def to_parameters(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}  # type: ignore[attr-defined]


def risk_prioritized_two_stage_gate_result(
    candidates: pd.DataFrame,
    frame_manifest: pd.DataFrame,
    episode_scores: pd.DataFrame,
    *,
    tau_stage1: float,
    tau_stage2: float,
    cfg: RiskPrioritizedTwoStageConfig | dict[str, Any] | None = None,
) -> tuple[GateResult, pd.DataFrame, pd.DataFrame]:
    qcfg = cfg if isinstance(cfg, RiskPrioritizedTwoStageConfig) else RiskPrioritizedTwoStageConfig.from_mapping(cfg)
    if "episode_id" not in candidates:
        raise ValueError("episode_id must be assigned before risk-prioritized quarantine")
    assert_unique_episode_observations(candidates)
    scores = _score_lookup(episode_scores)
    accepted = pd.Series(True, index=candidates.index)
    online = pd.Series(True, index=candidates.index)
    records: list[dict[str, Any]] = []
    ledger_rows: list[dict[str, Any]] = []
    admission_rows: list[dict[str, Any]] = []
    data = candidates.copy().sort_values(["sequence_id", "frame_id", "tracklet_id", "episode_id"])

    for seq, manifest_seq in frame_manifest.sort_values(["sequence_id", "frame_id"]).groupby("sequence_id", sort=False):
        seq = str(seq)
        seq_data = data[data["sequence_id"].astype(str).eq(seq)].copy()
        first_rows = seq_data.sort_values(["tracklet_id", "episode_id", "frame_id"]).groupby(["tracklet_id", "episode_id"], as_index=False, sort=False).first()
        by_episode = {_episode_key(k[0], k[1], k[2]): g.sort_values("frame_id") for k, g in seq_data.groupby(["sequence_id", "tracklet_id", "episode_id"], sort=False)}
        starts_by_frame = {int(f): fr.copy() for f, fr in first_rows.groupby("frame_id", sort=True)}
        stage2_by_frame: dict[int, list[tuple[str, pd.DataFrame]]] = {}
        bucket1 = FrameTokenBucket(fraction=qcfg.stage1_fraction_max, capacity=qcfg.stage1_budget_capacity, stage="stage1")
        bucket2 = FrameTokenBucket(fraction=qcfg.stage2_fraction_max, capacity=qcfg.stage2_budget_capacity, stage="stage2")
        stage1_admitted: set[str] = set()
        stage2_admitted: set[str] = set()

        for frame_id in sorted(int(x) for x in manifest_seq["frame_id"].unique()):
            starts = starts_by_frame.get(int(frame_id), pd.DataFrame(columns=first_rows.columns))
            bucket1.begin_frame(eligibility_event_count=len(starts))
            stage1_candidates = []
            for row in starts.itertuples(index=False):
                key = _episode_key(row.sequence_id, row.tracklet_id, row.episode_id)
                sc = scores.get(key, {})
                risk = float(sc.get("stage1_risk_score", 0.0))
                if risk >= float(tau_stage1):
                    stage1_candidates.append((risk, str(row.tracklet_id), int(row.episode_id), key, row))
            stage1_candidates.sort(key=lambda x: (-x[0], x[1], x[2]))
            stage1_admit_count = 0
            for rank, (risk, _tid, _eid, key, row) in enumerate(stage1_candidates, 1):
                admitted = bucket1.try_admit()
                admission_rows.append(_admission_row(seq, int(frame_id), "stage1", key, risk, tau_stage1, rank, admitted, "risk_above_threshold" if admitted else "no_token"))
                if not admitted:
                    continue
                stage1_admit_count += 1
                stage1_admitted.add(key)
                episode = by_episode[key]
                first_idx = episode.index[0]
                accepted.loc[first_idx] = False
                online.loc[first_idx] = False
                second = episode[episode["frame_id"].astype(int) > int(frame_id)]
                if second.empty:
                    records.append(_record(episode, "rejected", "REJECTED_BY_TEMPORAL_ABSENCE", confirmation=None, rejection=int(frame_id) + 1, stage1=True, stage2=False))
                else:
                    second_frame = int(second.iloc[0]["frame_id"])
                    stage2_by_frame.setdefault(second_frame, []).append((key, episode))
            ledger_rows.append(bucket1.record(sequence_id=seq, frame_id=int(frame_id), eligibility_event_count=len(starts), eligible_above_threshold=len(stage1_candidates), admitted_count=stage1_admit_count).__dict__)

            stage2_events = stage2_by_frame.pop(int(frame_id), [])
            bucket2.begin_frame(eligibility_event_count=len(stage2_events))
            stage2_candidates = []
            for key, episode in stage2_events:
                sc = scores.get(key, {})
                risk = float(sc.get("stage2_risk_score", 0.0))
                if risk >= float(tau_stage2):
                    stage2_candidates.append((risk, key, episode))
            stage2_candidates.sort(key=lambda x: (-x[0], x[1]))
            stage2_admit_count = 0
            handled_stage2 = set()
            for rank, (risk, key, episode) in enumerate(stage2_candidates, 1):
                admitted = bucket2.try_admit()
                admission_rows.append(_admission_row(seq, int(frame_id), "stage2", key, risk, tau_stage2, rank, admitted, "risk_above_threshold" if admitted else "no_token"))
                handled_stage2.add(key)
                if not admitted:
                    _backfill_release_after_stage1(episode, accepted)
                    records.append(_release_after_stage1(episode, int(frame_id)))
                    continue
                stage2_admit_count += 1
                stage2_admitted.add(key)
                first_frame = int(episode.iloc[0]["frame_id"])
                horizon_end = int(frame_id) + int(qcfg.stage2_horizon_frames)
                if (episode["confidence"].astype(float) < float(qcfg.critical_confidence_floor)).any():
                    records.append(_record(episode, "rejected", "REJECTED_BY_HARD_VETO", confirmation=None, rejection=int(frame_id), stage1=True, stage2=True))
                    continue
                future = episode[episode["frame_id"].astype(int) <= horizon_end]
                for idx in future.index:
                    accepted.loc[idx] = False
                    online.loc[idx] = False
                continuing = episode[episode["frame_id"].astype(int) > horizon_end]
                if continuing.empty:
                    # Bounded mute only: if no future rows exist before horizon, classify as temporal absence.
                    records.append(_record(episode, "rejected", "REJECTED_BY_TEMPORAL_ABSENCE", confirmation=None, rejection=horizon_end, stage1=True, stage2=True))
                else:
                    # Release/backfill after bounded mute.
                    if qcfg.buffer_backfill:
                        for idx in future.index:
                            accepted.loc[idx] = True
                    for idx in continuing.index:
                        accepted.loc[idx] = True
                        online.loc[idx] = True
                    records.append(_record(episode, "confirmed", "RELEASED_AFTER_STAGE2_HORIZON", confirmation=horizon_end, rejection=None, stage1=True, stage2=True))
            for key, episode in stage2_events:
                if key in handled_stage2:
                    continue
                _backfill_release_after_stage1(episode, accepted)
                records.append(_release_after_stage1(episode, int(frame_id)))
            ledger_rows.append(bucket2.record(sequence_id=seq, frame_id=int(frame_id), eligibility_event_count=len(stage2_events), eligible_above_threshold=len(stage2_candidates), admitted_count=stage2_admit_count).__dict__)

        recorded = {_episode_key(r["sequence_id"], r["tracklet_id"], r["episode_id"]) for r in records}
        final_frame_id = int(manifest_seq["frame_id"].max())
        for key, episode in by_episode.items():
            if key in recorded:
                continue
            if key in stage1_admitted:
                records.append(_record(episode, "confirmed", "CENSORED_AT_SEQUENCE_END", confirmation=final_frame_id, rejection=None, stage1=True, stage2=key in stage2_admitted, right_censored=True))
            else:
                records.append(_record(episode, "confirmed", "FAST_PASSED", confirmation=int(episode.iloc[0]["frame_id"]), rejection=None, stage1=False, stage2=False))

    metadata = pd.DataFrame(records)
    params = qcfg.to_parameters()
    params.update({"tau_stage1": float(tau_stage1), "tau_stage2": float(tau_stage2), "controller": "risk_prioritized_two_stage"})
    return (
        GateResult(accepted.astype(bool), metadata, METHOD_ID, json.dumps(params, sort_keys=True, separators=(",", ":")), online.astype(bool)),
        pd.DataFrame(ledger_rows),
        pd.DataFrame(admission_rows),
    )


def _score_lookup(scores: pd.DataFrame) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    if scores.empty:
        return out
    for row in scores.itertuples(index=False):
        key = _episode_key(row.sequence_id, row.tracklet_id, row.episode_id)
        out.setdefault(key, {}).update(row._asdict())
    return out


def _episode_key(seq: Any, tid: Any, episode_id: Any) -> str:
    return f"{seq}::{tid}::{int(episode_id)}"


def _release_after_stage1(episode: pd.DataFrame, frame_id: int) -> dict[str, Any]:
    return _record(episode, "confirmed", "RELEASED_AFTER_STAGE1", confirmation=int(frame_id), rejection=None, stage1=True, stage2=False)


def _backfill_release_after_stage1(episode: pd.DataFrame, accepted: pd.Series) -> None:
    for idx in episode.index:
        accepted.loc[idx] = True


def _record(episode: pd.DataFrame, terminal: str, selective: str, *, confirmation: int | None, rejection: int | None, stage1: bool, stage2: bool, right_censored: bool = False) -> dict[str, Any]:
    first = episode.iloc[0]
    return {
        **GateEpisodeRecord(
            sequence_id=str(first.sequence_id),
            tracklet_id=str(first.tracklet_id),
            episode_id=int(first.episode_id),
            first_candidate_frame_id=int(first.frame_id),
            confirmation_frame_id=confirmation,
            rejection_frame_id=rejection,
            terminal_status=terminal,
            gate_delay_frames=(int(confirmation) - int(first.frame_id)) if confirmation is not None else None,
            num_observed_frames_before_terminal=int(len(episode[episode["frame_id"].astype(int) <= (confirmation if confirmation is not None else rejection if rejection is not None else int(episode["frame_id"].max()))])),
            last_observed_frame_id=int(episode["frame_id"].max()),
        ).__dict__,
        "selective_terminal_status": selective,
        "terminal_reason": selective.lower(),
        "stage1_admitted": bool(stage1),
        "stage2_admitted": bool(stage2),
        "was_quarantined": bool(stage1),
        "configured_quarantine_fraction": 0.0,
        "realized_quarantine_fraction": 0.0,
        "right_censored": bool(right_censored),
        "pending_end_count": 0,
    }


def _admission_row(seq: str, frame_id: int, stage: str, key: str, risk: float, threshold: float, rank: int, admitted: bool, reason: str) -> dict[str, Any]:
    _, tid, eid = key.split("::", 2)
    return {
        "sequence_id": seq,
        "tracklet_id": tid,
        "episode_id": int(eid),
        "frame_id": int(frame_id),
        "stage": stage,
        "risk_score": float(risk),
        "risk_threshold": float(threshold),
        "rank_in_frame": int(rank),
        "admitted": bool(admitted),
        "budget_available": bool(admitted),
        "reason": reason,
        "model_sha256": "",
    }
