from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import json
import math
from typing import Any, Literal

import numpy as np
import pandas as pd

from defense4uavswarm.q1_v5.confidence_normalizer import (
    RollingPercentileConfidence,
    normalize_absolute_confidence,
    size_bin_from_area_norm,
)
from defense4uavswarm.q1_v5.initiation_gate import GateEpisodeRecord, GateResult, assert_unique_episode_observations


@dataclass(frozen=True)
class SelectiveTrustQuarantineConfig:
    episode_start_history_only: bool = True
    hard_budget_enabled: bool = True
    budget_capacity: float = 2.0
    timeout_policy: Literal["release_to_baseline", "reject"] = "release_to_baseline"
    reject_requires_hard_veto: bool = True
    critical_confidence_floor: float = 0.02
    robust_inner_max_f1_loss: float = 0.01
    robust_inner_mean_f1_loss: float = 0.005
    confidence_floor: float = 0.05
    relative_weight: float = 0.60
    relative_confidence_window: int = 500
    relative_confidence_min_history: int = 30
    support_history_window: int = 500
    support_history_min: int = 30
    survival_window_episodes: int = 200
    survival_low: float = 0.25
    survival_high: float = 0.65
    quarantine_fraction_min: float = 0.05
    quarantine_fraction_max: float = 0.35
    fast_pass_support: float = 0.85
    evidence_decay: float = 0.90
    missing_frame_penalty: float = 0.15
    confirmation_support: float = 0.75
    maximum_quarantine_frames: int = 3
    buffer_backfill: bool = True
    epsilon: float = 1e-6
    tracker_profile: str = "bytetrack"

    @classmethod
    def from_mapping(cls, data: dict[str, Any] | None) -> "SelectiveTrustQuarantineConfig":
        if not data:
            return cls()
        allowed = set(cls.__dataclass_fields__)  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in data.items() if k in allowed})

    def to_parameters(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}  # type: ignore[attr-defined]


@dataclass
class QuarantineState:
    first_frame_id: int
    last_frame_id: int
    accumulated_hazard: float = 0.0
    buffered_indices: list[Any] = field(default_factory=list)
    observed_hits: int = 0
    missing_frames: int = 0
    terminal_status: str = "quarantined"
    confirmation_frame_id: int | None = None
    rejection_frame_id: int | None = None
    support_history: list[float] = field(default_factory=list)
    first_reason: str = "quarantine"
    hard_veto_seen: bool = False
    configured_quarantine_fraction: float = 0.0
    realized_quarantine_fraction: float = 0.0
    budget_tokens_before: float = 0.0
    budget_tokens_after: float = 0.0
    episode_start_rank: float | None = None
    timeout_release_count: int = 0
    hard_veto_rejection_count: int = 0
    terminal_reason: str = ""
    was_quarantined: bool = False


@dataclass(frozen=True)
class SelectiveTrustDecision:
    status: str
    publish_current: bool
    publish_buffer: bool
    support: float
    accumulated_support: float
    quarantine_threshold: float
    quarantine_fraction: float
    reason: str
    budget_tokens_before: float = 0.0
    budget_tokens_after: float = 0.0
    episode_start_rank: float | None = None
    realized_quarantine_fraction: float = 0.0
    hard_veto: bool = False


class SequenceAdaptiveStats:
    def __init__(self, cfg: SelectiveTrustQuarantineConfig):
        self.cfg = cfg
        self.confidence = RollingPercentileConfidence(
            window=int(cfg.relative_confidence_window),
            min_history=int(cfg.relative_confidence_min_history),
        )
        self.support_history: deque[float] = deque(maxlen=int(cfg.support_history_window))
        self.survival_flags: deque[int] = deque(maxlen=int(cfg.survival_window_episodes))
        self.open_episode_starts: dict[tuple[str, int], int] = {}
        self.budget_tokens: float = 0.0
        self.start_decisions: deque[int] = deque(maxlen=int(cfg.survival_window_episodes))

    def close_expired_starts(self, frame_id: int) -> None:
        expired = [key for key, start in self.open_episode_starts.items() if int(frame_id) - int(start) > 2]
        for key in expired:
            self.survival_flags.append(0)
            del self.open_episode_starts[key]

    def note_episode_start(self, tracklet_id: str, episode_id: int, frame_id: int) -> None:
        self.open_episode_starts[(str(tracklet_id), int(episode_id))] = int(frame_id)

    def note_second_observation(self, tracklet_id: str, episode_id: int, frame_id: int) -> None:
        key = (str(tracklet_id), int(episode_id))
        start = self.open_episode_starts.pop(key, None)
        if start is not None:
            self.survival_flags.append(1 if int(frame_id) - int(start) <= 2 else 0)

    def survival_rate(self) -> float:
        if not self.survival_flags:
            return float(self.cfg.survival_low)
        return float(sum(self.survival_flags)) / float(len(self.survival_flags))

    def quarantine_fraction(self) -> float:
        cfg = self.cfg
        if cfg.survival_high <= cfg.survival_low:
            factor = 0.0
        else:
            factor = (self.survival_rate() - cfg.survival_low) / (cfg.survival_high - cfg.survival_low)
        factor = max(0.0, min(1.0, factor))
        return float(cfg.quarantine_fraction_min + (cfg.quarantine_fraction_max - cfg.quarantine_fraction_min) * factor)

    def quarantine_threshold(self) -> float:
        if not self.support_history:
            return 0.0
        if len(self.support_history) < int(self.cfg.support_history_min):
            return 0.0
        q = self.quarantine_fraction()
        return float(np.quantile(np.asarray(self.support_history, dtype=float), q))

    def score_support_batch(self, frame: pd.DataFrame) -> tuple[list[float], list[float | None], list[tuple[str, str]]]:
        keys: list[tuple[str, str]] = []
        confidences: list[float] = []
        for row in frame.itertuples():
            area_norm = _area_norm(row)
            keys.append((size_bin_from_area_norm(area_norm), str(getattr(row, "class_name", "unknown"))))
            confidences.append(float(row.confidence))
        rel_scores = self.confidence.score_batch(confidences, keys)
        supports: list[float] = []
        for c, rel in zip(confidences, rel_scores):
            abs_score = normalize_absolute_confidence(c, self.cfg.confidence_floor)
            if c < self.cfg.confidence_floor:
                support = 0.0
            elif rel is None:
                support = abs_score
            else:
                support = (1.0 - self.cfg.relative_weight) * abs_score + self.cfg.relative_weight * float(rel)
            supports.append(max(0.0, min(1.0, float(support))))
        return supports, rel_scores, keys

    def episode_start_rank(self, support: float) -> float | None:
        if len(self.support_history) < int(self.cfg.support_history_min):
            return None
        values = list(self.support_history)
        return sum(1 for x in values if x <= float(support)) / float(len(values))

    def budget_decision(self, quarantine_fraction: float, risk_candidate: bool) -> tuple[bool, float, float, float]:
        before = float(self.budget_tokens)
        if not self.cfg.hard_budget_enabled:
            should_quarantine = bool(risk_candidate)
            self.start_decisions.append(1 if should_quarantine else 0)
            return should_quarantine, before, before, self.realized_quarantine_fraction()
        self.budget_tokens = min(float(self.cfg.budget_capacity), self.budget_tokens + float(quarantine_fraction))
        before = float(self.budget_tokens)
        allowed_by_window = self.realized_quarantine_fraction() <= float(quarantine_fraction) + self._budget_tolerance()
        should_quarantine = bool(risk_candidate and allowed_by_window and self.budget_tokens >= 1.0)
        if should_quarantine:
            self.budget_tokens -= 1.0
        after = float(self.budget_tokens)
        self.start_decisions.append(1 if should_quarantine else 0)
        return should_quarantine, before, after, self.realized_quarantine_fraction()

    def realized_quarantine_fraction(self) -> float:
        if not self.start_decisions:
            return 0.0
        return float(sum(self.start_decisions)) / float(len(self.start_decisions))

    def _budget_tolerance(self) -> float:
        return 1.0 / float(max(1, len(self.start_decisions) + 1))

    def update_after_frame(
        self,
        confidences: list[float],
        keys: list[tuple[str, str]],
        supports: list[float],
        is_episode_start: list[bool],
    ) -> None:
        if self.cfg.episode_start_history_only:
            conf_keep = [c for c, keep in zip(confidences, is_episode_start) if keep]
            key_keep = [k for k, keep in zip(keys, is_episode_start) if keep]
            support_keep = [s for s, keep in zip(supports, is_episode_start) if keep]
        else:
            conf_keep = confidences
            key_keep = keys
            support_keep = supports
        self.confidence.update_batch(conf_keep, key_keep)
        for support in support_keep:
            self.support_history.append(float(support))


class SelectiveTrustQuarantine:
    def __init__(self, cfg: SelectiveTrustQuarantineConfig):
        self.cfg = cfg
        self.states: dict[tuple[str, str, int], QuarantineState] = {}

    def update(
        self,
        *,
        sequence_id: str,
        tracklet_id: str,
        episode_id: int,
        frame_id: int,
        support: float,
        quarantine_threshold: float,
        quarantine_fraction: float,
        should_quarantine_start: bool,
        hard_veto: bool,
        budget_tokens_before: float,
        budget_tokens_after: float,
        realized_quarantine_fraction: float,
        episode_start_rank: float | None,
        row_index: Any,
    ) -> SelectiveTrustDecision:
        key = (str(sequence_id), str(tracklet_id), int(episode_id))
        support = max(0.0, min(1.0, float(support)))
        state = self.states.get(key)
        if state is None:
            if support >= float(self.cfg.fast_pass_support):
                self.states[key] = QuarantineState(
                    first_frame_id=int(frame_id),
                    last_frame_id=int(frame_id),
                    observed_hits=1,
                    terminal_status="fast_passed",
                    confirmation_frame_id=int(frame_id),
                    support_history=[support],
                    first_reason="fast_pass_high_support",
                    configured_quarantine_fraction=float(quarantine_fraction),
                    realized_quarantine_fraction=float(realized_quarantine_fraction),
                    budget_tokens_before=float(budget_tokens_before),
                    budget_tokens_after=float(budget_tokens_after),
                    episode_start_rank=episode_start_rank,
                    hard_veto_seen=bool(hard_veto),
                    terminal_reason="fast_pass_high_support",
                )
                return SelectiveTrustDecision("FAST_PASSED", True, False, support, support, quarantine_threshold, quarantine_fraction, "fast_pass_high_support", budget_tokens_before, budget_tokens_after, episode_start_rank, realized_quarantine_fraction, hard_veto)
            if not should_quarantine_start:
                self.states[key] = QuarantineState(
                    first_frame_id=int(frame_id),
                    last_frame_id=int(frame_id),
                    observed_hits=1,
                    terminal_status="fast_passed",
                    confirmation_frame_id=int(frame_id),
                    support_history=[support],
                    first_reason="fast_pass_above_adaptive_threshold",
                    configured_quarantine_fraction=float(quarantine_fraction),
                    realized_quarantine_fraction=float(realized_quarantine_fraction),
                    budget_tokens_before=float(budget_tokens_before),
                    budget_tokens_after=float(budget_tokens_after),
                    episode_start_rank=episode_start_rank,
                    hard_veto_seen=bool(hard_veto),
                    terminal_reason="fast_pass_budget_or_threshold",
                )
                return SelectiveTrustDecision("FAST_PASSED", True, False, support, support, quarantine_threshold, quarantine_fraction, "fast_pass_budget_or_threshold", budget_tokens_before, budget_tokens_after, episode_start_rank, realized_quarantine_fraction, hard_veto)
            state = QuarantineState(
                first_frame_id=int(frame_id),
                last_frame_id=int(frame_id),
                observed_hits=1,
                buffered_indices=[row_index],
                support_history=[support],
                first_reason="quarantine_low_support",
                hard_veto_seen=bool(hard_veto),
                configured_quarantine_fraction=float(quarantine_fraction),
                realized_quarantine_fraction=float(realized_quarantine_fraction),
                budget_tokens_before=float(budget_tokens_before),
                budget_tokens_after=float(budget_tokens_after),
                episode_start_rank=episode_start_rank,
                terminal_reason="quarantine_low_support",
                was_quarantined=True,
            )
            state.accumulated_hazard = _support_to_hazard(support, self.cfg.epsilon)
            self.states[key] = state
            return SelectiveTrustDecision(
                "QUARANTINED",
                False,
                False,
                support,
                _hazard_to_support(state.accumulated_hazard),
                quarantine_threshold,
                quarantine_fraction,
                "quarantine_low_support",
                budget_tokens_before,
                budget_tokens_after,
                episode_start_rank,
                realized_quarantine_fraction,
                hard_veto,
            )

        if state.terminal_status in {"confirmed", "fast_passed"}:
            state.last_frame_id = int(frame_id)
            state.observed_hits += 1
            return SelectiveTrustDecision(state.terminal_status.upper(), True, False, support, 1.0, quarantine_threshold, quarantine_fraction, "already_confirmed")
        if state.terminal_status == "released_to_baseline":
            state.last_frame_id = int(frame_id)
            state.observed_hits += 1
            return SelectiveTrustDecision("RELEASED_TO_BASELINE", True, False, support, _hazard_to_support(state.accumulated_hazard), quarantine_threshold, quarantine_fraction, "already_released")
        if state.terminal_status == "rejected":
            return SelectiveTrustDecision("REJECTED", False, False, support, _hazard_to_support(state.accumulated_hazard), quarantine_threshold, quarantine_fraction, "already_rejected")

        dt = max(1, int(frame_id) - int(state.last_frame_id))
        state.missing_frames += max(0, dt - 1)
        state.last_frame_id = int(frame_id)
        state.observed_hits += 1
        state.buffered_indices.append(row_index)
        state.support_history.append(support)
        state.hard_veto_seen = bool(state.hard_veto_seen or hard_veto)
        state.accumulated_hazard = (
            (float(self.cfg.evidence_decay) ** dt) * state.accumulated_hazard
            + _support_to_hazard(support, self.cfg.epsilon)
            - float(self.cfg.missing_frame_penalty) * max(0, dt - 1)
        )
        state.accumulated_hazard = max(0.0, state.accumulated_hazard)
        accumulated_support = _hazard_to_support(state.accumulated_hazard)
        age = int(frame_id) - int(state.first_frame_id) + 1
        if support >= float(self.cfg.fast_pass_support) or accumulated_support >= float(self.cfg.confirmation_support):
            state.terminal_status = "confirmed"
            state.confirmation_frame_id = int(frame_id)
            state.terminal_reason = "hazard_confirmed"
            return SelectiveTrustDecision(
                "CONFIRMED",
                True,
                bool(self.cfg.buffer_backfill),
                support,
                accumulated_support,
                quarantine_threshold,
                quarantine_fraction,
                "hazard_confirmed",
            )
        if age > int(self.cfg.maximum_quarantine_frames):
            if self.cfg.timeout_policy == "release_to_baseline" or (self.cfg.reject_requires_hard_veto and not state.hard_veto_seen):
                state.terminal_status = "released_to_baseline"
                state.confirmation_frame_id = int(frame_id)
                state.timeout_release_count += 1
                state.terminal_reason = "timeout_release_to_baseline"
                return SelectiveTrustDecision("RELEASED_TO_BASELINE", True, bool(self.cfg.buffer_backfill), support, accumulated_support, quarantine_threshold, quarantine_fraction, "timeout_release_to_baseline")
            state.terminal_status = "rejected"
            state.rejection_frame_id = int(frame_id)
            state.hard_veto_rejection_count += 1 if state.hard_veto_seen else 0
            state.terminal_reason = "maximum_quarantine_reject_with_veto" if state.hard_veto_seen else "maximum_quarantine_reject"
            return SelectiveTrustDecision("REJECTED", False, False, support, accumulated_support, quarantine_threshold, quarantine_fraction, state.terminal_reason)
        return SelectiveTrustDecision("QUARANTINED", False, False, support, accumulated_support, quarantine_threshold, quarantine_fraction, "hazard_pending")


def selective_quarantine_gate_result(
    candidates: pd.DataFrame,
    cfg: SelectiveTrustQuarantineConfig | dict[str, Any] | None = None,
) -> GateResult:
    qcfg = cfg if isinstance(cfg, SelectiveTrustQuarantineConfig) else SelectiveTrustQuarantineConfig.from_mapping(cfg)
    if "episode_id" not in candidates:
        raise ValueError("selective quarantine requires episode_id assigned before running")
    assert_unique_episode_observations(candidates)
    data = _ensure_area_norm(candidates)
    accepted = pd.Series(False, index=data.index)
    online_accepted = pd.Series(False, index=data.index)
    layers: dict[str, SelectiveTrustQuarantine] = {}
    records: dict[tuple[str, str, int], GateEpisodeRecord] = {}
    stats_by_seq: dict[str, SequenceAdaptiveStats] = {}
    event_rows: list[dict[str, Any]] = []

    for seq, seq_group in data.sort_values(["sequence_id", "frame_id", "tracklet_id", "episode_id"]).groupby("sequence_id", sort=False):
        seq = str(seq)
        stats = stats_by_seq.setdefault(seq, SequenceAdaptiveStats(qcfg))
        layer = layers.setdefault(seq, SelectiveTrustQuarantine(qcfg))
        for frame_id, frame in seq_group.groupby("frame_id", sort=True):
            stats.close_expired_starts(int(frame_id))
            frame = frame.sort_values(["tracklet_id", "episode_id", "det_id" if "det_id" in frame else "frame_id"])
            supports, rel_scores, rel_keys = stats.score_support_batch(frame)
            threshold = stats.quarantine_threshold()
            fraction = stats.quarantine_fraction()
            confidences = [float(row.confidence) for row in frame.itertuples()]
            start_flags: list[bool] = []
            for support, rel, row in zip(supports, rel_scores, frame.itertuples()):
                key = (str(row.sequence_id), str(row.tracklet_id), int(row.episode_id))
                state_before = layer.states.get(key)
                is_start = state_before is None
                start_flags.append(is_start)
                hard_veto = float(row.confidence) < float(qcfg.critical_confidence_floor)
                rank = stats.episode_start_rank(support) if is_start else None
                risk_candidate = bool(is_start and support < threshold)
                if is_start:
                    should_quarantine, budget_before, budget_after, realized_fraction = stats.budget_decision(fraction, risk_candidate)
                else:
                    should_quarantine, budget_before, budget_after, realized_fraction = False, stats.budget_tokens, stats.budget_tokens, stats.realized_quarantine_fraction()
                if state_before is None:
                    stats.note_episode_start(str(row.tracklet_id), int(row.episode_id), int(row.frame_id))
                elif state_before.observed_hits == 1:
                    stats.note_second_observation(str(row.tracklet_id), int(row.episode_id), int(row.frame_id))
                decision = layer.update(
                    sequence_id=str(row.sequence_id),
                    tracklet_id=str(row.tracklet_id),
                    episode_id=int(row.episode_id),
                    frame_id=int(row.frame_id),
                    support=support,
                    quarantine_threshold=threshold,
                    quarantine_fraction=fraction,
                    should_quarantine_start=should_quarantine,
                    hard_veto=hard_veto,
                    budget_tokens_before=budget_before,
                    budget_tokens_after=budget_after,
                    realized_quarantine_fraction=realized_fraction,
                    episode_start_rank=rank,
                    row_index=row.Index,
                )
                if decision.publish_current:
                    accepted.loc[row.Index] = True
                    online_accepted.loc[row.Index] = True
                state = layer.states[key]
                if decision.publish_buffer:
                    for idx in state.buffered_indices:
                        accepted.loc[idx] = True
                event_rows.append(
                    {
                        "row_index": row.Index,
                        "sequence_id": str(row.sequence_id),
                        "tracklet_id": str(row.tracklet_id),
                        "episode_id": int(row.episode_id),
                        "frame_id": int(row.frame_id),
                        "support": support,
                        "relative_confidence": rel,
                        "quarantine_threshold": threshold,
                        "quarantine_fraction": fraction,
                        "configured_quarantine_fraction": fraction,
                        "realized_quarantine_fraction": realized_fraction,
                        "budget_tokens_before": budget_before,
                        "budget_tokens_after": budget_after,
                        "episode_start_rank": rank,
                        "hard_veto": hard_veto,
                        "accumulated_support": decision.accumulated_support,
                        "decision_status": decision.status,
                        "decision_reason": decision.reason,
                    }
                )
            stats.update_after_frame(confidences, rel_keys, supports, start_flags)

    for seq, layer in layers.items():
        for (state_seq, tid, episode_id), state in layer.states.items():
            if state_seq != seq:
                continue
            terminal = state.terminal_status
            confirmation = state.confirmation_frame_id
            rejection = state.rejection_frame_id
            if terminal == "quarantined":
                terminal = "pending_end"
            status_for_evaluator = "confirmed" if terminal in {"confirmed", "fast_passed", "released_to_baseline"} else "rejected" if terminal == "rejected" else terminal
            records[(state_seq, tid, episode_id)] = {
                **GateEpisodeRecord(
                    sequence_id=state_seq,
                    tracklet_id=tid,
                    episode_id=int(episode_id),
                    first_candidate_frame_id=int(state.first_frame_id),
                    confirmation_frame_id=confirmation,
                    rejection_frame_id=rejection,
                    terminal_status=status_for_evaluator,
                    gate_delay_frames=(int(confirmation) - int(state.first_frame_id)) if confirmation is not None else None,
                    num_observed_frames_before_terminal=int(state.observed_hits),
                    last_observed_frame_id=int(state.last_frame_id),
                ).__dict__,
                "selective_terminal_status": terminal,
                "terminal_reason": state.terminal_reason,
                "configured_quarantine_fraction": state.configured_quarantine_fraction,
                "realized_quarantine_fraction": state.realized_quarantine_fraction,
                "budget_tokens_before": state.budget_tokens_before,
                "budget_tokens_after": state.budget_tokens_after,
                "episode_start_rank": state.episode_start_rank,
                "timeout_release_count": state.timeout_release_count,
                "hard_veto_rejection_count": state.hard_veto_rejection_count,
                "hard_veto_seen": state.hard_veto_seen,
                "was_quarantined": state.was_quarantined,
            }
    metadata = pd.DataFrame(list(records.values()))
    parameters = qcfg.to_parameters()
    parameters["evidence_model"] = "hazard_noisy_or"
    parameters["buffer_backfill"] = bool(qcfg.buffer_backfill)
    if event_rows:
        parameters["event_columns"] = [
            "support",
            "relative_confidence",
            "quarantine_threshold",
            "quarantine_fraction",
            "accumulated_support",
            "decision_status",
            "decision_reason",
            "budget_tokens_before",
            "budget_tokens_after",
            "episode_start_rank",
            "realized_quarantine_fraction",
            "hard_veto",
        ]
    return GateResult(
        accepted.astype(bool),
        metadata,
        "selective_trust_quarantine",
        json.dumps(parameters, sort_keys=True, separators=(",", ":")),
        online_accepted.astype(bool),
    )


def _support_to_hazard(support: float, epsilon: float) -> float:
    return max(0.0, -math.log(max(float(epsilon), 1.0 - max(0.0, min(1.0, float(support))))))


def _hazard_to_support(hazard: float) -> float:
    return max(0.0, min(1.0, 1.0 - math.exp(-max(0.0, float(hazard)))))


def _area_norm(row: Any) -> float:
    if hasattr(row, "bbox_area_norm"):
        return float(row.bbox_area_norm)
    if hasattr(row, "bbox_area") and hasattr(row, "image_width") and hasattr(row, "image_height"):
        denom = float(row.image_width) * float(row.image_height)
        return float(row.bbox_area) / max(1.0, denom)
    if all(hasattr(row, name) for name in ["x1", "y1", "x2", "y2", "image_width", "image_height"]):
        area = max(1.0, float(row.x2) - float(row.x1)) * max(1.0, float(row.y2) - float(row.y1))
        return area / max(1.0, float(row.image_width) * float(row.image_height))
    raise ValueError("selective quarantine requires bbox_area_norm or explicit image dimensions")


def _ensure_area_norm(det: pd.DataFrame) -> pd.DataFrame:
    out = det.copy()
    if "bbox_area_norm" in out:
        return out
    if "bbox_area" in out and {"image_width", "image_height"}.issubset(out.columns):
        out["bbox_area_norm"] = out["bbox_area"].astype(float) / (out["image_width"].astype(float) * out["image_height"].astype(float)).clip(lower=1)
        return out
    if {"x1", "y1", "x2", "y2", "image_width", "image_height"}.issubset(out.columns):
        out["bbox_area_norm"] = ((out["x2"].astype(float) - out["x1"].astype(float)).clip(lower=1) * (out["y2"].astype(float) - out["y1"].astype(float)).clip(lower=1)) / (
            out["image_width"].astype(float) * out["image_height"].astype(float)
        ).clip(lower=1)
        return out
    raise ValueError("selective quarantine requires bbox_area_norm or explicit image dimensions")
