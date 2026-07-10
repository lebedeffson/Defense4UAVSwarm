from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Literal

import pandas as pd

from defense4uavswarm.q1_v5.candidate_state import CandidateState, CandidateStatus, status_value
from defense4uavswarm.q1_v5.confidence_normalizer import (
    RollingPercentileConfidence,
    normalize_absolute_confidence,
    size_bin_from_area_norm,
)
from defense4uavswarm.q1_v5.explanation import DecisionExplanation
from defense4uavswarm.q1_v5.kinematic_consistency import KinematicTrackState


@dataclass(frozen=True)
class TrustGuardConfig:
    mode: Literal["operational", "strict"] = "operational"
    absolute_confidence_floor: float = 0.07
    relative_confidence_weight: float = 0.30
    relative_confidence_window: int = 500
    relative_confidence_min_history: int = 30
    evidence_decay: float = 0.60
    confirmation_threshold: float = 0.55
    minimum_valid_hits: int = 2
    maximum_pending_age: int = 6
    maximum_consecutive_misses: int = 2
    kinematic_veto_threshold: float = 0.12
    geometric_veto_threshold: float = 0.12
    veto_penalty: float = 0.35
    maximum_consecutive_vetoes: int = 2
    strict_threshold: float = 0.55
    position_scale: float = 0.75
    area_scale: float = 0.60
    use_relative_confidence: bool = True
    use_kinematic: bool = True
    use_geometric: bool = False
    use_veto: bool = True
    use_evidence: bool = True

    @classmethod
    def from_mapping(cls, data: dict[str, Any] | None) -> "TrustGuardConfig":
        if not data:
            return cls()
        allowed = set(cls.__dataclass_fields__)  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in data.items() if k in allowed})


@dataclass(frozen=True)
class CandidateDecision:
    status: str
    accepted: bool
    evidence: float
    instantaneous_trust: float
    limiting_feature: str
    veto_reasons: tuple[str, ...]
    confidence_absolute: float
    confidence_support: float
    confidence_relative: float | None
    kinematic_support: float | None
    geometric_support: float | None


class TrustGuard:
    def __init__(self, cfg: TrustGuardConfig):
        self.cfg = cfg
        self.states: dict[tuple[str, str], CandidateState] = {}
        self.kinematic: dict[tuple[str, str], KinematicTrackState] = {}
        self.confidence = RollingPercentileConfidence(
            window=int(cfg.relative_confidence_window),
            min_history=int(cfg.relative_confidence_min_history),
        )

    def update(
        self,
        *,
        sequence_id: str,
        tracklet_id: str,
        frame_id: int,
        confidence: float,
        bbox: tuple[float, float, float, float],
        area_norm: float,
        class_group: str,
        geometric: float | None = None,
        observed: bool = True,
    ) -> tuple[CandidateDecision, DecisionExplanation]:
        key = (str(sequence_id), str(tracklet_id))
        state = self.states.setdefault(key, CandidateState())
        status_before = status_value(state.status)
        if not observed:
            decision = self._missing_update(state)
            return decision, self._explain(sequence_id, tracklet_id, frame_id, status_before, decision)

        kinematic_state = self.kinematic.setdefault(
            key,
            KinematicTrackState(position_scale=self.cfg.position_scale, area_scale=self.cfg.area_scale),
        )
        kinematic_value, kinematic_available = kinematic_state.score(bbox)
        rel_key = (size_bin_from_area_norm(area_norm), class_group)
        rel = self.confidence.score_then_update(float(confidence), rel_key) if self.cfg.use_relative_confidence else None
        decision = self.update_candidate(
            state=state,
            confidence=float(confidence),
            relative_confidence=rel,
            kinematic=kinematic_value if (self.cfg.use_kinematic and kinematic_available) else None,
            geometric=geometric if self.cfg.use_geometric else None,
            observed=True,
        )
        kinematic_state.update(bbox)
        explanation = self._explain(sequence_id, tracklet_id, frame_id, status_before, decision)
        return decision, explanation

    def _missing_update(self, state: CandidateState) -> CandidateDecision:
        state.age += 1
        state.consecutive_misses += 1
        state.evidence *= self.cfg.evidence_decay
        if state.consecutive_misses > self.cfg.maximum_consecutive_misses:
            state.status = CandidateStatus.REJECTED
        return CandidateDecision(
            status=status_value(state.status),
            accepted=state.status == CandidateStatus.CONFIRMED,
            evidence=float(state.evidence),
            instantaneous_trust=0.0,
            limiting_feature="missing",
            veto_reasons=(),
            confidence_absolute=0.0,
            confidence_support=0.0,
            confidence_relative=None,
            kinematic_support=None,
            geometric_support=None,
        )

    def update_candidate(
        self,
        state: CandidateState,
        confidence: float,
        relative_confidence: float | None,
        kinematic: float | None,
        geometric: float | None,
        observed: bool,
    ) -> CandidateDecision:
        if not observed:
            return self._missing_update(state)
        cfg = self.cfg
        state.age += 1
        state.consecutive_misses = 0
        confidence_abs = normalize_absolute_confidence(confidence, cfg.absolute_confidence_floor)
        if confidence < cfg.absolute_confidence_floor:
            confidence_support = 0.0
        elif relative_confidence is None:
            confidence_support = confidence_abs
        else:
            confidence_support = (1.0 - cfg.relative_confidence_weight) * confidence_abs + cfg.relative_confidence_weight * float(relative_confidence)
        channels: dict[str, float] = {"confidence": max(0.0, min(1.0, confidence_support))}
        if kinematic is not None:
            channels["kinematic"] = max(0.0, min(1.0, float(kinematic)))
        if geometric is not None:
            channels["geometric"] = max(0.0, min(1.0, float(geometric)))
        limiting_feature = min(channels, key=channels.get)
        q = min(channels.values())
        veto_reasons = []
        if confidence < cfg.absolute_confidence_floor:
            veto_reasons.append("absolute_confidence")
        if cfg.use_veto and kinematic is not None and kinematic < cfg.kinematic_veto_threshold:
            veto_reasons.append("kinematic")
        if cfg.use_veto and geometric is not None and geometric < cfg.geometric_veto_threshold:
            veto_reasons.append("geometric")
        veto = bool(veto_reasons)
        if cfg.mode == "strict":
            return self._strict_update(
                state,
                q,
                limiting_feature,
                veto_reasons,
                confidence_abs,
                confidence_support,
                relative_confidence,
                kinematic,
                geometric,
            )
        state.evidence = cfg.evidence_decay * state.evidence + (1.0 - cfg.evidence_decay) * q if cfg.use_evidence else q
        if veto:
            state.evidence = max(0.0, state.evidence - cfg.veto_penalty)
            state.consecutive_vetoes += 1
            state.consecutive_valid_hits = 0
        else:
            state.valid_hits += 1
            state.consecutive_valid_hits += 1
            state.consecutive_vetoes = 0
        if state.consecutive_vetoes >= cfg.maximum_consecutive_vetoes:
            state.status = CandidateStatus.REJECTED
        elif state.evidence >= cfg.confirmation_threshold and state.valid_hits >= cfg.minimum_valid_hits and not veto:
            state.status = CandidateStatus.CONFIRMED
        elif state.age >= cfg.maximum_pending_age:
            state.status = CandidateStatus.REJECTED
        return CandidateDecision(
            status=status_value(state.status),
            accepted=state.status == CandidateStatus.CONFIRMED,
            evidence=float(state.evidence),
            instantaneous_trust=float(q),
            limiting_feature=limiting_feature,
            veto_reasons=tuple(veto_reasons),
            confidence_support=float(confidence_support),
            confidence_absolute=float(confidence_abs),
            confidence_relative=relative_confidence,
            kinematic_support=kinematic,
            geometric_support=geometric,
        )

    def _strict_update(
        self,
        state: CandidateState,
        q: float,
        limiting_feature: str,
        veto_reasons: list[str],
        confidence_absolute: float,
        confidence_support: float,
        relative_confidence: float | None,
        kinematic: float | None,
        geometric: float | None,
    ) -> CandidateDecision:
        cfg = self.cfg
        state.evidence = q
        if q >= cfg.strict_threshold and not veto_reasons:
            state.valid_hits += 1
            state.consecutive_valid_hits += 1
            state.consecutive_vetoes = 0
        else:
            state.consecutive_valid_hits = 0
            if veto_reasons:
                state.consecutive_vetoes += 1
        state.recent_q_values.append(float(q))
        state.recent_q_values = state.recent_q_values[-max(1, int(cfg.minimum_valid_hits)) :]
        if state.consecutive_vetoes >= cfg.maximum_consecutive_vetoes:
            state.status = CandidateStatus.REJECTED
        elif (
            state.consecutive_valid_hits >= cfg.minimum_valid_hits
            and len(state.recent_q_values) >= cfg.minimum_valid_hits
            and min(state.recent_q_values) >= cfg.strict_threshold
        ):
            state.status = CandidateStatus.CONFIRMED
        elif state.age >= cfg.maximum_pending_age:
            state.status = CandidateStatus.REJECTED
        return CandidateDecision(
            status=status_value(state.status),
            accepted=state.status == CandidateStatus.CONFIRMED,
            evidence=float(state.evidence),
            instantaneous_trust=float(q),
            limiting_feature=limiting_feature,
            veto_reasons=tuple(veto_reasons),
            confidence_support=float(confidence_support),
            confidence_absolute=float(confidence_absolute),
            confidence_relative=relative_confidence,
            kinematic_support=kinematic,
            geometric_support=geometric,
        )

    @staticmethod
    def _explain(sequence_id: str, tracklet_id: str, frame_id: int, status_before: str, decision: CandidateDecision) -> DecisionExplanation:
        return DecisionExplanation(
            sequence_id=str(sequence_id),
            tracklet_id=str(tracklet_id),
            frame_id=int(frame_id),
            status_before=status_before,
            status_after=decision.status,
            confidence_absolute=decision.confidence_absolute,
            confidence_relative=decision.confidence_relative,
            confidence_support=decision.confidence_support,
            kinematic_support=decision.kinematic_support,
            geometric_support=decision.geometric_support,
            instantaneous_trust=decision.instantaneous_trust,
            accumulated_evidence=decision.evidence,
            limiting_feature=decision.limiting_feature,
            veto_reasons=decision.veto_reasons,
        )


def run_trust_guard_dataframe(det: pd.DataFrame, cfg: TrustGuardConfig) -> pd.DataFrame:
    guard = TrustGuard(cfg)
    rows: list[dict[str, Any]] = []
    sort_cols = ["sequence_id", "tracklet_id", "frame_id"]
    for row in det.sort_values(sort_cols).itertuples():
        area_norm = _area_norm(row)
        decision, explanation = guard.update(
            sequence_id=str(row.sequence_id),
            tracklet_id=str(row.tracklet_id),
            frame_id=int(row.frame_id),
            confidence=float(row.confidence),
            bbox=(float(row.x1), float(row.y1), float(row.x2), float(row.y2)),
            area_norm=area_norm,
            class_group=str(getattr(row, "class_name", "unknown")),
            geometric=float(getattr(row, "s_i", 1.0)),
        )
        rows.append(
            {
                "row_index": int(row.Index),
                "sequence_id": explanation.sequence_id,
                "tracklet_id": explanation.tracklet_id,
                "frame_id": explanation.frame_id,
                "accepted": decision.accepted,
                "status": decision.status,
                "evidence": decision.evidence,
                "instantaneous_trust": decision.instantaneous_trust,
                "limiting_feature": decision.limiting_feature,
                "veto_reasons": ";".join(decision.veto_reasons),
                "confidence_support": decision.confidence_support,
                "confidence_absolute": decision.confidence_absolute,
                "confidence_relative": decision.confidence_relative,
                "kinematic_support": decision.kinematic_support,
                "geometric_support": decision.geometric_support,
                "eval_is_tp": bool(getattr(row, "eval_is_tp", False)),
            }
        )
    return pd.DataFrame(rows).sort_values("row_index").reset_index(drop=True)


def _area_norm(row: Any) -> float:
    area = float(getattr(row, "bbox_area", 0.0))
    image_path = str(getattr(row, "image_path", ""))
    # VisDrone validation frames in this project are 960x540 or 1920x1080-like;
    # feature_audit does not store image dimensions. The exact scale is only
    # used for stable rolling bins, so use a conservative HD denominator.
    if "1920" in image_path:
        denom = 1920.0 * 1080.0
    else:
        denom = 960.0 * 540.0
    return area / max(1.0, denom)
