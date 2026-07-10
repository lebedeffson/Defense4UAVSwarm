from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DecisionExplanation:
    sequence_id: str
    tracklet_id: str
    frame_id: int
    status_before: str
    status_after: str
    confidence_absolute: float
    confidence_relative: float | None
    confidence_support: float
    kinematic_support: float | None
    geometric_support: float | None
    instantaneous_trust: float
    accumulated_evidence: float
    limiting_feature: str
    veto_reasons: tuple[str, ...]
