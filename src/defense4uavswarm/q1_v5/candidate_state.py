from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class CandidateStatus(str, Enum):
    PENDING = "pending"
    CONFIRMED = "confirmed"
    REJECTED = "rejected"


@dataclass
class CandidateState:
    evidence: float = 0.0
    age: int = 0
    first_frame_id: int | None = None
    last_frame_id: int | None = None
    observed_hits: int = 0
    valid_hits: int = 0
    consecutive_valid_hits: int = 0
    consecutive_misses: int = 0
    consecutive_vetoes: int = 0
    status: CandidateStatus = CandidateStatus.PENDING
    confirmation_frame_id: int | None = None
    rejection_frame_id: int | None = None
    recent_q_values: list[float] = field(default_factory=list)
    evidence_history: list[float] = field(default_factory=list)


def status_value(status: CandidateStatus | str) -> str:
    return status.value if isinstance(status, CandidateStatus) else str(status)
