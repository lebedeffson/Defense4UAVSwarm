from __future__ import annotations

from dataclasses import dataclass


@dataclass
class FrameBudgetRecord:
    sequence_id: str
    frame_id: int
    stage: str
    eligibility_event_count: int
    configured_fraction: float
    capacity: float
    balance_before_refill: float
    refill_amount: float
    overflow_amount: float
    balance_before_decision: float
    eligible_above_threshold: int
    admitted_count: int
    balance_after_decision: float
    cumulative_refill: float
    cumulative_overflow: float
    cumulative_admissions: int
    conservation_residual: float


class FrameTokenBucket:
    def __init__(self, *, fraction: float, capacity: float, stage: str) -> None:
        if fraction < 0:
            raise ValueError("fraction must be non-negative")
        if capacity < 0:
            raise ValueError("capacity must be non-negative")
        self.fraction = float(fraction)
        self.capacity = float(capacity)
        self.stage = str(stage)
        self.balance = 0.0
        self.cumulative_refill = 0.0
        self.cumulative_overflow = 0.0
        self.cumulative_admissions = 0
        self._frame_ready = False
        self._balance_before_refill = 0.0
        self._refill_amount = 0.0
        self._overflow_amount = 0.0
        self._balance_before_decision = 0.0

    def begin_frame(self, *, eligibility_event_count: int) -> None:
        m = int(eligibility_event_count)
        if m < 0:
            raise ValueError("eligibility_event_count must be non-negative")
        self._balance_before_refill = float(self.balance)
        self._refill_amount = float(m) * self.fraction
        self.cumulative_refill += self._refill_amount
        raw = self.balance + self._refill_amount
        self._overflow_amount = max(0.0, raw - self.capacity)
        self.cumulative_overflow += self._overflow_amount
        self.balance = raw - self._overflow_amount
        self._balance_before_decision = float(self.balance)
        self._frame_ready = True

    def try_admit(self) -> bool:
        if not self._frame_ready:
            raise RuntimeError("begin_frame must be called before admissions")
        if self.balance + 1e-12 < 1.0:
            return False
        self.balance -= 1.0
        self.cumulative_admissions += 1
        return True

    def record(
        self,
        *,
        sequence_id: str,
        frame_id: int,
        eligibility_event_count: int,
        eligible_above_threshold: int,
        admitted_count: int,
    ) -> FrameBudgetRecord:
        residual = self.conservation_residual()
        self._frame_ready = False
        return FrameBudgetRecord(
            sequence_id=str(sequence_id),
            frame_id=int(frame_id),
            stage=self.stage,
            eligibility_event_count=int(eligibility_event_count),
            configured_fraction=float(self.fraction),
            capacity=float(self.capacity),
            balance_before_refill=float(self._balance_before_refill),
            refill_amount=float(self._refill_amount),
            overflow_amount=float(self._overflow_amount),
            balance_before_decision=float(self._balance_before_decision),
            eligible_above_threshold=int(eligible_above_threshold),
            admitted_count=int(admitted_count),
            balance_after_decision=float(self.balance),
            cumulative_refill=float(self.cumulative_refill),
            cumulative_overflow=float(self.cumulative_overflow),
            cumulative_admissions=int(self.cumulative_admissions),
            conservation_residual=float(residual),
        )

    def conservation_residual(self) -> float:
        expected = self.cumulative_refill - self.cumulative_overflow - self.balance
        return float(self.cumulative_admissions - expected)


def assert_bucket_invariant(bucket: FrameTokenBucket, *, tolerance: float = 1e-9) -> None:
    residual = abs(bucket.conservation_residual())
    if residual > tolerance:
        raise AssertionError(f"{bucket.stage} token bucket conservation residual {residual} > {tolerance}")
