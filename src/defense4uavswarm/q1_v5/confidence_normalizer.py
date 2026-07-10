from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Hashable


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def normalize_absolute_confidence(confidence: float, floor: float) -> float:
    c = float(confidence)
    f = float(floor)
    if c < f:
        return 0.0
    if f >= 1.0:
        return 1.0
    return clamp01((c - f) / (1.0 - f))


def size_bin_from_area_norm(area_norm: float, small_q: float = 0.0009, medium_q: float = 0.0040) -> str:
    area = float(area_norm)
    if area < small_q:
        return "small"
    if area < medium_q:
        return "medium"
    return "large"


@dataclass
class RollingPercentileConfidence:
    window: int = 500
    min_history: int = 30
    history: dict[Hashable, deque[float]] = field(default_factory=lambda: defaultdict(deque))

    def percentile_rank(self, confidence: float, key: Hashable) -> float | None:
        values = self.history[key]
        if len(values) < self.min_history:
            return None
        c = float(confidence)
        return sum(1 for x in values if x <= c) / float(len(values))

    def update(self, confidence: float, key: Hashable) -> None:
        values = self.history[key]
        values.append(float(confidence))
        while len(values) > self.window:
            values.popleft()

    def score_then_update(self, confidence: float, key: Hashable) -> float | None:
        score = self.percentile_rank(confidence, key)
        self.update(confidence, key)
        return score
