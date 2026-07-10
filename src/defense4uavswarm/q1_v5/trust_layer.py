from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


Array = np.ndarray


@dataclass(frozen=True)
class TrustFeatures:
    confidence: Array
    kinematic: Array | None = None
    temporal: Array | None = None
    geometric: Array | None = None


@dataclass(frozen=True)
class TrustConfig:
    mode: Literal["strict", "balanced", "adaptive"]
    active_channels: tuple[str, ...]
    beta: float
    acceptance_threshold: float
    temporal_age_scale: float
    temporal_floor: float
    recovery_min_age: int | None = None
    recovery_min_confidence: float | None = None


@dataclass(frozen=True)
class TrustDecision:
    q: Array
    corrected_confidence: Array
    accepted: Array
    recovery: Array


def _clip01(values: Array) -> Array:
    return np.clip(np.asarray(values, dtype=float), 0.0, 1.0)


def temporal_from_age(age: Array, scale: float, floor: float = 0.0) -> Array:
    if scale <= 0:
        raise ValueError("temporal_age_scale must be positive")
    values = _clip01(np.asarray(age, dtype=float) / float(scale))
    return np.maximum(values, float(floor))


def _channel(features: TrustFeatures, channel: str) -> Array:
    if channel == "confidence":
        return _clip01(features.confidence)
    value = getattr(features, channel, None)
    if value is None:
        raise ValueError(f"Active trust channel is missing: {channel}")
    return _clip01(value)


def compute_q(features: TrustFeatures, active_channels: tuple[str, ...]) -> Array:
    if not active_channels:
        raise ValueError("At least one active trust channel is required")
    stacked = np.vstack([_channel(features, ch) for ch in active_channels])
    return np.min(stacked, axis=0)


def decide(features: TrustFeatures, cfg: TrustConfig, temporal_age: Array | None = None) -> TrustDecision:
    confidence = _clip01(features.confidence)
    q = compute_q(features, cfg.active_channels)
    corrected = confidence * (float(cfg.beta) + (1.0 - float(cfg.beta)) * q)
    accepted = corrected >= float(cfg.acceptance_threshold)
    recovery = np.zeros(len(confidence), dtype=bool)
    if cfg.mode == "strict":
        if cfg.recovery_min_age is not None or cfg.recovery_min_confidence is not None:
            raise ValueError("Strict mode cannot define recovery thresholds")
    elif cfg.mode in {"balanced", "adaptive"}:
        if cfg.recovery_min_age is not None and cfg.recovery_min_confidence is not None:
            if temporal_age is None:
                raise ValueError("Balanced/adaptive recovery requires temporal_age")
            recovery = (np.asarray(temporal_age, dtype=float) >= float(cfg.recovery_min_age)) & (
                confidence >= float(cfg.recovery_min_confidence)
            )
            accepted = accepted | recovery
    else:
        raise ValueError(f"Unknown trust mode: {cfg.mode}")
    return TrustDecision(q=q, corrected_confidence=corrected, accepted=np.asarray(accepted, dtype=bool), recovery=recovery)


def config_hash_payload(cfg: TrustConfig) -> dict[str, object]:
    return {
        "mode": cfg.mode,
        "active_channels": list(cfg.active_channels),
        "beta": cfg.beta,
        "acceptance_threshold": cfg.acceptance_threshold,
        "temporal_age_scale": cfg.temporal_age_scale,
        "temporal_floor": cfg.temporal_floor,
        "recovery_min_age": cfg.recovery_min_age,
        "recovery_min_confidence": cfg.recovery_min_confidence,
    }

