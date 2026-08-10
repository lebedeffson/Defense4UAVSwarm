from __future__ import annotations

import numpy as np
import pytest

from defense4uavswarm.q1_v5.trust_layer import TrustConfig, TrustFeatures, decide, temporal_from_age


def test_strict_q_not_above_any_feature():
    f = TrustFeatures(confidence=np.array([0.9, 0.4]), temporal=np.array([0.2, 0.8]))
    cfg = TrustConfig("strict", ("confidence", "temporal"), 0.6, 0.5, 3.0, 0.0)
    d = decide(f, cfg)
    assert np.all(d.q <= f.confidence)
    assert np.all(d.q <= f.temporal)


def test_strict_corrected_conf_not_above_original():
    f = TrustFeatures(confidence=np.array([0.9, 0.4]), temporal=np.array([0.2, 0.8]))
    cfg = TrustConfig("strict", ("confidence", "temporal"), 0.6, 0.1, 3.0, 0.0)
    d = decide(f, cfg)
    assert np.all(d.corrected_confidence <= f.confidence)


def test_strict_has_no_recovery():
    f = TrustFeatures(confidence=np.array([0.9]), temporal=np.array([1.0]))
    cfg = TrustConfig("strict", ("confidence", "temporal"), 0.6, 0.5, 3.0, 0.0, 2, 0.1)
    with pytest.raises(ValueError):
        decide(f, cfg, temporal_age=np.array([3]))


def test_balanced_recovery_is_explicit():
    f = TrustFeatures(confidence=np.array([0.2]), temporal=np.array([0.0]))
    cfg = TrustConfig("balanced", ("confidence", "temporal"), 0.6, 0.9, 3.0, 0.0, 2, 0.1)
    d = decide(f, cfg, temporal_age=np.array([3]))
    assert d.recovery[0]
    assert d.accepted[0]


def test_temporal_values_clipped_to_unit_interval():
    out = temporal_from_age(np.array([-1, 1, 10]), 3.0, 0.2)
    assert np.all((0 <= out) & (out <= 1))
    assert out[0] == 0.2

