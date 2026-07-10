from __future__ import annotations

import pandas as pd
import pytest

from defense4uavswarm.q1_v5.confidence_normalizer import RollingPercentileConfidence, normalize_absolute_confidence
from defense4uavswarm.q1_v5.kinematic_consistency import KinematicTrackState
from defense4uavswarm.q1_v5.trust_guard import TrustGuardConfig, run_trust_guard_dataframe


def test_absolute_confidence_floor_zeroes_below_floor():
    assert normalize_absolute_confidence(0.06, 0.07) == 0.0
    assert 0.0 < normalize_absolute_confidence(0.50, 0.07) < 1.0


def test_relative_confidence_is_past_only():
    r = RollingPercentileConfidence(window=5, min_history=2)
    key = ("small", "car")
    assert r.score_then_update(0.4, key) is None
    assert r.score_then_update(0.6, key) is None
    # This score can only see the previous two values, not the current update.
    assert r.score_then_update(0.5, key) == pytest.approx(0.5)


def test_kinematics_unavailable_for_first_two_observations():
    k = KinematicTrackState()
    assert k.score((0, 0, 10, 10)) == (None, False)
    k.update((0, 0, 10, 10))
    assert k.score((1, 0, 11, 10)) == (None, False)
    k.update((1, 0, 11, 10))
    score, available = k.score((2, 0, 12, 10))
    assert available
    assert score > 0.9


def test_veto_frame_cannot_confirm():
    det = pd.DataFrame(
        {
            "sequence_id": ["s", "s", "s"],
            "tracklet_id": ["t", "t", "t"],
            "frame_id": [1, 2, 3],
            "confidence": [0.9, 0.9, 0.9],
            "x1": [0, 100, 0],
            "y1": [0, 100, 0],
            "x2": [10, 110, 10],
            "y2": [10, 110, 10],
            "bbox_area": [100, 100, 100],
            "image_path": ["", "", ""],
            "class_name": ["car", "car", "car"],
            "eval_is_tp": [True, True, True],
        }
    )
    cfg = TrustGuardConfig(
        relative_confidence_min_history=100,
        confirmation_threshold=0.70,
        minimum_valid_hits=3,
        kinematic_veto_threshold=0.99,
        veto_penalty=0.35,
        use_kinematic=True,
        use_veto=True,
    )
    events = run_trust_guard_dataframe(det, cfg)
    third = events.iloc[2]
    assert "kinematic" in third["veto_reasons"]
    assert not bool(third["accepted"])


def test_operational_mode_does_not_accept_by_age_only():
    det = pd.DataFrame(
        {
            "sequence_id": ["s"] * 4,
            "tracklet_id": ["t"] * 4,
            "frame_id": [1, 2, 3, 4],
            "confidence": [0.08, 0.08, 0.08, 0.08],
            "x1": [0, 1, 2, 3],
            "y1": [0, 0, 0, 0],
            "x2": [10, 11, 12, 13],
            "y2": [10, 10, 10, 10],
            "bbox_area": [100, 100, 100, 100],
            "image_path": [""] * 4,
            "class_name": ["car"] * 4,
            "eval_is_tp": [True] * 4,
        }
    )
    cfg = TrustGuardConfig(
        absolute_confidence_floor=0.07,
        relative_confidence_min_history=100,
        confirmation_threshold=0.55,
        minimum_valid_hits=2,
        use_kinematic=False,
        use_veto=True,
    )
    events = run_trust_guard_dataframe(det, cfg)
    assert not events["accepted"].any()
