from __future__ import annotations

import pandas as pd
import pytest

from defense4uavswarm.q1_v5.confidence_normalizer import RollingPercentileConfidence
from defense4uavswarm.q1_v5.evaluation_matching import MatchingConfig, build_frame_index, evaluate_acceptance_mask
from defense4uavswarm.q1_v5.initiation_gate import bayesian_terminal_gate, m_of_n_confirmation
from defense4uavswarm.q1_v5.kinematic_consistency import KinematicTrackState


def _gt() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["s", "s"],
            "frame_id": [1, 2],
            "object_id": ["g1", "g2"],
            "class_id": [4, 4],
            "class_name": ["car", "car"],
            "x1": [0.0, 50.0],
            "y1": [0.0, 50.0],
            "x2": [10.0, 60.0],
            "y2": [10.0, 60.0],
        }
    )


def test_filtering_recomputes_matching_second_detection_becomes_tp():
    det = pd.DataFrame(
        {
            "det_id": ["a", "b"],
            "sequence_id": ["s", "s"],
            "frame_id": [1, 1],
            "tracklet_id": ["ta", "tb"],
            "class_id": [2, 2],
            "class_name": ["car", "car"],
            "confidence": [0.9, 0.8],
            "x1": [0.0, 0.0],
            "y1": [0.0, 0.0],
            "x2": [10.0, 10.0],
            "y2": [10.0, 10.0],
        }
    )
    matched, _, metrics = evaluate_acceptance_mask(det, pd.Series([False, True]), _gt().iloc[:1], build_frame_index(_gt().iloc[:1], det), MatchingConfig())
    assert len(matched) == 1
    assert bool(matched.iloc[0]["eval_is_tp"])
    assert metrics["TP"] == 1
    assert metrics["FN"] == 0


def test_empty_detection_frame_counts_fn():
    det = pd.DataFrame(columns=["det_id", "sequence_id", "frame_id", "tracklet_id", "class_id", "class_name", "confidence", "x1", "y1", "x2", "y2"])
    _, _, metrics = evaluate_acceptance_mask(det, pd.Series([], dtype=bool), _gt(), build_frame_index(_gt(), det), MatchingConfig())
    assert metrics["TP"] == 0
    assert metrics["FN"] == 2


def test_mofn_and_bayesian_terminal_after_confirmation():
    det = pd.DataFrame(
        {
            "sequence_id": ["s"] * 5,
            "tracklet_id": ["t"] * 5,
            "frame_id": [1, 2, 3, 4, 5],
            "confidence": [0.9, 0.9, 0.9, 0.01, 0.01],
        }
    )
    assert m_of_n_confirmation(det, 2, 3, 0.5).tolist() == [False, True, True, True, True]
    assert bayesian_terminal_gate(det, threshold=1.0).tolist()[-2:] == [True, True]


def test_same_frame_candidates_do_not_influence_relative_rank():
    r = RollingPercentileConfidence(window=10, min_history=1)
    key = ("s", "small", "car")
    r.update_batch([0.5], [key])
    scores = r.score_batch([0.4, 0.9], [key, key])
    assert scores == [0.0, 1.0]
    # If the first same-frame row leaked into history, the second score would
    # be based on two values. It must still see only the previous-frame history.
    assert len(r.history[key]) == 1


def test_kinematic_prediction_uses_frame_dt():
    k = KinematicTrackState()
    k.update((0, 0, 10, 10), frame_id=1)
    k.update((10, 0, 20, 10), frame_id=3)
    score, available = k.score((20, 0, 30, 10), frame_id=5)
    assert available
    assert score == pytest.approx(1.0)
