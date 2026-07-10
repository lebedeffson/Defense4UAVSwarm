from __future__ import annotations

import pandas as pd
import pytest

from defense4uavswarm.q1_v5.confidence_normalizer import RollingPercentileConfidence
from defense4uavswarm.q1_v5.evaluation_matching import MatchingConfig, build_frame_index, evaluate_acceptance_mask
from defense4uavswarm.q1_v5.initiation_gate import GateConfig, assign_episode_ids, bayesian_terminal_gate, m_of_n_confirmation
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


def test_matching_maximizes_cardinality_not_confidence_order():
    gt = pd.DataFrame(
        {
            "sequence_id": ["s", "s"],
            "frame_id": [1, 1],
            "object_id": ["g1", "g2"],
            "class_id": [4, 4],
            "class_name": ["car", "car"],
            "x1": [0.0, 20.0],
            "y1": [0.0, 0.0],
            "x2": [10.0, 30.0],
            "y2": [10.0, 10.0],
        }
    )
    det = pd.DataFrame(
        {
            "det_id": ["high", "low"],
            "sequence_id": ["s", "s"],
            "frame_id": [1, 1],
            "tracklet_id": ["ta", "tb"],
            "class_id": [4, 4],
            "class_name": ["car", "car"],
            "confidence": [0.99, 0.50],
            # high overlaps both and would greedily take g2; low can only take g1.
            "x1": [10.0, 0.0],
            "y1": [0.0, 0.0],
            "x2": [30.0, 10.0],
            "y2": [10.0, 10.0],
        }
    )
    _, _, metrics = evaluate_acceptance_mask(det, pd.Series([True, True]), gt, build_frame_index(gt, det), MatchingConfig(iou_threshold=0.3, class_matching="class_agnostic"))
    assert metrics["TP"] == 2
    assert metrics["FN"] == 0


def test_ignore_is_applied_only_to_unmatched_detections():
    gt = pd.DataFrame(
        {
            "sequence_id": ["s"],
            "frame_id": [1],
            "object_id": ["g1"],
            "class_id": [4],
            "class_name": ["car"],
            "x1": [0.0],
            "y1": [0.0],
            "x2": [10.0],
            "y2": [10.0],
        }
    )
    ignored = gt.rename(columns={"object_id": "gt_track_id"}).assign(is_ignored=True)
    det = pd.DataFrame(
        {
            "det_id": ["tp", "ignored_fp"],
            "sequence_id": ["s", "s"],
            "frame_id": [1, 1],
            "tracklet_id": ["t1", "t2"],
            "class_id": [4, 4],
            "class_name": ["car", "car"],
            "confidence": [0.9, 0.8],
            "x1": [0.0, 0.0],
            "y1": [0.0, 0.0],
            "x2": [10.0, 10.0],
            "y2": [10.0, 10.0],
        }
    )
    matched, _, metrics = evaluate_acceptance_mask(det, pd.Series([True, True]), gt, build_frame_index(gt, det), MatchingConfig(), ignored)
    assert metrics["TP"] == 1
    assert metrics["FP"] == 0
    assert metrics["ignored_unmatched_detections"] == 1
    assert matched.loc[matched["det_id"].eq("tp"), "eval_is_tp"].iloc[0]


def test_empty_detection_frame_counts_fn():
    det = pd.DataFrame(columns=["det_id", "sequence_id", "frame_id", "tracklet_id", "class_id", "class_name", "confidence", "x1", "y1", "x2", "y2"])
    _, _, metrics = evaluate_acceptance_mask(det, pd.Series([], dtype=bool), _gt(), build_frame_index(_gt(), det), MatchingConfig())
    assert metrics["TP"] == 0
    assert metrics["FN"] == 2


def test_duplicate_observation_key_fixture_is_explicit():
    det = pd.DataFrame(
        {
            "det_id": ["a", "a"],
            "sequence_id": ["s", "s"],
            "frame_id": [1, 1],
            "tracklet_id": ["t", "t"],
            "episode_id": [0, 0],
        }
    )
    assert det.duplicated(["sequence_id", "tracklet_id", "episode_id", "frame_id", "det_id"]).any()


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


def test_rejected_episode_is_terminal_and_no_confirm_after_reject():
    det = pd.DataFrame(
        {
            "sequence_id": ["s"] * 4,
            "tracklet_id": ["t"] * 4,
            "frame_id": [1, 2, 3, 4],
            "confidence": [0.1, 0.1, 0.99, 0.99],
        }
    )
    # max age 1 rejects before the later high-confidence trigger can confirm.
    accepted = m_of_n_confirmation(det, 1, 1, 0.5, GateConfig(maximum_pending_age_frames=1))
    assert accepted.tolist() == [False, False, False, False]


def test_long_gap_creates_new_episode_and_requires_new_confirmation():
    det = pd.DataFrame(
        {
            "sequence_id": ["s"] * 4,
            "tracklet_id": ["t"] * 4,
            "frame_id": [1, 2, 10, 11],
            "confidence": [0.9, 0.9, 0.9, 0.9],
        }
    )
    with_episode = assign_episode_ids(det, max_track_gap=1)
    assert with_episode["episode_id"].tolist() == [0, 0, 1, 1]
    assert m_of_n_confirmation(with_episode, 2, 3, 0.5, GateConfig(max_track_gap=1)).tolist() == [False, True, False, True]


def test_summary_does_not_include_fake_idf1():
    det = pd.DataFrame(
        {
            "det_id": ["a"],
            "sequence_id": ["s"],
            "frame_id": [1],
            "tracklet_id": ["t"],
            "class_id": [4],
            "class_name": ["car"],
            "confidence": [0.9],
            "x1": [0.0],
            "y1": [0.0],
            "x2": [10.0],
            "y2": [10.0],
        }
    )
    _, _, metrics = evaluate_acceptance_mask(det, pd.Series([True]), _gt().iloc[:1], build_frame_index(_gt().iloc[:1], det), MatchingConfig())
    assert "IDF1" not in metrics


def test_composite_gt_identity_contains_sequence():
    det = pd.DataFrame(
        {
            "det_id": ["a"],
            "sequence_id": ["s"],
            "frame_id": [1],
            "tracklet_id": ["t"],
            "class_id": [4],
            "class_name": ["car"],
            "confidence": [0.9],
            "x1": [0.0],
            "y1": [0.0],
            "x2": [10.0],
            "y2": [10.0],
        }
    )
    matched, _, _ = evaluate_acceptance_mask(det, pd.Series([True]), _gt().iloc[:1], build_frame_index(_gt().iloc[:1], det), MatchingConfig())
    assert matched["matched_gt_id"].iloc[0] == "s:g1"


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
