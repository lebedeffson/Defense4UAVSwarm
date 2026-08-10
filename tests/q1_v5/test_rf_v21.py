from __future__ import annotations

import pandas as pd

from defense4uavswarm.q1_v5.rf_v21 import (
    FEATURE_COLUMNS,
    RFBudgetSpec,
    build_episode_feature_table,
    score_episode_predictions,
    simulate_budgeted_interventions,
)


def test_rf_features_use_episode_start_only() -> None:
    candidates = pd.DataFrame(
        {
            "sequence_id": ["s", "s"],
            "tracklet_id": ["t", "t"],
            "episode_id": [0, 0],
            "frame_id": [1, 2],
            "confidence": [0.2, 0.9],
            "c_i": [0.2, 0.9],
            "k_i": [0.1, 0.8],
            "s_i": [1.0, 1.0],
            "Q_i": [0.1, 0.8],
            "confidence_new": [0.12, 0.8],
            "bbox_area": [100.0, 400.0],
            "bbox_aspect_ratio": [1.0, 2.0],
            "num_detections_in_frame": [4, 3],
            "temporal_age": [0, 1],
            "x1": [0.0, 10.0],
            "y1": [0.0, 10.0],
            "x2": [10.0, 30.0],
            "y2": [10.0, 30.0],
        }
    )
    features = build_episode_feature_table(candidates)
    assert len(features) == 1
    assert features.iloc[0]["confidence"] == 0.2
    assert features.iloc[0]["bbox_area"] == 100.0
    assert set(FEATURE_COLUMNS).issubset(features.columns)


def test_score_episode_predictions_rewards_true_retention_and_false_rejection() -> None:
    episodes = pd.DataFrame({"true_episode": [1, 0], "false_episode": [0, 1]})
    score = score_episode_predictions(episodes, [0.1, 0.9], threshold=0.5)
    assert score["true_retention"] == 1.0
    assert score["false_rejection"] == 1.0
    assert score["false_acceptance"] == 0.0


def test_budgeted_interventions_need_accumulated_budget() -> None:
    episodes = pd.DataFrame(
        {
            "sequence_id": ["s", "s", "s"],
            "tracklet_id": ["a", "b", "c"],
            "episode_id": [0, 0, 0],
            "first_frame_id": [1, 2, 3],
        }
    )
    decisions = simulate_budgeted_interventions(
        episodes,
        [0.9, 0.9, 0.9],
        RFBudgetSpec(decision_threshold=0.5, intervention_fraction=0.5, budget_capacity=2.0),
    )
    assert decisions["intervened"].tolist() == [False, True, False]
    assert decisions["budget_tokens_before"].tolist() == [0.5, 1.0, 0.5]
