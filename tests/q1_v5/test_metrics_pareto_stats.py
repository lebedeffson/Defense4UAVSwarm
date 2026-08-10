from __future__ import annotations

import numpy as np
import pandas as pd

from defense4uavswarm.q1_v5.map_contamination import contamination_metrics
from defense4uavswarm.q1_v5.pareto import pareto_front
from defense4uavswarm.q1_v5.statistics import noninferior, paired_bootstrap_delta, superiority_lower_is_better


def test_false_track_occupancy_and_peak_concurrency():
    det = pd.DataFrame(
        {
            "tracklet_id": ["a", "a", "b", "c"],
            "sequence_id": ["s", "s", "s", "s"],
            "frame_id": [1, 2, 2, 2],
            "eval_is_tp": [False, False, True, False],
            "confidence": [0.5, 0.4, 0.9, 0.3],
        }
    )
    m = contamination_metrics(det, np.array([True, True, True, True]))
    assert m["false_new_tracks"] == 2
    assert m["false_track_occupancy_frames"] == 3
    assert m["peak_concurrent_false_tracks"] == 2


def test_pareto_equal_points_not_strictly_dominating():
    df = pd.DataFrame({"F1": [0.5, 0.5], "false_new_tracks": [10, 10]})
    flags = pareto_front(df)
    assert flags.tolist() == [True, True]


def test_pareto_dominance():
    df = pd.DataFrame({"F1": [0.6, 0.5], "false_new_tracks": [8, 10]})
    flags = pareto_front(df)
    assert flags.tolist() == [True, False]


def test_paired_bootstrap_reproducible_and_rules():
    base = np.array([1.0, 1.0, 1.0])
    method = np.array([0.9, 0.9, 0.9])
    stat = paired_bootstrap_delta(base, method, n_resamples=100, seed=1)
    assert stat["mean_delta"] < 0
    assert noninferior(stat["ci95_low"], 0.2)
    assert superiority_lower_is_better(stat["ci95_high"])

