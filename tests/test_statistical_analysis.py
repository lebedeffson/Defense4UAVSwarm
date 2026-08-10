import numpy as np

from scripts.run_q1_statistical_analysis import bootstrap_ci


def test_bootstrap_ci_order() -> None:
    lo, hi = bootstrap_ci(np.array([1.0, 2.0, 3.0]), 20, np.random.default_rng(1))
    assert lo <= hi
