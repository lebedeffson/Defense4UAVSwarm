import pandas as pd

from scripts.run_q1_error_attribution import reason


def test_error_reason_confidence() -> None:
    r = pd.Series({"c_i": 0.1, "k_i": 0.8, "temporal_age": 3})
    assert reason(r) == "detector_confidence_min"
