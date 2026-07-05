import pandas as pd

from defense4uavswarm.swarm_tnorm import _q


def test_feature_sets_min():
    f = pd.DataFrame({"c_i": [0.8], "k_i": [0.5], "s_i": [0.4], "x_i": [1.0]})
    assert float(_q(f, "min", "c_only").iloc[0]) == 0.8
    assert float(_q(f, "min", "c_k").iloc[0]) == 0.5
    assert float(_q(f, "min", "c_s").iloc[0]) == 0.4
    assert float(_q(f, "min", "k_s").iloc[0]) == 0.4
    assert float(_q(f, "min", "c_k_s").iloc[0]) == 0.4
