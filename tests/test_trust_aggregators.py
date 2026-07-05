import pandas as pd

from defense4uavswarm.swarm_tnorm import _q


def frame():
    return pd.DataFrame({"c_i": [0.8, 0.2], "k_i": [0.5, 0.9], "s_i": [0.4, 0.7], "x_i": [1.0, 1.0]})


def test_min_product_lukasiewicz():
    f = frame()
    assert _q(f, "min").tolist() == [0.4, 0.2]
    assert _q(f, "product").round(6).tolist() == [0.16, 0.126]
    assert _q(f, "lukasiewicz").between(0, 1).all()


def test_mean_aggregators_in_range_with_zeros():
    f = frame()
    f.loc[0, "k_i"] = 0.0
    for name in ["weighted_mean", "geometric_mean", "harmonic_mean"]:
        q = _q(f, name)
        assert q.between(0, 1).all()
