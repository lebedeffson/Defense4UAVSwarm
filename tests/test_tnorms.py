from defense4uavswarm.filtering.tnorms import t_norm


def test_tnorms_reduce_to_confidence_and_kinematics_without_xai():
    c, k = 0.7, 0.4
    assert t_norm("T_min", c, k, 1.0, 1.0) == min(c, k)
    assert t_norm("T_prod", c, k, 1.0, 1.0) == c * k
    assert t_norm("T_Lukasiewicz", c, k, 1.0, 1.0) == max(0.0, c + k - 1.0)
