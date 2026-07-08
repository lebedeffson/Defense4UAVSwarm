from scripts.run_q1_agent_scaling_simulation import method_scale


def test_agent_scaling_method_scale() -> None:
    assert method_scale("S2_v9_selected")["fp"] < method_scale("s_naive")["fp"]
