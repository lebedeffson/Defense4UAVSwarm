import pandas as pd

from defense4uavswarm.swarm_tnorm import apply_calibration_noise


def data():
    return pd.DataFrame(
        {
            "sequence_id": ["s1", "s1", "s1"],
            "agent_id": ["a1", "a1", "a2"],
            "x1": [0.0, 1.0, 2.0],
            "x2": [10.0, 11.0, 12.0],
            "y1": [0.0, 1.0, 2.0],
            "y2": [10.0, 11.0, 12.0],
        }
    )


def test_zero_noise_no_change():
    noisy, manifest = apply_calibration_noise(data(), 0, 123, "sequence_static")
    assert noisy[["x1", "x2", "y1", "y2"]].equals(data()[["x1", "x2", "y1", "y2"]])
    assert (manifest[["dx", "dy"]] == 0).all().all()


def test_seed_reproducible_and_static_per_agent():
    n1, m1 = apply_calibration_noise(data(), 5, 123, "sequence_static")
    n2, m2 = apply_calibration_noise(data(), 5, 123, "sequence_static")
    assert n1[["x1", "x2", "y1", "y2"]].equals(n2[["x1", "x2", "y1", "y2"]])
    assert m1.equals(m2)
    assert len(m1.drop_duplicates(["sequence_id", "agent_id"])) == 2
