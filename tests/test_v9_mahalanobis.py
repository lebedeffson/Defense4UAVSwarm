import pandas as pd

from defense4uavswarm.v9_geomdyn import add_mahalanobis_features


def test_mahalanobis_available_after_first_track_hit():
    det = pd.DataFrame(
        [
            {"scene_id": "s", "agent_id": "a", "class_id": 0, "_track_key": "t", "frame_id": 0, "x1": 0, "y1": 0, "x2": 10, "y2": 10, "k_i": 0.5, "temporal_age": 0},
            {"scene_id": "s", "agent_id": "a", "class_id": 0, "_track_key": "t", "frame_id": 1, "x1": 1, "y1": 1, "x2": 11, "y2": 11, "k_i": 0.5, "temporal_age": 1},
        ]
    )
    out = add_mahalanobis_features(det, {})
    assert not out.loc[0, "maha_available"]
    assert out.loc[1, "maha_available"]
    assert 0 <= out.loc[1, "k_maha"] <= 1
