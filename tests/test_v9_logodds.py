import pandas as pd

from defense4uavswarm.v9_geomdyn import logodds_acceptance


def test_logodds_accepts_supported_candidate():
    det = pd.DataFrame(
        [
            {"scene_id": "s", "agent_id": "a", "class_id": 0, "_track_key": "t", "frame_id": 0, "det_id": "d0", "confidence": 0.8, "world_support_count": 1, "support_count": 1, "temporal_age": 0},
        ]
    )
    accepted, extra = logodds_acceptance(det, pd.Series([0.8]), {"trust": {"selected_min_support": 1}}, scenario="s2_v9_selected")
    assert accepted.iloc[0]
    assert extra["events"][0]["event_type"] == "pending_logodds_confirmed"
