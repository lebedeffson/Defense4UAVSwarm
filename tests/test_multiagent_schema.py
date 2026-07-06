from defense4uavswarm.datasets.common_multiagent_schema import AgentObservation, MultiAgentFrame, frame_to_dict
from defense4uavswarm.datasets.u2u_adapter import U2UDataAdapter


def test_multiagent_frame_to_dict() -> None:
    frame = MultiAgentFrame(dataset="x", scene_id="s", timestamp=1.0, frame_id=1, agents=[AgentObservation(agent_id="uav_1")])
    data = frame_to_dict(frame)
    assert data["dataset"] == "x"
    assert data["agents"][0]["agent_id"] == "uav_1"


def test_missing_u2u_dataset_reports_blocked(tmp_path) -> None:
    report = U2UDataAdapter(tmp_path / "missing").report()
    assert report["status"] == "blocked_missing_or_unrecognized_dataset"
    assert report["num_scenes"] == 0
