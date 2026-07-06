from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Pose:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0


@dataclass
class AgentObservation:
    agent_id: str
    sensor_id: str = ""
    image_path: str | None = None
    lidar_path: str | None = None
    depth_path: str | None = None
    pose: Pose = field(default_factory=Pose)
    intrinsics: dict[str, Any] | None = None
    extrinsics: dict[str, Any] | None = None
    detections: list[dict[str, Any]] = field(default_factory=list)
    gt_boxes_2d: list[dict[str, Any]] = field(default_factory=list)
    gt_boxes_3d: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class MultiAgentFrame:
    dataset: str
    scene_id: str
    timestamp: float
    frame_id: int
    agents: list[AgentObservation] = field(default_factory=list)


def frame_to_dict(frame: MultiAgentFrame) -> dict[str, Any]:
    return asdict(frame)


def as_posix_or_none(path: Path | None) -> str | None:
    return None if path is None else path.as_posix()
