from __future__ import annotations

import json
import csv
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None

from .common_multiagent_schema import AgentObservation, MultiAgentFrame, Pose, frame_to_dict


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".ppm"}
LIDAR_EXTS = {".pcd", ".bin", ".npy", ".npz"}
DEPTH_HINTS = {"depth", "dep"}


class U2UDataAdapter:
    """Best-effort adapter for U2UData/OpenCOOD-style multi-agent data.

    U2UData is distributed through the U2U/OpenCOOD ecosystem. Local copies may
    differ in exact directory names, so this adapter scans for common
    scene/agent/frame patterns and exports a conservative manifest. It does not
    invent labels or metrics when required files are missing.
    """

    dataset_name = "U2UData"

    def __init__(self, root: str | Path):
        self.root = Path(root)

    def exists(self) -> bool:
        return self.root.exists()

    def list_scenes(self) -> list[str]:
        if not self.root.exists():
            return []
        grouped = self._drone_scene_dirs()
        if grouped:
            return sorted(grouped)
        candidates = [p for p in self.root.iterdir() if p.is_dir() and not p.name.startswith(".")]
        split_dirs = {"train", "validate", "validation", "val", "test"}
        if len(candidates) == 1 and candidates[0].name.lower() in split_dirs:
            candidates = [p for p in candidates[0].iterdir() if p.is_dir()]
        return sorted(p.name for p in candidates)

    def scene_path(self, scene_id: str) -> Path:
        direct = self.root / scene_id
        if direct.exists():
            return direct
        for split in ["validate", "validation", "val", "test", "train"]:
            candidate = self.root / split / scene_id
            if candidate.exists():
                return candidate
        return direct

    def list_agents(self, scene_id: str) -> list[str]:
        grouped = self._drone_scene_dirs().get(scene_id)
        if grouped:
            return sorted(grouped)
        scene = self.scene_path(scene_id)
        if not scene.exists():
            return []
        agent_dirs = []
        for path in scene.iterdir():
            if path.is_dir() and self._looks_like_agent(path):
                agent_dirs.append(path.name)
        if agent_dirs:
            return sorted(agent_dirs)
        by_agent = self._index_scene(scene_id)
        return sorted(by_agent.keys())

    def list_frame_ids(self, scene_id: str, agent_ids: list[str] | None = None) -> list[int]:
        indexed = self._index_scene(scene_id)
        if agent_ids:
            indexed = {k: v for k, v in indexed.items() if k in set(agent_ids)}
        counts: dict[int, int] = defaultdict(int)
        for frames in indexed.values():
            for frame_id in frames:
                counts[frame_id] += 1
        required = len(indexed) if indexed else 1
        return sorted(frame_id for frame_id, count in counts.items() if count >= required)

    def load_frame(self, scene_id: str, frame_id: int, agent_ids: list[str] | None = None) -> MultiAgentFrame:
        indexed = self._index_scene(scene_id)
        selected_agents = agent_ids or sorted(indexed.keys())
        agents = []
        for agent_id in selected_agents:
            record = indexed.get(agent_id, {}).get(frame_id, {})
            agents.append(self._observation_from_record(agent_id, record))
        return MultiAgentFrame(dataset=self.dataset_name, scene_id=scene_id, timestamp=float(frame_id), frame_id=frame_id, agents=agents)

    def load_agent_observation(self, scene_id: str, frame_id: int, agent_id: str) -> AgentObservation:
        return self.load_frame(scene_id, frame_id, [agent_id]).agents[0]

    def load_gt(self, scene_id: str, frame_id: int) -> list[dict[str, Any]]:
        frame = self.load_frame(scene_id, frame_id)
        gt: list[dict[str, Any]] = []
        for agent in frame.agents:
            gt.extend(agent.gt_boxes_3d or agent.gt_boxes_2d)
        return gt

    def project_to_common_frame(self, detection: dict[str, Any], agent: AgentObservation) -> dict[str, Any]:
        projected = dict(detection)
        projected["projection_status"] = "identity_or_dataset_pose_unavailable"
        projected["agent_pose"] = agent.pose.__dict__
        return projected

    def export_manifest(self, output_root: str | Path, split: str, agents: int, min_frames: int) -> dict[str, Any]:
        out = Path(output_root)
        out.mkdir(parents=True, exist_ok=True)
        scenes = []
        agents_rows = []
        frames_rows = []
        gt_rows = []
        for scene_id in self.list_scenes():
            scene_agents = self.list_agents(scene_id)[:agents]
            if len(scene_agents) < agents:
                continue
            frame_ids = self.list_frame_ids(scene_id, scene_agents)
            if len(frame_ids) < min_frames:
                continue
            scene_frame_ids = frame_ids[:min_frames]
            scenes.append({"dataset": self.dataset_name, "split": split, "scene_id": scene_id, "num_agents": len(scene_agents), "num_frames": len(scene_frame_ids)})
            for agent_id in scene_agents:
                agents_rows.append({"scene_id": scene_id, "agent_id": agent_id})
            for frame_id in scene_frame_ids:
                frame = self.load_frame(scene_id, frame_id, scene_agents)
                frames_rows.append(frame_to_dict(frame))
                gt_rows.append({"scene_id": scene_id, "frame_id": frame_id, "num_gt_2d": sum(len(a.gt_boxes_2d) for a in frame.agents), "num_gt_3d": sum(len(a.gt_boxes_3d) for a in frame.agents)})
        manifest = {"dataset": self.dataset_name, "split": split, "root": self.root.as_posix(), "scenes": scenes, "frames": frames_rows}
        (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        write_csv(out / "scenes.csv", scenes, ["dataset", "split", "scene_id", "num_agents", "num_frames"])
        write_csv(out / "agents.csv", agents_rows, ["scene_id", "agent_id"])
        write_csv(out / "frames.csv", flatten_frames(frames_rows), ["dataset", "scene_id", "frame_id", "timestamp", "agent_id", "image_path", "lidar_path", "depth_path", "num_gt_2d", "num_gt_3d"])
        write_csv(out / "gt_summary.csv", gt_rows, ["scene_id", "frame_id", "num_gt_2d", "num_gt_3d"])
        return manifest

    def report(self) -> dict[str, Any]:
        scenes = self.list_scenes()
        agent_counts = [len(self.list_agents(scene)) for scene in scenes[:10]]
        synchronized = [len(self.list_frame_ids(scene, self.list_agents(scene))) for scene in scenes[:10]]
        files = list(self.root.rglob("*")) if self.root.exists() else []
        suffixes = {p.suffix.lower() for p in files if p.is_file()}
        has_rgb = bool(suffixes & IMAGE_EXTS)
        has_lidar = bool(suffixes & LIDAR_EXTS)
        has_depth = any(any(h in p.name.lower() for h in DEPTH_HINTS) for p in files if p.is_file())
        has_pose_files = any(p.name.lower() == "airsim_rec.txt" for p in files if p.is_file())
        has_labels = bool(suffixes & {".yaml", ".yml", ".json"})
        has_gt = any(name in p.name.lower() for p in files if p.is_file() for name in ["label", "annotation", "gt", "bbox"])
        return {
            "dataset": self.dataset_name,
            "dataset_root": self.root.as_posix(),
            "status": "ok" if self.root.exists() and scenes else "blocked_missing_or_unrecognized_dataset",
            "num_scenes": len(scenes),
            "num_agents_min_sample": min(agent_counts) if agent_counts else 0,
            "num_agents_max_sample": max(agent_counts) if agent_counts else 0,
            "synchronized_frames_max_sample": max(synchronized) if synchronized else 0,
            "modalities_available": sorted(suffixes),
            "classes": [],
            "has_poses": has_pose_files or has_labels,
            "has_3d_boxes": has_gt,
            "has_gt": has_gt,
            "has_rgb": has_rgb,
            "has_lidar": has_lidar,
            "has_depth": has_depth,
            "usable_for_2d_tracking": has_rgb and has_gt,
            "usable_for_3d_tracking": has_lidar and has_gt,
            "notes": "Install/download U2UData locally before running experiments." if not self.root.exists() else "Report is based on conservative filesystem scan.",
        }

    def _index_scene(self, scene_id: str) -> dict[str, dict[int, dict[str, Any]]]:
        grouped = self._drone_scene_dirs().get(scene_id)
        if grouped:
            return self._index_airsim_drone_scene(grouped)
        scene = self.scene_path(scene_id)
        by_agent: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)
        if not scene.exists():
            return {}
        for path in scene.rglob("*"):
            if not path.is_file():
                continue
            agent_id = self._infer_agent_id(scene, path)
            frame_id = self._infer_frame_id(path)
            if agent_id is None or frame_id is None:
                continue
            rec = by_agent[agent_id].setdefault(frame_id, {"metadata": []})
            suffix = path.suffix.lower()
            lower = path.name.lower()
            if suffix in IMAGE_EXTS and not any(h in lower for h in DEPTH_HINTS):
                rec["image_path"] = path.as_posix()
            elif suffix in LIDAR_EXTS:
                rec["lidar_path"] = path.as_posix()
            elif suffix in IMAGE_EXTS and any(h in lower for h in DEPTH_HINTS):
                rec["depth_path"] = path.as_posix()
            elif suffix in {".yaml", ".yml", ".json"}:
                rec["metadata"].append(path.as_posix())
        return dict(by_agent)

    def _drone_scene_dirs(self) -> dict[str, dict[str, Path]]:
        grouped: dict[str, dict[str, Path]] = defaultdict(dict)
        if not self.root.exists():
            return {}
        scan_roots = [self.root]
        if (self.root / "extracted").is_dir():
            scan_roots.append(self.root / "extracted")
        for scan_root in scan_roots:
            for path in scan_root.iterdir():
                if not path.is_dir():
                    continue
                match = re.match(r"(.+)_drone_(\d+)$", path.name)
                if match:
                    scene_id = match.group(1)
                    grouped[scene_id][f"drone_{match.group(2)}"] = path
        return dict(grouped)

    def _index_airsim_drone_scene(self, grouped: dict[str, Path]) -> dict[str, dict[int, dict[str, Any]]]:
        by_agent: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)
        for agent_id, root in grouped.items():
            rec = root / "airsim_rec.txt"
            if not rec.exists():
                continue
            rows = parse_airsim_rec(rec)
            for frame_id, row in enumerate(rows):
                image_name = choose_image(row.get("FumeImageFile", ""))
                by_agent[agent_id][frame_id] = {
                    "image_path": (root / "images" / image_name).as_posix() if image_name else None,
                    "pose": {
                        "x": float(row.get("POS_X", 0.0)),
                        "y": float(row.get("POS_Y", 0.0)),
                        "z": float(row.get("POS_Z", 0.0)),
                        "qw": float(row.get("Q_W", 1.0)),
                        "qx": float(row.get("Q_X", 0.0)),
                        "qy": float(row.get("Q_Y", 0.0)),
                        "qz": float(row.get("Q_Z", 0.0)),
                    },
                    "timestamp": row.get("TimeStamp", frame_id),
                    "metadata": [rec.as_posix()],
                }
        return dict(by_agent)

    def _observation_from_record(self, agent_id: str, record: dict[str, Any]) -> AgentObservation:
        pose = pose_from_record(record) or Pose()
        gt2d: list[dict[str, Any]] = []
        gt3d: list[dict[str, Any]] = []
        intrinsics = None
        extrinsics = None
        for meta_path in record.get("metadata", [])[:3]:
            data = load_structured(Path(meta_path))
            if isinstance(data, dict):
                pose = parse_pose(data) or pose
                intrinsics = data.get("intrinsics") or data.get("camera_intrinsic") or intrinsics
                extrinsics = data.get("extrinsics") or data.get("lidar_pose") or extrinsics
                boxes = data.get("vehicles") or data.get("objects") or data.get("gt_boxes") or []
                if isinstance(boxes, dict):
                    boxes = list(boxes.values())
                if isinstance(boxes, list):
                    gt3d.extend([b for b in boxes if isinstance(b, dict)])
        return AgentObservation(
            agent_id=agent_id,
            sensor_id=agent_id,
            image_path=record.get("image_path"),
            lidar_path=record.get("lidar_path"),
            depth_path=record.get("depth_path"),
            pose=pose,
            intrinsics=intrinsics,
            extrinsics=extrinsics,
            gt_boxes_2d=gt2d,
            gt_boxes_3d=gt3d,
        )

    def _looks_like_agent(self, path: Path) -> bool:
        name = path.name.lower()
        return name.startswith(("uav", "agent", "cav", "vehicle")) or name.isdigit()

    def _infer_agent_id(self, scene: Path, path: Path) -> str | None:
        rel = path.relative_to(scene).parts
        for part in rel[:-1]:
            low = part.lower()
            if low.startswith(("uav", "agent", "cav", "vehicle")) or part.isdigit():
                return part
        return rel[0] if len(rel) > 1 else None

    def _infer_frame_id(self, path: Path) -> int | None:
        digits = "".join(ch if ch.isdigit() else " " for ch in path.stem).split()
        if not digits:
            return None
        return int(digits[-1])


def load_structured(path: Path) -> Any:
    try:
        if path.suffix.lower() == ".json":
            return json.loads(path.read_text(encoding="utf-8"))
        if yaml is not None:
            return yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return None


def parse_pose(data: dict[str, Any]) -> Pose | None:
    pose_data = data.get("pose") or data.get("lidar_pose") or data.get("camera_pose")
    if isinstance(pose_data, dict):
        return Pose(**{k: float(pose_data.get(k, 0.0)) for k in ["x", "y", "z", "roll", "pitch", "yaw"]})
    if isinstance(pose_data, (list, tuple)) and len(pose_data) >= 6:
        return Pose(*(float(v) for v in pose_data[:6]))
    return None


def pose_from_record(record: dict[str, Any]) -> Pose | None:
    pose_data = record.get("pose")
    if not isinstance(pose_data, dict):
        return None
    return Pose(
        x=float(pose_data.get("x", 0.0)),
        y=float(pose_data.get("y", 0.0)),
        z=float(pose_data.get("z", 0.0)),
        roll=0.0,
        pitch=0.0,
        yaw=0.0,
    )


def parse_airsim_rec(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        reader = csv.DictReader(f, delimiter="\t")
        return [dict(row) for row in reader]


def choose_image(field: str) -> str | None:
    names = [part for part in field.split(";") if part]
    for preferred in ["front_center_0", "front_center", "back_center_0"]:
        for name in names:
            if preferred in name:
                return name
    return names[0] if names else None


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def flatten_frames(frames: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for frame in frames:
        for agent in frame.get("agents", []):
            rows.append(
                {
                    "dataset": frame.get("dataset"),
                    "scene_id": frame.get("scene_id"),
                    "frame_id": frame.get("frame_id"),
                    "timestamp": frame.get("timestamp"),
                    "agent_id": agent.get("agent_id"),
                    "image_path": agent.get("image_path"),
                    "lidar_path": agent.get("lidar_path"),
                    "depth_path": agent.get("depth_path"),
                    "num_gt_2d": len(agent.get("gt_boxes_2d") or []),
                    "num_gt_3d": len(agent.get("gt_boxes_3d") or []),
                }
            )
    return rows
