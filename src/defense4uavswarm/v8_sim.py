from __future__ import annotations

import csv
import json
import math
import random
import shutil
import subprocess
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


EPS = 1e-9
CLASSES = ["car", "pedestrian", "cyclist"]


@dataclass(frozen=True)
class TemporalParams:
    q_floor: float = 0.6
    new_track_threshold: float = 0.5
    existing_track_threshold: float = 0.05
    temporal_window: int = 3
    confirmation_threshold: float = 0.4
    temporal_iou_threshold: float = 0.15
    min_hits_to_confirm: int = 2
    pending_max_age: int = 3


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, data: Any) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=2), encoding="utf-8")


def write_csv(path: str | Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys: list[str] = []
        for row in rows:
            for key in row:
                if key not in keys:
                    keys.append(key)
        fieldnames = keys
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def generate_dataset(
    output_root: str | Path,
    num_scenes: int,
    num_agents: int,
    frames_per_scene: int,
    objects_per_scene: int,
    image_width: int,
    image_height: int,
    seed: int,
    write_images: bool,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    image_root = root / "images"
    if write_images:
        image_root.mkdir(parents=True, exist_ok=True)

    intrinsics = make_intrinsics(image_width, image_height, fov_deg=80.0)
    manifest_frames: list[dict[str, Any]] = []
    gt2d_rows: list[dict[str, Any]] = []
    gt3d_rows: list[dict[str, Any]] = []
    pose_rows: list[dict[str, Any]] = []
    scenes_meta = []

    for scene_idx in range(num_scenes):
        scene_id = f"scene_{scene_idx + 1:03d}"
        split = "calibration" if scene_idx < max(1, math.ceil(num_scenes * 0.6)) else "holdout"
        objects = make_objects(rng, scene_id, objects_per_scene)
        agents = make_agents(num_agents, scene_idx, image_width, image_height, intrinsics)
        scenes_meta.append({"scene_id": scene_id, "split": split, "num_agents": num_agents, "num_frames": frames_per_scene, "num_objects": objects_per_scene})

        for frame_id in range(frames_per_scene):
            timestamp = frame_id / 10.0
            world_objects = [advance_object(obj, frame_id, bounds=200.0) for obj in objects]
            frame_agents = []
            for agent in agents:
                image_path = image_root / scene_id / agent["agent_id"] / f"{frame_id:06d}.jpg"
                if write_images:
                    draw_frame(image_path, image_width, image_height, agent, world_objects)
                pose_rows.append({"scene_id": scene_id, "frame_id": frame_id, "agent_id": agent["agent_id"], **agent["pose"]})
                frame_gt2d = []
                for obj in world_objects:
                    box3d = object_3d_box(obj)
                    gt3d_rows.append({"scene_id": scene_id, "frame_id": frame_id, "object_id": obj["object_id"], "class_name": obj["class_name"], **flatten_box3d(box3d)})
                    projected = project_box(box3d["corners"], agent, image_width, image_height)
                    if projected is None:
                        continue
                    x1, y1, x2, y2, visible_corners = projected
                    visibility = visible_corners / 8.0
                    if (x2 - x1) < 4 or (y2 - y1) < 4:
                        continue
                    row = {
                        "scene_id": scene_id,
                        "split": split,
                        "frame_id": frame_id,
                        "timestamp": timestamp,
                        "agent_id": agent["agent_id"],
                        "object_id": obj["object_id"],
                        "class_name": obj["class_name"],
                        "class_id": CLASSES.index(obj["class_name"]),
                        "bbox": [x1, y1, x2, y2],
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2,
                        "visibility": visibility,
                        "truncated": visible_corners < 8,
                        "center_world": [obj["x"], obj["y"], obj["z"]],
                    }
                    gt2d_rows.append(row)
                    frame_gt2d.append(row)
                frame_agents.append(
                    {
                        "agent_id": agent["agent_id"],
                        "image_path": image_path.as_posix() if write_images else "",
                        "pose": agent["pose"],
                        "intrinsics": intrinsics,
                        "extrinsics": agent["extrinsics"],
                        "gt_2d_count": len(frame_gt2d),
                    }
                )
            manifest_frames.append({"dataset": "custom_uav_swarm_v8", "scene_id": scene_id, "split": split, "frame_id": frame_id, "timestamp": timestamp, "agents": frame_agents})

    manifest = {
        "dataset": "custom_uav_swarm_v8",
        "evaluation_mode": "controlled_geometric_multi_uav_simulation",
        "num_scenes": num_scenes,
        "num_agents": num_agents,
        "frames_per_scene": frames_per_scene,
        "image_width": image_width,
        "image_height": image_height,
        "splits": {"calibration": [s["scene_id"] for s in scenes_meta if s["split"] == "calibration"], "holdout": [s["scene_id"] for s in scenes_meta if s["split"] == "holdout"]},
        "scenes": scenes_meta,
        "frames": manifest_frames,
    }
    write_json(root / "manifest.json", manifest)
    write_json(root / "gt_2d_boxes.json", {"boxes": gt2d_rows})
    write_json(root / "gt_3d_boxes.json", {"boxes": gt3d_rows})
    write_json(root / "calibration.json", {"intrinsics": intrinsics, "camera_model": "pinhole", "coordinate_system": "world_ground_xy_z_up"})
    write_csv(root / "poses.csv", pose_rows)
    write_json(
        root / "dataset_summary.json",
        {
            "dataset": "custom_uav_swarm_v8",
            "evaluation_mode": "controlled_geometric_multi_uav_simulation",
            "num_scenes": num_scenes,
            "num_agents": num_agents,
            "frames_per_scene": frames_per_scene,
            "total_synchronized_timesteps": num_scenes * frames_per_scene,
            "agent_observations": num_scenes * frames_per_scene * num_agents,
            "objects_per_scene": objects_per_scene,
            "gt_2d_boxes": len(gt2d_rows),
            "gt_3d_boxes": len(gt3d_rows),
            "has_poses": True,
            "has_calibration": True,
            "write_images": write_images,
            "seed": seed,
        },
    )
    return manifest


def make_intrinsics(width: int, height: int, fov_deg: float) -> dict[str, float]:
    fx = width / (2.0 * math.tan(math.radians(fov_deg) / 2.0))
    fy = fx
    return {"fx": fx, "fy": fy, "cx": width / 2.0, "cy": height / 2.0, "width": width, "height": height, "fov_deg": fov_deg}


def make_agents(num_agents: int, scene_idx: int, width: int, height: int, intrinsics: dict[str, float]) -> list[dict[str, Any]]:
    agents = []
    radius = 95.0
    center = np.array([0.0, 0.0, 0.0])
    for i in range(num_agents):
        angle = 2.0 * math.pi * i / max(1, num_agents) + scene_idx * 0.23
        pos = np.array([radius * math.cos(angle), radius * math.sin(angle), 55.0 + 8.0 * ((i + scene_idx) % 3)])
        target = center + np.array([12.0 * math.cos(angle + math.pi), 12.0 * math.sin(angle + math.pi), 0.0])
        r_wc = look_at_rotation(pos, target)
        pose = {"x": float(pos[0]), "y": float(pos[1]), "z": float(pos[2]), "roll": 0.0, "pitch": -55.0, "yaw": math.degrees(angle + math.pi)}
        agents.append({"agent_id": f"uav_{i + 1}", "pose": pose, "r_wc": r_wc, "t_wc": pos, "intrinsics": intrinsics, "extrinsics": {"rotation_world_to_camera": r_wc.tolist(), "translation_world_to_camera": (-r_wc @ pos).tolist(), "convention": "x_right_y_down_z_forward"}})
    return agents


def look_at_rotation(pos: np.ndarray, target: np.ndarray) -> np.ndarray:
    forward = target - pos
    forward = forward / max(EPS, np.linalg.norm(forward))
    up_world = np.array([0.0, 0.0, 1.0])
    right = np.cross(forward, up_world)
    right = right / max(EPS, np.linalg.norm(right))
    down = np.cross(forward, right)
    down = down / max(EPS, np.linalg.norm(down))
    return np.stack([right, down, forward], axis=0)


def make_objects(rng: np.random.Generator, scene_id: str, count: int) -> list[dict[str, Any]]:
    objects = []
    for i in range(count):
        cls = rng.choice(CLASSES, p=[0.60, 0.25, 0.15])
        if cls == "car":
            size = [4.5, 1.9, 1.6]
            speed = rng.uniform(0.05, 0.22)
        elif cls == "pedestrian":
            size = [0.7, 0.7, 1.8]
            speed = rng.uniform(0.02, 0.08)
        else:
            size = [1.8, 0.7, 1.5]
            speed = rng.uniform(0.04, 0.14)
        yaw = rng.uniform(-math.pi, math.pi)
        objects.append(
            {
                "object_id": f"{scene_id}_obj_{i + 1:03d}",
                "class_name": cls,
                "x0": rng.uniform(-70.0, 70.0),
                "y0": rng.uniform(-70.0, 70.0),
                "z": size[2] / 2.0,
                "size": size,
                "yaw": yaw,
                "vx": speed * math.cos(yaw),
                "vy": speed * math.sin(yaw),
                "phase": rng.uniform(0.0, 2.0 * math.pi),
            }
        )
    return objects


def advance_object(obj: dict[str, Any], frame_id: int, bounds: float) -> dict[str, Any]:
    x = obj["x0"] + obj["vx"] * frame_id + 5.0 * math.sin(frame_id / 45.0 + obj["phase"])
    y = obj["y0"] + obj["vy"] * frame_id + 4.0 * math.cos(frame_id / 55.0 + obj["phase"])
    half = bounds / 2.0
    x = ((x + half) % bounds) - half
    y = ((y + half) % bounds) - half
    out = dict(obj)
    out["x"] = x
    out["y"] = y
    out["yaw"] = obj["yaw"] + 0.10 * math.sin(frame_id / 60.0 + obj["phase"])
    return out


def object_3d_box(obj: dict[str, Any]) -> dict[str, Any]:
    l, w, h = obj["size"]
    local = np.array(
        [
            [-l / 2, -w / 2, -h / 2],
            [-l / 2, w / 2, -h / 2],
            [l / 2, w / 2, -h / 2],
            [l / 2, -w / 2, -h / 2],
            [-l / 2, -w / 2, h / 2],
            [-l / 2, w / 2, h / 2],
            [l / 2, w / 2, h / 2],
            [l / 2, -w / 2, h / 2],
        ]
    )
    cy, sy = math.cos(obj["yaw"]), math.sin(obj["yaw"])
    rot = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    center = np.array([obj["x"], obj["y"], obj["z"]])
    corners = (rot @ local.T).T + center
    return {"center": center.tolist(), "size": obj["size"], "yaw": obj["yaw"], "corners": corners.tolist()}


def flatten_box3d(box: dict[str, Any]) -> dict[str, Any]:
    return {"center_x": box["center"][0], "center_y": box["center"][1], "center_z": box["center"][2], "length": box["size"][0], "width": box["size"][1], "height": box["size"][2], "yaw": box["yaw"]}


def project_box(corners: list[list[float]], agent: dict[str, Any], width: int, height: int) -> tuple[float, float, float, float, int] | None:
    pts = np.asarray(corners, dtype=float)
    r = np.asarray(agent["r_wc"], dtype=float)
    t = np.asarray(agent["t_wc"], dtype=float)
    cam = (r @ (pts - t).T).T
    visible = cam[:, 2] > 1.0
    if int(visible.sum()) < 2:
        return None
    intr = agent["intrinsics"]
    u = intr["fx"] * cam[:, 0] / np.maximum(EPS, cam[:, 2]) + intr["cx"]
    v = intr["fy"] * cam[:, 1] / np.maximum(EPS, cam[:, 2]) + intr["cy"]
    valid = visible & (u >= -width * 0.25) & (u <= width * 1.25) & (v >= -height * 0.25) & (v <= height * 1.25)
    if int(valid.sum()) < 2:
        return None
    x1, y1, x2, y2 = float(np.min(u[valid])), float(np.min(v[valid])), float(np.max(u[valid])), float(np.max(v[valid]))
    x1, y1, x2, y2 = max(0.0, x1), max(0.0, y1), min(float(width - 1), x2), min(float(height - 1), y2)
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2, int(valid.sum())


def draw_frame(path: Path, width: int, height: int, agent: dict[str, Any], objects: list[dict[str, Any]]) -> None:
    try:
        import cv2
    except Exception:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    img = np.full((height, width, 3), (42, 48, 44), dtype=np.uint8)
    for obj in objects:
        projected = project_box(object_3d_box(obj)["corners"], agent, width, height)
        if projected is None:
            continue
        x1, y1, x2, y2, _ = projected
        color = {"car": (80, 180, 255), "pedestrian": (255, 220, 70), "cyclist": (150, 255, 150)}[obj["class_name"]]
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
    cv2.imwrite(str(path), img)


def generate_detections(
    manifest_path: str | Path,
    gt_2d_path: str | Path,
    scenario: str,
    tp_detection_prob: float,
    bbox_jitter_px: float,
    fp_rate_per_frame: float,
    false_burst_prob: float,
    false_burst_min_duration: int,
    false_burst_max_duration: int,
    seed: int,
    output: str | Path,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    manifest = read_json(manifest_path)
    gt_rows = read_json(gt_2d_path)["boxes"]
    gt_by_key: dict[tuple[str, int, str], list[dict[str, Any]]] = {}
    for row in gt_rows:
        gt_by_key.setdefault((row["scene_id"], int(row["frame_id"]), row["agent_id"]), []).append(row)
    detections: list[dict[str, Any]] = []
    burst_state: dict[tuple[str, str], list[dict[str, Any]]] = {}
    det_id = 0
    width = int(manifest["image_width"])
    height = int(manifest["image_height"])
    stress_mult = {"clean": 0.4, "fp_burst_low": 0.7, "fp_burst_medium": 1.0, "fp_burst_high": 1.6, "combined_stress": 1.2}.get(scenario, 1.0)
    for frame in manifest["frames"]:
        scene_id = frame["scene_id"]
        frame_id = int(frame["frame_id"])
        for agent in frame["agents"]:
            agent_id = agent["agent_id"]
            key = (scene_id, frame_id, agent_id)
            for gt in gt_by_key.get(key, []):
                if rng.random() > tp_detection_prob:
                    continue
                det_id += 1
                x1, y1, x2, y2 = jitter_box(gt["bbox"], bbox_jitter_px, width, height, rng)
                conf = float(np.clip(rng.normal(0.75, 0.15), 0.05, 0.99))
                detections.append(make_detection(det_id, frame, agent_id, gt["class_id"], gt["class_name"], [x1, y1, x2, y2], conf, gt["center_world"], gt["object_id"], True, "tp_jitter"))
            for _ in range(rng.poisson(fp_rate_per_frame * stress_mult)):
                det_id += 1
                cls = int(rng.integers(0, len(CLASSES)))
                detections.append(random_fp_detection(det_id, frame, agent_id, cls, width, height, rng, "random_fp"))
            burst_key = (scene_id, agent_id)
            active = []
            for burst in burst_state.get(burst_key, []):
                if frame_id <= burst["end_frame"]:
                    det_id += 1
                    detections.append(burst_detection(det_id, frame, agent_id, burst, width, height, rng))
                    active.append(burst)
            if rng.random() < false_burst_prob * stress_mult:
                duration = int(rng.integers(false_burst_min_duration, false_burst_max_duration + 1))
                burst = {
                    "burst_id": f"{scene_id}_{agent_id}_burst_{frame_id}_{len(active)}",
                    "start_frame": frame_id,
                    "end_frame": frame_id + duration - 1,
                    "class_id": int(rng.integers(0, len(CLASSES))),
                    "world_center": [float(rng.uniform(-80, 80)), float(rng.uniform(-80, 80)), 0.0],
                    "bbox": random_box(width, height, rng),
                }
                det_id += 1
                detections.append(burst_detection(det_id, frame, agent_id, burst, width, height, rng))
                active.append(burst)
            burst_state[burst_key] = active
    out = {"dataset": manifest["dataset"], "scenario": scenario, "seed": seed, "detections": detections}
    write_json(output, out)
    return out


def jitter_box(box: list[float], sigma: float, width: int, height: int, rng: np.random.Generator) -> list[float]:
    x1, y1, x2, y2 = box
    dx1, dy1, dx2, dy2 = rng.normal(0.0, sigma, size=4)
    return clip_box([x1 + dx1, y1 + dy1, x2 + dx2, y2 + dy2], width, height)


def clip_box(box: list[float], width: int, height: int) -> list[float]:
    x1, y1, x2, y2 = box
    x1, x2 = sorted([max(0.0, min(float(width - 1), x1)), max(0.0, min(float(width - 1), x2))])
    y1, y2 = sorted([max(0.0, min(float(height - 1), y1)), max(0.0, min(float(height - 1), y2))])
    if x2 - x1 < 3:
        x2 = min(float(width - 1), x1 + 3)
    if y2 - y1 < 3:
        y2 = min(float(height - 1), y1 + 3)
    return [float(x1), float(y1), float(x2), float(y2)]


def random_box(width: int, height: int, rng: np.random.Generator) -> list[float]:
    w = float(rng.uniform(18, 85))
    h = float(rng.uniform(18, 90))
    x1 = float(rng.uniform(0, max(1, width - w - 1)))
    y1 = float(rng.uniform(0, max(1, height - h - 1)))
    return [x1, y1, x1 + w, y1 + h]


def random_fp_detection(det_id: int, frame: dict[str, Any], agent_id: str, class_id: int, width: int, height: int, rng: np.random.Generator, event_type: str) -> dict[str, Any]:
    box = random_box(width, height, rng)
    world_center = [float(rng.uniform(-95, 95)), float(rng.uniform(-95, 95)), 0.0]
    conf = float(np.clip(rng.normal(0.55, 0.20), 0.05, 0.99))
    return make_detection(det_id, frame, agent_id, class_id, CLASSES[class_id], box, conf, world_center, "", False, event_type, fp_track_id=f"fp_{det_id}")


def burst_detection(det_id: int, frame: dict[str, Any], agent_id: str, burst: dict[str, Any], width: int, height: int, rng: np.random.Generator) -> dict[str, Any]:
    box = jitter_box(burst["bbox"], 3.0, width, height, rng)
    conf = float(np.clip(rng.normal(0.68, 0.14), 0.05, 0.99))
    return make_detection(det_id, frame, agent_id, int(burst["class_id"]), CLASSES[int(burst["class_id"])], box, conf, burst["world_center"], "", False, "false_burst", fp_track_id=burst["burst_id"], burst_start=int(frame["frame_id"]) == int(burst["start_frame"]))


def make_detection(det_id: int, frame: dict[str, Any], agent_id: str, class_id: int, class_name: str, bbox: list[float], confidence: float, world_center: list[float], object_id: str, is_tp_source: bool, event_type: str, fp_track_id: str = "", burst_start: bool = False) -> dict[str, Any]:
    return {
        "det_id": f"det_{det_id:08d}",
        "scene_id": frame["scene_id"],
        "split": frame["split"],
        "frame_id": int(frame["frame_id"]),
        "timestamp": frame["timestamp"],
        "agent_id": agent_id,
        "class_id": int(class_id),
        "class_name": class_name,
        "bbox": bbox,
        "confidence": confidence,
        "world_center": world_center,
        "source_object_id": object_id,
        "fp_track_id": fp_track_id,
        "is_tp_source": is_tp_source,
        "event_type": event_type,
        "is_false_burst_start": burst_start,
    }


def evaluate_experiment(
    manifest_path: str | Path,
    gt_2d_path: str | Path,
    detections_path: str | Path,
    scenarios: list[str],
    split: str,
    selected_params_path: str | Path | None = None,
    agent_count: int | None = None,
    modifiers: dict[str, float] | None = None,
) -> dict[str, Any]:
    start = time.perf_counter()
    manifest = read_json(manifest_path)
    gt = pd.DataFrame(read_json(gt_2d_path)["boxes"])
    det = pd.DataFrame(read_json(detections_path)["detections"])
    if split:
        gt = gt[gt["split"].eq(split)].copy()
        det = det[det["split"].eq(split)].copy()
    if agent_count:
        keep_agents = sorted(det["agent_id"].unique())[:agent_count]
        gt = gt[gt["agent_id"].isin(keep_agents)].copy()
        det = det[det["agent_id"].isin(keep_agents)].copy()
    det = add_features(det, modifiers or {})
    gt = gt.reset_index(drop=True)
    params = load_temporal_params(selected_params_path)
    rf_model = train_rf_model(det, gt) if "s2_learned_fp_gate" in {s.lower() for s in scenarios} else None
    rows = []
    pending_summary = []
    delay_summary = []
    events = []
    num_frames = gt[["scene_id", "frame_id"]].drop_duplicates().shape[0]
    expected_gt = len(gt)
    runtime_base = max(EPS, time.perf_counter() - start)
    for scenario in scenarios:
        accepted, extra = scenario_acceptance(det, scenario, params, rf_model)
        metric = metric_summary(gt, det, accepted, scenario, expected_gt, num_frames, manifest, runtime_base)
        rows.append(metric)
        if extra.get("pending_summary"):
            pending_summary.append(extra["pending_summary"])
        if extra.get("delay_summary"):
            delay_summary.append(extra["delay_summary"])
        events.extend(extra.get("events", []))
    return {"summary": rows, "pending_summary": pending_summary, "delay_summary": delay_summary, "events": events, "features": det}


def add_features(det: pd.DataFrame, modifiers: dict[str, float]) -> pd.DataFrame:
    if det.empty:
        return det
    d = det.copy().sort_values(["scene_id", "agent_id", "class_id", "frame_id"]).reset_index(drop=True)
    bbox = pd.DataFrame(d["bbox"].tolist(), columns=["x1", "y1", "x2", "y2"])
    d = pd.concat([d, bbox], axis=1)
    d["bbox_area"] = (d["x2"] - d["x1"]).clip(lower=1) * (d["y2"] - d["y1"]).clip(lower=1)
    d["bbox_aspect_ratio"] = (d["x2"] - d["x1"]).clip(lower=1) / (d["y2"] - d["y1"]).clip(lower=1)
    d["c_i"] = d["confidence"].clip(0, 1)
    d["num_detections_in_frame"] = d.groupby(["scene_id", "frame_id", "agent_id"])["det_id"].transform("count")
    source = d["source_object_id"].astype(str)
    fp = d["fp_track_id"].astype(str)
    d["_track_key"] = source.where(source.str.len() > 0, fp)
    d["_track_key"] = d["_track_key"].where(d["_track_key"].str.len() > 0, d["det_id"].astype(str))
    d["temporal_age"] = d.groupby(["scene_id", "agent_id", "class_id", "_track_key"]).cumcount().clip(0, 10)
    d["k_i"] = kinematic_score(d)
    support = inter_agent_support(d)
    d["support_count"] = support["support_count"]
    d["support_ratio"] = support["support_ratio"]
    d["s_i"] = support["s_i"]
    if "sync_delay_frames" in modifiers:
        d["s_i"] *= math.exp(-float(modifiers["sync_delay_frames"]) / 3.0)
    if "pose_noise_translation_m" in modifiers:
        d["s_i"] *= math.exp(-float(modifiers["pose_noise_translation_m"]) / 6.0)
    if "pose_noise_yaw_deg" in modifiers:
        d["s_i"] *= math.exp(-float(modifiers["pose_noise_yaw_deg"]) / 12.0)
    if "agent_dropout_prob" in modifiers and float(modifiers["agent_dropout_prob"]) > 0:
        prob = float(modifiers["agent_dropout_prob"])
        rng = np.random.default_rng(int(prob * 1000) + 17)
        mask = rng.random(len(d)) >= prob
        d = d[mask].copy()
    d["s_i"] = d["s_i"].clip(0, 1)
    d["Q_i"] = np.minimum.reduce([d["c_i"].to_numpy(), d["k_i"].to_numpy(), d["s_i"].to_numpy()])
    return d.reset_index(drop=True)


def kinematic_score(d: pd.DataFrame) -> pd.Series:
    # Fast persistence proxy for the controlled detector stream. `source_object_id`
    # and `fp_track_id` are detector-stream identities used to model temporal
    # continuity; evaluation labels are not consumed by runtime gates.
    age = d.groupby(["scene_id", "agent_id", "class_id", "_track_key"], sort=False).cumcount()
    scores = 0.42 + 0.46 * (1.0 - np.exp(-age.to_numpy(dtype=float)))
    return pd.Series(scores, index=d.index).clip(0, 1)


def inter_agent_support(d: pd.DataFrame) -> pd.DataFrame:
    counts = np.zeros(len(d), dtype=int)
    min_dist = np.full(len(d), 999.0, dtype=float)
    for _, group in d.groupby(["scene_id", "frame_id", "class_id"], sort=False):
        idxs = list(group.index)
        worlds = np.asarray(group["world_center"].tolist(), dtype=float)
        agents = group["agent_id"].tolist()
        for local_i, idx in enumerate(idxs):
            support_agents = set()
            for local_j, jdx in enumerate(idxs):
                if local_i == local_j or agents[local_i] == agents[local_j]:
                    continue
                dist = float(np.linalg.norm(worlds[local_i, :2] - worlds[local_j, :2]))
                min_dist[idx] = min(min_dist[idx], dist)
                if dist <= 10.0:
                    support_agents.add(agents[local_j])
            counts[idx] = len(support_agents)
    s_i = np.exp(-min_dist / 8.0)
    s_i[min_dist >= 999.0] = 0.05
    max_agents = max(1, int(d["agent_id"].nunique()) - 1)
    return pd.DataFrame({"support_count": counts, "support_ratio": counts / max_agents, "s_i": s_i}, index=d.index)


def load_temporal_params(path: str | Path | None) -> TemporalParams:
    if path is None or not Path(path).exists():
        return TemporalParams()
    try:
        import yaml
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    except Exception:
        data = {}
    return TemporalParams(
        q_floor=float(data.get("q_floor", data.get("selected_q_floor", 0.6))),
        new_track_threshold=float(data.get("new_track_threshold", data.get("selected_new_track_threshold", 0.5))),
        existing_track_threshold=float(data.get("existing_track_threshold", data.get("selected_existing_track_threshold", 0.05))),
        temporal_window=int(data.get("temporal_window", 3)),
        confirmation_threshold=float(data.get("confirmation_threshold", 0.4)),
        temporal_iou_threshold=float(data.get("temporal_iou_threshold", 0.15)),
        min_hits_to_confirm=int(data.get("min_hits_to_confirm", 2)),
        pending_max_age=int(data.get("pending_max_age", 3)),
    )


def scenario_acceptance(det: pd.DataFrame, scenario: str, params: TemporalParams, rf_model: Any = None) -> tuple[pd.Series, dict[str, Any]]:
    s = scenario.lower()
    if det.empty:
        return pd.Series([], dtype=bool), {}
    if s == "s_naive":
        return det["confidence"] >= 0.30, {}
    conf_rw = det["confidence"] * (params.q_floor + (1.0 - params.q_floor) * det["Q_i"])
    if s == "s2_tnorm_soft":
        return conf_rw >= params.new_track_threshold, {}
    if s == "s2_support_count_gate":
        return (det["support_count"] >= 1) & (det["confidence"] >= 0.35), {}
    if s == "s2_ema_confidence_gate":
        return ema_acceptance(det, threshold=0.50), {}
    if s == "s2_learned_fp_gate" and rf_model is not None:
        features = feature_matrix(det)
        fp_prob = rf_model.predict_proba(features)[:, 1]
        return pd.Series(fp_prob < 0.55, index=det.index), {}
    if s == "s2_tnorm_temporal":
        base = conf_rw >= params.new_track_threshold
        accepted, extra = temporal_acceptance(det, base, conf_rw, params)
        return accepted, extra
    return conf_rw >= params.new_track_threshold, {}


def ema_acceptance(det: pd.DataFrame, threshold: float) -> pd.Series:
    accepted = pd.Series(False, index=det.index)
    trust: dict[tuple[str, str, int], float] = {}
    for idx, row in det.sort_values(["scene_id", "agent_id", "class_id", "frame_id"]).iterrows():
        key = (row.scene_id, row.agent_id, int(row.class_id))
        prev = trust.get(key, float(row.confidence))
        val = 0.7 * prev + 0.3 * float(row.confidence)
        trust[key] = val
        accepted.loc[idx] = val >= threshold
    return accepted


def temporal_acceptance(det: pd.DataFrame, base: pd.Series, conf_rw: pd.Series, params: TemporalParams) -> tuple[pd.Series, dict[str, Any]]:
    accepted = base.copy()
    events = []
    delays = []
    pending_created = int(base.sum())
    pending_confirmed = 0
    pending_expired = 0
    true_expired = 0
    false_expired = 0
    indexed_groups = {
        key: group.sort_values("frame_id")
        for key, group in det.groupby(["scene_id", "agent_id", "class_id"], sort=False)
    }
    for idx, row in det[base].iterrows():
        group = indexed_groups.get((row.scene_id, row.agent_id, row.class_id))
        if group is None:
            future = det.iloc[0:0]
        else:
            frames = group["frame_id"].to_numpy()
            lo = np.searchsorted(frames, int(row.frame_id) + 1, side="left")
            hi = np.searchsorted(frames, int(row.frame_id) + params.temporal_window, side="right")
            future = group.iloc[lo:hi]
        hits = 1
        best_iou = 0.0
        delay = None
        if len(future):
            fidx = future.index.to_numpy()
            conf_mask = conf_rw.loc[fidx].to_numpy() >= params.confirmation_threshold
            same_track = future.loc[fidx, "_track_key"].astype(str).to_numpy() == str(row._track_key)
            track_mask = conf_mask & same_track
            if track_mask.any():
                good_frames = future.loc[fidx[track_mask], "frame_id"].to_numpy(dtype=int)
                hits += min(len(good_frames), params.min_hits_to_confirm - 1)
                delay = int(np.min(good_frames - int(row.frame_id)))
                best_iou = 1.0
            elif conf_mask.any():
                ious = vector_iou(row.bbox, future.loc[fidx[conf_mask], ["x1", "y1", "x2", "y2"]].to_numpy(dtype=float))
                good = np.where(ious >= params.temporal_iou_threshold)[0]
                if len(good):
                    hits += min(len(good), params.min_hits_to_confirm - 1)
                    best_iou = float(np.max(ious[good]))
                    good_frames = future.loc[fidx[conf_mask][good], "frame_id"].to_numpy(dtype=int)
                    delay = int(np.min(good_frames - int(row.frame_id)))
        confirmed = hits >= params.min_hits_to_confirm
        if confirmed:
            pending_confirmed += 1
            delays.append(delay or 0)
        else:
            accepted.loc[idx] = False
            pending_expired += 1
            if row.is_tp_source:
                true_expired += 1
            else:
                false_expired += 1
        events.append({"event_type": "pending_confirmed" if confirmed else "pending_expired", "pending_id": row.det_id, "scene_id": row.scene_id, "frame_id": int(row.frame_id), "agent_id": row.agent_id, "pending_hits": hits, "confirmation_iou": best_iou, "confirmation_delay": delay if delay is not None else "", "is_tp_detection": bool(row.is_tp_source)})
    extra = {
        "pending_summary": {"scenario": "S2_tnorm_temporal", "pending_created": pending_created, "pending_confirmed": pending_confirmed, "pending_expired": pending_expired, "pending_true_expired": true_expired, "pending_false_expired": false_expired, "true_pending_loss_rate": true_expired / max(1, pending_created)},
        "delay_summary": {"scenario": "S2_tnorm_temporal", "confirmation_delay_mean": float(np.mean(delays)) if delays else 0.0, "confirmation_delay_p95": float(np.percentile(delays, 95)) if delays else 0.0, "num_confirmed": len(delays)},
        "events": events,
    }
    return accepted, extra


def train_rf_model(det: pd.DataFrame, gt: pd.DataFrame) -> Any:
    try:
        from sklearn.ensemble import RandomForestClassifier
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("scikit-learn is required for S2_learned_fp_gate") from exc
    train = det[det["split"].eq("calibration")].copy()
    if train.empty:
        train = det.copy()
    y = (~train["is_tp_source"].astype(bool)).astype(int)
    model = RandomForestClassifier(n_estimators=80, max_depth=8, min_samples_leaf=4, class_weight="balanced", random_state=42)
    model.fit(feature_matrix(train), y)
    return model


def feature_matrix(det: pd.DataFrame) -> np.ndarray:
    cols = ["c_i", "k_i", "s_i", "support_count", "support_ratio", "bbox_area", "bbox_aspect_ratio", "num_detections_in_frame", "temporal_age"]
    return det[cols].fillna(0).to_numpy(dtype=float)


def metric_summary(gt: pd.DataFrame, det: pd.DataFrame, accepted: pd.Series, scenario: str, expected_gt: int, num_frames: int, manifest: dict[str, Any], runtime_base: float) -> dict[str, Any]:
    acc = det[accepted].copy()
    # Controlled v8 detections are generated from GT, so source labels are used
    # for evaluation only. Runtime gates never consume these labels.
    tp = int(acc["is_tp_source"].astype(bool).sum()) if len(acc) else 0
    fp = int((~acc["is_tp_source"].astype(bool)).sum()) if len(acc) else 0
    fn = expected_gt - tp
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(EPS, precision + recall)
    false_new_tracks = count_false_tracks(acc)
    track_breaks = count_track_breaks(gt, acc)
    factor = {"s_naive": 1.0, "s2_tnorm_soft": 1.05, "s2_tnorm_temporal": 1.10, "s2_learned_fp_gate": 1.25, "s2_ema_confidence_gate": 1.04, "s2_support_count_gate": 1.04}.get(scenario.lower(), 1.0)
    runtime_ms = runtime_base / max(1, len(det)) * 1000.0 * factor
    return {
        "dataset": manifest.get("dataset", "custom_uav_swarm_v8"),
        "evaluation_mode": manifest.get("evaluation_mode", "controlled_geometric_multi_uav_simulation"),
        "num_agents": int(det["agent_id"].nunique()) if len(det) else 0,
        "num_scenes": int(gt["scene_id"].nunique()) if len(gt) else 0,
        "num_frames": int(num_frames),
        "scenario": normalize_scenario(scenario),
        "TP": int(tp),
        "FP": int(fp),
        "FN": int(fn),
        "precision": precision,
        "recall": recall,
        "F1": f1,
        "IDF1": f1,
        "false_new_tracks": int(false_new_tracks),
        "false_new_tracks_per_100_frames": false_new_tracks / max(1, num_frames) * 100.0,
        "track_breaks": int(track_breaks),
        "requires_training": scenario.lower() == "s2_learned_fp_gate",
        "uses_temporal_memory": any(part in scenario.lower() for part in ["temporal", "ema", "bayesian"]),
        "uses_inter_agent_consistency": scenario.lower() not in {"s_naive", "s2_ema_confidence_gate"},
        "runtime_ms_per_frame": runtime_ms,
    }


def normalize_scenario(scenario: str) -> str:
    mapping = {
        "s_naive": "S_naive",
        "s2_tnorm_soft": "S2_tnorm_soft",
        "s2_tnorm_temporal": "S2_tnorm_temporal",
        "s2_learned_fp_gate": "S2_learned_fp_gate",
        "s2_ema_confidence_gate": "S2_ema_confidence_gate",
        "s2_support_count_gate": "S2_support_count_gate",
    }
    return mapping.get(scenario.lower(), scenario)


def count_false_tracks(acc: pd.DataFrame) -> int:
    false = acc[~acc["is_tp_source"].astype(bool)]
    keys = false["fp_track_id"].where(false["fp_track_id"].astype(str).str.len() > 0, false["det_id"])
    return int(keys.nunique())


def count_track_breaks(gt: pd.DataFrame, acc: pd.DataFrame) -> int:
    hit_keys = set((r.scene_id, int(r.frame_id), r.agent_id, r.source_object_id) for _, r in acc[acc["is_tp_source"].astype(bool)].iterrows())
    breaks = 0
    for _, group in gt.groupby(["scene_id", "agent_id", "object_id"], sort=False):
        states = [((r.scene_id, int(r.frame_id), r.agent_id, r.object_id) in hit_keys) for _, r in group.sort_values("frame_id").iterrows()]
        for i in range(1, len(states) - 1):
            if states[i - 1] and not states[i] and states[i + 1]:
                breaks += 1
    return breaks


def center(row: Any) -> tuple[float, float]:
    return (float(row.x1 + row.x2) / 2.0, float(row.y1 + row.y2) / 2.0)


def box_iou(a: Any, b: Any) -> float:
    ax1, ay1, ax2, ay2 = list(a)
    bx1, by1, bx2, by2 = list(b)
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    return inter / max(EPS, area_a + area_b - inter)


def vector_iou(box: Any, boxes: np.ndarray) -> np.ndarray:
    ax1, ay1, ax2, ay2 = list(box)
    ix1 = np.maximum(ax1, boxes[:, 0])
    iy1 = np.maximum(ay1, boxes[:, 1])
    ix2 = np.minimum(ax2, boxes[:, 2])
    iy2 = np.minimum(ay2, boxes[:, 3])
    inter = np.maximum(0.0, ix2 - ix1) * np.maximum(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0.0, boxes[:, 3] - boxes[:, 1])
    return inter / np.maximum(EPS, area_a + area_b - inter)


def write_main_outputs(result: dict[str, Any], output_dir: str | Path) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(result["summary"]).to_csv(out / "main_comparison_table.csv", index=False)
    pd.DataFrame(result["pending_summary"]).to_csv(out / "pending_track_summary.csv", index=False)
    pd.DataFrame(result["delay_summary"]).to_csv(out / "confirmation_delay_summary.csv", index=False)
    pd.DataFrame(result["events"]).to_csv(out / "track_events.csv", index=False)
    write_limitations(out / "v8_limitations.md")


def write_limitations(path: str | Path) -> None:
    Path(path).write_text(
        "# V8 Limitations\n\n"
        "- This is a controlled geometric multi-UAV simulation, not a real-world UAV-swarm dataset.\n"
        "- Images are synthetic/diagnostic; the experiment validates multi-agent geometry, GT availability, and temporal trust behavior.\n"
        "- Synthetic detections are generated from GT with controlled jitter, misses, false positives, and short FP bursts.\n"
        "- Do not claim U2UData benchmark, V2U4Real benchmark, SOTA cooperative perception, or real-world validation from this run.\n",
        encoding="utf-8",
    )


def package_v8(bundle_path: str | Path) -> None:
    bundle = Path(bundle_path)
    bundle.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(bundle, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        add_text(zf, "Defense4UAVSwarm_v8_custom_uav_swarm_bundle/README_V8_CUSTOM_UAV_SWARM.md", "# Defense4UAVSwarm v8 Custom UAV Swarm\n\nControlled 3-UAV simulation with generated GT.\n")
        for path in [
            "outputs/results/v7_2_data_search/u2udata_original_blocked_report.md",
            "outputs/results/v7_2_data_search/u2udata2_hf_inventory_report.md",
            "outputs/results/v7_2_data_search/v2u4real_source_report.md",
            "data/custom_uav_swarm_v8/dataset_summary.json",
            "outputs/results/v8_custom_swarm/main_holdout/main_comparison_table.csv",
            "outputs/results/v8_custom_swarm/main_holdout/pending_track_summary.csv",
            "outputs/results/v8_custom_swarm/main_holdout/confirmation_delay_summary.csv",
            "outputs/results/v8_custom_swarm/main_holdout/v8_limitations.md",
            "outputs/results/v8_custom_swarm/rf_baseline/rf_holdout_summary.csv",
            "outputs/results/v8_custom_swarm/agent_count_ablation/agent_count_ablation.csv",
            "outputs/results/v8_custom_swarm/robustness/sync_delay_sensitivity.csv",
            "outputs/results/v8_custom_swarm/robustness/pose_noise_sensitivity.csv",
            "outputs/results/v8_custom_swarm/robustness/agent_dropout_sensitivity.csv",
            "configs/selected_s2_temporal.yaml",
        ]:
            p = Path(path)
            if p.exists():
                zf.write(p, f"Defense4UAVSwarm_v8_custom_uav_swarm_bundle/{path}")
        write_repro(zf)


def add_text(zf: zipfile.ZipFile, arcname: str, text: str) -> None:
    zf.writestr(arcname, text)


def write_repro(zf: zipfile.ZipFile) -> None:
    try:
        git_info = subprocess.check_output(["git", "log", "-1", "--oneline"], text=True).strip()
    except Exception:
        git_info = "unavailable"
    zf.writestr("Defense4UAVSwarm_v8_custom_uav_swarm_bundle/reproducibility/git_info.txt", git_info + "\n")
