#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml

from defense4uavswarm.datasets.visdrone import VisDroneDataset
from defense4uavswarm.matrix import load_split_sequences


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/pseudo_swarm.yaml")
    p.add_argument("--split-config", default="configs/vid_split.yaml")
    p.add_argument("--split", choices=["calibration", "holdout"], default="calibration")
    p.add_argument("--limit-sequences", type=int, default=None)
    p.add_argument("--materialization-mode", choices=["materialized", "manifest_only"], default="materialized")
    p.add_argument("--output", default=None)
    args = p.parse_args()

    cfg = load_yaml(args.config)
    output = Path(args.output or Path(cfg["output_root"]) / args.split)
    output.mkdir(parents=True, exist_ok=True)
    metadata = output / "metadata"
    metadata.mkdir(parents=True, exist_ok=True)

    ds = VisDroneDataset(cfg["source_root"], "val", subset_hint="VID")
    sequences = load_split_sequences(args.split_config, args.split) or ds.sequence_ids()
    if args.limit_sequences:
        sequences = sequences[: args.limit_sequences]

    rows = []
    transforms = {
        "dataset_type": "pseudo_swarm",
        "split": args.split,
        "source_root": str(Path(cfg["source_root"])),
        "materialization_mode": args.materialization_mode,
        "reference_agent": cfg["projection"]["reference_agent"],
        "use_known_transform": bool(cfg["projection"].get("use_known_transform", True)),
        "agents": {},
    }
    for agent in cfg["agents"]:
        agent_id = agent["id"]
        if args.materialization_mode == "materialized":
            (output / agent_id).mkdir(parents=True, exist_ok=True)
        transforms["agents"][agent_id] = {"transform": agent["transform"]}

    for rec in ds.frames(sequences):
        image = cv2.imread(str(rec.image_path))
        if image is None:
            raise RuntimeError(f"Cannot read image: {rec.image_path}")
        height, width = image.shape[:2]
        for agent in cfg["agents"]:
            agent_id = agent["id"]
            matrix = affine_matrix(agent["transform"], width, height)
            rel_path = Path(agent_id) / rec.sequence_id / f"{rec.frame_id:07d}.jpg"
            out_path = output / rel_path
            if args.materialization_mode == "materialized":
                warped = cv2.warpAffine(image, matrix[:2], (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101)
                warped = apply_brightness(warped, float(agent["transform"].get("brightness_delta", 0.0)))
                warped = apply_noise(warped, float(agent["transform"].get("noise_std", 0.0)), rec.frame_id, agent_id)
                warped = apply_occlusion(warped, float(agent["transform"].get("occlusion_prob", 0.0)), rec.frame_id, agent_id)
                out_path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(out_path), warped)
                materialized_path = str(out_path)
                image_path = materialized_path
                is_materialized = True
            else:
                materialized_path = ""
                image_path = str(rec.image_path)
                is_materialized = False
            matrix_inv = np.linalg.inv(matrix)
            rows.append(
                {
                    "sequence_id": rec.sequence_id,
                    "frame_id": rec.frame_id,
                    "agent_id": agent_id,
                    "image_path": image_path,
                    "source_image_path": str(rec.image_path),
                    "materialized_image_path": materialized_path,
                    "is_materialized": is_materialized,
                    "transform_to_reference": json.dumps(matrix_inv.round(8).tolist()),
                    "transform_from_reference": json.dumps(matrix.round(8).tolist()),
                }
            )

    frame_index = pd.DataFrame(rows)
    frame_index.to_csv(metadata / "frame_index.csv", index=False)
    agent_index = pd.DataFrame(
        [
            {
                "agent_id": agent["id"],
                "num_frames": int((frame_index["agent_id"] == agent["id"]).sum()),
                "is_reference": agent["id"] == cfg["projection"]["reference_agent"],
            }
            for agent in cfg["agents"]
        ]
    )
    agent_index.to_csv(metadata / "agent_index.csv", index=False)
    (metadata / "transforms.json").write_text(json.dumps(transforms, indent=2) + "\n", encoding="utf-8")
    (metadata / "build_info.json").write_text(
        json.dumps(
            {
                "materialization_mode": args.materialization_mode,
                "split": args.split,
                "num_sequences": len(sequences),
                "num_frame_rows": len(frame_index),
                "writes_transformed_images": args.materialization_mode == "materialized",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"output: {output}")
    print(f"num_agents: {len(cfg['agents'])}")
    print(f"num_sequences: {len(sequences)}")
    print(f"num_frame_rows: {len(frame_index)}")
    print(f"materialization_mode: {args.materialization_mode}")
    print(f"frame_index: {metadata / 'frame_index.csv'}")
    print(f"transforms: {metadata / 'transforms.json'}")


def load_yaml(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def affine_matrix(transform: dict, width: int, height: int) -> np.ndarray:
    if transform.get("type") == "identity":
        return np.eye(3, dtype=np.float32)
    scale = float(transform.get("scale", 1.0))
    angle = float(transform.get("rotate_deg", 0.0))
    shift_x = float(transform.get("shift_x", 0.0))
    shift_y = float(transform.get("shift_y", 0.0))
    center = (width / 2.0, height / 2.0)
    mat2 = cv2.getRotationMatrix2D(center, angle, scale).astype(np.float32)
    mat2[0, 2] += shift_x
    mat2[1, 2] += shift_y
    mat = np.eye(3, dtype=np.float32)
    mat[:2] = mat2
    return mat


def apply_brightness(image: np.ndarray, delta: float) -> np.ndarray:
    if delta == 0:
        return image
    out = image.astype(np.float32) + delta * 255.0
    return np.clip(out, 0, 255).astype(np.uint8)


def apply_noise(image: np.ndarray, std: float, frame_id: int, agent_id: str) -> np.ndarray:
    if std <= 0:
        return image
    rng = np.random.default_rng(stable_seed(frame_id, agent_id, "noise"))
    noise = rng.normal(0.0, std * 255.0, image.shape)
    return np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def apply_occlusion(image: np.ndarray, prob: float, frame_id: int, agent_id: str) -> np.ndarray:
    if prob <= 0:
        return image
    rng = np.random.default_rng(stable_seed(frame_id, agent_id, "occ"))
    if rng.random() > prob:
        return image
    out = image.copy()
    h, w = out.shape[:2]
    ow, oh = max(20, w // 8), max(20, h // 8)
    x = int(rng.integers(0, max(1, w - ow)))
    y = int(rng.integers(0, max(1, h - oh)))
    out[y : y + oh, x : x + ow] = 0
    return out


def stable_seed(*parts: object) -> int:
    text = "|".join(str(p) for p in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(text).digest()[:4], "little", signed=False)


if __name__ == "__main__":
    main()
