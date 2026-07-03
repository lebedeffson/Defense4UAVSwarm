#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--output", default="outputs/results/swarm_dataset_check.txt")
    args = p.parse_args()
    root = Path(args.root)
    report = inspect(root)
    text = "\n".join(f"{k}: {v}" for k, v in report.items()) + "\n"
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print(text, end="")


def inspect(root: Path) -> dict:
    metadata = root / "metadata"
    frame_index_path = metadata / "frame_index.csv"
    agent_index_path = metadata / "agent_index.csv"
    transforms_path = metadata / "transforms.json"
    if not frame_index_path.exists():
        raise FileNotFoundError(frame_index_path)
    if not agent_index_path.exists():
        raise FileNotFoundError(agent_index_path)
    if not transforms_path.exists():
        raise FileNotFoundError(transforms_path)
    frame_index = pd.read_csv(frame_index_path)
    agent_index = pd.read_csv(agent_index_path)
    transforms = json.loads(transforms_path.read_text(encoding="utf-8"))
    agents = sorted(frame_index["agent_id"].unique())
    per_agent = frame_index.groupby("agent_id").size().to_dict()
    sync = frame_index.groupby(["sequence_id", "frame_id"])["agent_id"].nunique()
    expected = len(agents)
    synchronized = int((sync == expected).sum())
    total_sync_groups = int(len(sync))
    missing_images = int((~frame_index["image_path"].map(lambda p: Path(p).exists())).sum())
    return {
        "root": root,
        "num_agents": len(agents),
        "agents": ",".join(agents),
        "num_sequences": int(frame_index["sequence_id"].nunique()),
        "num_frame_rows": int(len(frame_index)),
        "num_frames_per_agent": json.dumps(per_agent, sort_keys=True),
        "synchronized_frames": synchronized,
        "synchronization_groups": total_sync_groups,
        "has_transforms": transforms_path.exists(),
        "reference_agent": transforms.get("reference_agent"),
        "has_frame_index": frame_index_path.exists(),
        "has_agent_index": agent_index_path.exists(),
        "missing_images": missing_images,
        "status": "ok" if synchronized == total_sync_groups and missing_images == 0 else "warning",
    }


if __name__ == "__main__":
    main()
