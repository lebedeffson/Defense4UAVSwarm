#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--min-visible-corners", type=int, default=4)
    p.add_argument("--clip-to-image", action="store_true")
    p.add_argument("--output-report", required=True)
    args = p.parse_args()

    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    projected = []
    rows = []
    for frame in manifest.get("frames", []):
        for agent in frame.get("agents", []):
            gt3d = agent.get("gt_boxes_3d") or []
            intrinsics = agent.get("intrinsics")
            extrinsics = agent.get("extrinsics")
            status = "ok" if gt3d and intrinsics else "no_projectable_3d_boxes"
            rows.append(
                {
                    "dataset": frame.get("dataset"),
                    "scene_id": frame.get("scene_id"),
                    "frame_id": frame.get("frame_id"),
                    "agent_id": agent.get("agent_id"),
                    "num_gt_3d": len(gt3d),
                    "num_projected_2d": 0,
                    "has_intrinsics": bool(intrinsics),
                    "has_extrinsics": bool(extrinsics),
                    "status": status,
                }
            )
    out = Path(args.output)
    report = Path(args.output_report)
    out.parent.mkdir(parents=True, exist_ok=True)
    report.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"boxes": projected, "status": "no_projectable_3d_boxes" if not projected else "ok"}, indent=2), encoding="utf-8")
    with report.open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["dataset", "scene_id", "frame_id", "agent_id", "num_gt_3d", "num_projected_2d", "has_intrinsics", "has_extrinsics", "status"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"status={'ok' if projected else 'no_projectable_3d_boxes'} frames={len(manifest.get('frames', []))}")


if __name__ == "__main__":
    main()
