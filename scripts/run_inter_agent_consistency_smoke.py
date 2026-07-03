#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from defense4uavswarm.datasets.visdrone import VISDRONE_CLASSES, VisDroneDataset
from defense4uavswarm.swarm import compute_inter_agent_consistency, load_frame_index, transform_bbox


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--swarm-root", default="data/swarm/pseudo_visdrone/smoke")
    p.add_argument("--source-root", default="data/visdrone")
    p.add_argument("--output-dir", default="outputs/results/swarm_v2_smoke_s_i")
    p.add_argument("--max-frames", type=int, default=80)
    p.add_argument("--iou-min", type=float, default=0.3)
    args = p.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    frame_index = load_frame_index(args.swarm_root)
    frame_keys = frame_index[["sequence_id", "frame_id"]].drop_duplicates().head(args.max_frames)
    sequences = sorted(frame_keys["sequence_id"].unique())
    ds = VisDroneDataset(args.source_root, "val", subset_hint="VID")
    gt = ds.all_annotations(sequences)
    gt = gt.merge(frame_keys, on=["sequence_id", "frame_id"])
    detections = build_pseudo_detections(gt, frame_index)
    features, matches = compute_inter_agent_consistency(detections, frame_index, iou_min=args.iou_min)
    features.to_csv(out / "swarm_feature_audit.csv", index=False)
    matches.to_csv(out / "inter_agent_matching_audit.csv", index=False)
    report = {
        "num_detections": len(features),
        "num_matches": len(matches),
        "mean_s_i": float(features["s_i"].mean()) if len(features) else 0.0,
        "share_s_i_low": float((features["s_i"] < args.iou_min).mean()) if len(features) else 0.0,
        "s_i_min": float(features["s_i"].min()) if len(features) else 0.0,
        "s_i_max": float(features["s_i"].max()) if len(features) else 0.0,
    }
    text = "\n".join(f"{k}: {v}" for k, v in report.items()) + "\n"
    (out / "s_i_smoke_report.txt").write_text(text, encoding="utf-8")
    print(text, end="")


def build_pseudo_detections(gt: pd.DataFrame, frame_index: pd.DataFrame) -> pd.DataFrame:
    transforms = frame_index.set_index(["sequence_id", "frame_id", "agent_id"])["transform_from_reference_matrix"].to_dict()
    rows = []
    det_id = 0
    for frame_row in frame_index[["sequence_id", "frame_id", "agent_id"]].drop_duplicates().itertuples(index=False):
        frame_gt = gt[(gt.sequence_id == frame_row.sequence_id) & (gt.frame_id == frame_row.frame_id)]
        matrix = transforms[(frame_row.sequence_id, frame_row.frame_id, frame_row.agent_id)]
        for g in frame_gt.itertuples(index=False):
            box = transform_bbox((g.x1, g.y1, g.x2, g.y2), matrix)
            rows.append(
                {
                    "det_id": det_id,
                    "sequence_id": frame_row.sequence_id,
                    "frame_id": frame_row.frame_id,
                    "agent_id": frame_row.agent_id,
                    "track_id": g.gt_track_id,
                    "class_id": g.class_id,
                    "class_name": VISDRONE_CLASSES.get(int(g.class_id), "unknown"),
                    "confidence": 0.9,
                    "x1": box[0],
                    "y1": box[1],
                    "x2": box[2],
                    "y2": box[3],
                }
            )
            det_id += 1
        if frame_row.agent_id == "agent_2" and int(frame_row.frame_id) % 10 == 0:
            rows.append(
                {
                    "det_id": det_id,
                    "sequence_id": frame_row.sequence_id,
                    "frame_id": frame_row.frame_id,
                    "agent_id": frame_row.agent_id,
                    "track_id": -1,
                    "class_id": 4,
                    "class_name": "car",
                    "confidence": 0.25,
                    "x1": 8.0,
                    "y1": 8.0,
                    "x2": 42.0,
                    "y2": 38.0,
                }
            )
            det_id += 1
    return pd.DataFrame(rows)


if __name__ == "__main__":
    main()
