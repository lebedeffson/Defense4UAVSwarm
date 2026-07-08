#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from defense4uavswarm.q1_visdrone import compute_metrics, label_detections_protocol, load_detections, load_gt_protocol


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-root", required=True)
    p.add_argument("--detections", required=True)
    p.add_argument("--output-dir", required=True)
    args = p.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    det = load_detections(args.detections)
    gt, ignored = load_gt_protocol(args.dataset_root, sorted(det["sequence_id"].unique()) if not det.empty else None)
    keys = det[["sequence_id", "frame_id"]].drop_duplicates()
    gt = gt.merge(keys, on=["sequence_id", "frame_id"], how="inner")
    ignored = ignored.merge(keys, on=["sequence_id", "frame_id"], how="inner") if not ignored.empty else ignored
    rows = []
    num_frames = max(1, gt[["sequence_id", "frame_id"]].drop_duplicates().shape[0])
    for mode in ["class_aware", "class_agnostic", "coarse_class", "vehicle_person_only"]:
        labeled = label_detections_protocol(det, gt, ignored, 0.5, mode, "exclude_ignored", 0.05)
        row = compute_metrics(labeled, pd.Series(True, index=labeled.index), len(gt), num_frames, mode)
        row["matching_mode"] = mode
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(out / "class_mapping_metrics.csv", index=False)
    best = df.sort_values("F1", ascending=False).iloc[0]
    md = ["# Class Mapping Audit", "", f"- best matching mode by F1: {best['matching_mode']} (F1={best['F1']:.6f})."]
    (out / "class_mapping_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"status=ok output={out / 'class_mapping_metrics.csv'}")


if __name__ == "__main__":
    main()
