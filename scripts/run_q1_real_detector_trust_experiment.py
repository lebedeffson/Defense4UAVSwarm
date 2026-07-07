#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from defense4uavswarm.q1_visdrone import add_single_camera_features, evaluate_methods, label_detections, load_detections, load_gt


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-root", required=True)
    p.add_argument("--detections", required=True)
    p.add_argument("--methods", nargs="+", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--iou-threshold", type=float, default=0.5)
    args = p.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    det = load_detections(args.detections)
    gt = load_gt(args.dataset_root, sorted(det["sequence_id"].unique()) if not det.empty else None)
    if not det.empty and not gt.empty:
        keys = det[["sequence_id", "frame_id"]].drop_duplicates()
        gt = gt.merge(keys, on=["sequence_id", "frame_id"], how="inner")
    det = add_single_camera_features(label_detections(det, gt, args.iou_threshold))
    summary = evaluate_methods(det, gt, args.methods)
    summary.to_csv(out / "main_comparison_table.csv", index=False)
    det.to_csv(out / "feature_audit.csv", index=False)
    pd.DataFrame([{"dataset": "VisDrone2019-VID-val", "evaluation_mode": "real_detector_single_uav", "num_sequences": gt["sequence_id"].nunique() if not gt.empty else 0, "num_frames": gt[["sequence_id", "frame_id"]].drop_duplicates().shape[0] if not gt.empty else 0, "num_detections": len(det)}]).to_csv(out / "metadata.csv", index=False)
    print(f"status=ok rows={len(summary)} output={out / 'main_comparison_table.csv'}")


if __name__ == "__main__":
    main()
