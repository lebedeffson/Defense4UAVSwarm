#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from defense4uavswarm.q1_visdrone import (
    compute_metrics,
    label_detections_protocol,
    load_detections,
    load_gt_protocol,
)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-root", required=True)
    p.add_argument("--detections", required=True)
    p.add_argument("--conf-thresholds", nargs="+", type=float, required=True)
    p.add_argument("--iou-thresholds", nargs="+", type=float, required=True)
    p.add_argument("--matching-modes", nargs="+", required=True)
    p.add_argument("--ignore-policy", default="exclude_ignored")
    p.add_argument("--output-dir", required=True)
    args = p.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    det = load_detections(args.detections)
    gt, ignored = load_gt_protocol(args.dataset_root, sorted(det["sequence_id"].unique()) if not det.empty else None)
    if not det.empty and not gt.empty:
        keys = det[["sequence_id", "frame_id"]].drop_duplicates()
        gt = gt.merge(keys, on=["sequence_id", "frame_id"], how="inner")
        ignored = ignored.merge(keys, on=["sequence_id", "frame_id"], how="inner") if not ignored.empty else ignored
    rows = []
    detector = str(det["detector"].dropna().iloc[0]) if len(det) and "detector" in det else Path(args.detections).stem
    num_frames = max(1, gt[["sequence_id", "frame_id"]].drop_duplicates().shape[0]) if not gt.empty else 1
    min_conf = min(args.conf_thresholds)
    for iou in args.iou_thresholds:
        for mode in args.matching_modes:
            labeled_all = label_detections_protocol(det, gt, ignored, iou, mode, args.ignore_policy, min_conf)
            for conf in args.conf_thresholds:
                labeled = labeled_all[labeled_all["confidence"].astype(float) >= conf].copy()
                row = compute_metrics(labeled, pd.Series(True, index=labeled.index), len(gt), num_frames, "raw_detector")
                row.update(
                    {
                        "detector": detector,
                        "conf_threshold": conf,
                        "iou_threshold": iou,
                        "matching_mode": mode,
                        "ignore_policy": args.ignore_policy,
                        "num_gt": len(gt),
                        "num_detections": len(labeled),
                    }
                )
                rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(out / "detector_pr_curve.csv", index=False)
    for metric, name in [("F1", "best_thresholds_by_f1.csv"), ("recall", "best_thresholds_by_recall.csv"), ("precision", "best_thresholds_by_precision.csv")]:
        best = df.sort_values(metric, ascending=False).groupby(["iou_threshold", "matching_mode"], as_index=False).head(1)
        best.to_csv(out / name, index=False)
    best_f1 = df.sort_values("F1", ascending=False).iloc[0] if len(df) else {}
    md = ["# Detector PR Audit", "", f"- detector: {detector}", f"- ignore_policy: {args.ignore_policy}"]
    if len(df):
        md.append(f"- best F1: {best_f1['F1']:.6f} at conf={best_f1['conf_threshold']}, iou={best_f1['iou_threshold']}, matching={best_f1['matching_mode']}")
    (out / "detector_pr_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"status=ok rows={len(df)} output={out / 'detector_pr_curve.csv'}")


if __name__ == "__main__":
    main()
