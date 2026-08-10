#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml

from defense4uavswarm.q1_visdrone import label_detections_protocol, load_detections, load_gt_protocol, make_chunk_split, compute_metrics


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-root", required=True)
    p.add_argument("--detections", required=True)
    p.add_argument("--split-mode", default="sequence_chunks")
    p.add_argument("--candidate-thresholds", nargs="+", type=float, required=True)
    p.add_argument("--matching-mode", default="coarse_class")
    p.add_argument("--iou-threshold", type=float, default=0.5)
    p.add_argument("--ignore-policy", default="exclude_ignored")
    p.add_argument("--objective", choices=["f1", "recall", "precision"], default="f1")
    p.add_argument("--output", required=True)
    args = p.parse_args()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    det = load_detections(args.detections)
    gt, ignored = load_gt_protocol(args.dataset_root, sorted(det["sequence_id"].unique()) if not det.empty else None)
    keys = det[["sequence_id", "frame_id"]].drop_duplicates()
    gt = gt.merge(keys, on=["sequence_id", "frame_id"], how="inner")
    ignored = ignored.merge(keys, on=["sequence_id", "frame_id"], how="inner") if not ignored.empty else ignored
    chunks = make_chunk_split(gt)
    cal_keys = chunks[chunks["split"].eq("calibration")][["sequence_id", "frame_id"]].drop_duplicates()
    cal_gt = gt.merge(cal_keys, on=["sequence_id", "frame_id"], how="inner")
    cal_ignored = ignored.merge(cal_keys, on=["sequence_id", "frame_id"], how="inner") if not ignored.empty else ignored
    cal_det = det.merge(cal_keys, on=["sequence_id", "frame_id"], how="inner")
    rows = []
    num_frames = max(1, cal_gt[["sequence_id", "frame_id"]].drop_duplicates().shape[0])
    for conf in args.candidate_thresholds:
        labeled = label_detections_protocol(cal_det, cal_gt, cal_ignored, args.iou_threshold, args.matching_mode, args.ignore_policy, conf)
        row = compute_metrics(labeled, pd.Series(True, index=labeled.index), len(cal_gt), num_frames, "raw_detector")
        row["conf_threshold"] = conf
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(out.with_suffix(".csv"), index=False)
    best = df.sort_values({"f1": "F1", "recall": "recall", "precision": "precision"}[args.objective], ascending=False).iloc[0].to_dict()
    payload = {
        "selection_scope": "calibration_chunks_only",
        "objective": args.objective,
        "matching_mode": args.matching_mode,
        "ignore_policy": args.ignore_policy,
        "iou_threshold": args.iou_threshold,
        "selected_threshold": float(best["conf_threshold"]),
        "best": {k: (float(v) if isinstance(v, (int, float)) else v) for k, v in best.items()},
    }
    out.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    print(f"status=ok selected_threshold={payload['selected_threshold']} output={out}")


if __name__ == "__main__":
    main()
