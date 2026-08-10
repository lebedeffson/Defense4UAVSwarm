#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from defense4uavswarm.q1_visdrone import EPS, label_detections_protocol, load_detections, load_gt_protocol


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-root", required=True)
    p.add_argument("--detections", required=True)
    p.add_argument("--conf-threshold", type=float, default=0.05)
    p.add_argument("--iou-threshold", type=float, default=0.5)
    p.add_argument("--matching-mode", default="coarse_class")
    p.add_argument("--ignore-policy", default="exclude_ignored")
    p.add_argument("--output-dir", required=True)
    args = p.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    det = load_detections(args.detections)
    gt, ignored = load_gt_protocol(args.dataset_root, sorted(det["sequence_id"].unique()) if not det.empty else None)
    keys = det[["sequence_id", "frame_id"]].drop_duplicates()
    gt = gt.merge(keys, on=["sequence_id", "frame_id"], how="inner")
    ignored = ignored.merge(keys, on=["sequence_id", "frame_id"], how="inner") if not ignored.empty else ignored
    gt = gt.copy()
    gt["area"] = (gt["x2"] - gt["x1"]).clip(lower=1) * (gt["y2"] - gt["y1"]).clip(lower=1)
    q1, q2 = gt["area"].quantile([1 / 3, 2 / 3]).tolist()
    gt["size_bin"] = pd.cut(gt["area"], bins=[-1, q1, q2, float("inf")], labels=["small", "medium", "large"])
    labeled = label_detections_protocol(det, gt, ignored, args.iou_threshold, args.matching_mode, args.ignore_policy, args.conf_threshold)
    total_fp = int((~labeled["eval_is_tp"].astype(bool)).sum()) if len(labeled) else 0
    rows = []
    for size_bin, gt_bin in gt.groupby("size_bin", observed=False):
        keys = set(zip(gt_bin["sequence_id"].astype(str), gt_bin["frame_id"].astype(int), gt_bin["object_id"].astype(str)))
        tp_mask = labeled.apply(
            lambda r: bool(r.get("eval_is_tp", False))
            and (str(r["sequence_id"]), int(r["frame_id"]), str(r.get("matched_gt_id", ""))) in keys,
            axis=1,
        )
        tp = int(tp_mask.sum())
        fn = max(0, len(gt_bin) - tp)
        precision = tp / max(1, tp + total_fp)
        recall = tp / max(1, tp + fn)
        f1 = 2 * precision * recall / max(EPS, precision + recall)
        row = {
            "method": str(size_bin),
            "TP": tp,
            "FP": total_fp,
            "FN": fn,
            "precision": precision,
            "recall": recall,
            "F1": f1,
            "IDF1": f1,
            "false_new_tracks": total_fp,
            "track_breaks": 0,
        }
        row.update(
            {
                "size_bin": str(size_bin),
                "num_gt": len(gt_bin),
                "mean_area": float(gt_bin["area"].mean()) if len(gt_bin) else 0,
                "median_area": float(gt_bin["area"].median()) if len(gt_bin) else 0,
            }
        )
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(out / "size_stratified_metrics.csv", index=False)
    md = ["# Size-Stratified Metrics", "", df[["size_bin", "num_gt", "F1", "recall", "precision"]].to_string(index=False)]
    (out / "size_stratified_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"status=ok output={out / 'size_stratified_metrics.csv'}")


if __name__ == "__main__":
    main()
