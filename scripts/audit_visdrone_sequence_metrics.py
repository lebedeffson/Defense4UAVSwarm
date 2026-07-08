#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml

from defense4uavswarm.q1_visdrone import (
    add_adaptive_metrics,
    add_single_camera_features,
    evaluate_methods,
    label_detections_protocol,
    load_detections,
    load_gt_protocol,
    train_rf,
)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-root", required=True)
    p.add_argument("--detections", required=True)
    p.add_argument("--methods", nargs="+", required=True)
    p.add_argument("--selected-configs", default="")
    p.add_argument("--matching-mode", default="coarse_class")
    p.add_argument("--ignore-policy", default="exclude_ignored")
    p.add_argument("--iou-threshold", type=float, default=0.5)
    p.add_argument("--detector-conf-threshold", type=float, default=0.05)
    p.add_argument("--output-dir", required=True)
    args = p.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    det_raw = load_detections(args.detections)
    gt, ignored = load_gt_protocol(args.dataset_root, sorted(det_raw["sequence_id"].unique()) if not det_raw.empty else None)
    keys = det_raw[["sequence_id", "frame_id"]].drop_duplicates()
    gt = gt.merge(keys, on=["sequence_id", "frame_id"], how="inner")
    ignored = ignored.merge(keys, on=["sequence_id", "frame_id"], how="inner") if not ignored.empty else ignored
    det = add_single_camera_features(label_detections_protocol(det_raw, gt, ignored, args.iou_threshold, args.matching_mode, args.ignore_policy, args.detector_conf_threshold))
    cfgs = yaml.safe_load(Path(args.selected_configs).read_text(encoding="utf-8")) if args.selected_configs and Path(args.selected_configs).exists() else {}
    rows = []
    for seq in sorted(det["sequence_id"].unique()):
        sub_det = det[det["sequence_id"].eq(seq)].copy()
        sub_gt = gt[gt["sequence_id"].eq(seq)].copy()
        methods = [m for m in args.methods if m not in {"geometry_dynamic_adaptive_balanced", "geometry_dynamic_false_new_safe"}]
        rf_model = None
        params = None
        if "rf_learned_gate" in methods and sub_det["eval_is_tp"].nunique() > 1:
            from defense4uavswarm.q1_visdrone import Q1Params

            rf_model, tau = train_rf(sub_det)
            params = Q1Params(rf_threshold=tau)
        summary = evaluate_methods(sub_det, sub_gt, methods, params=params, rf_model=rf_model)
        extra = []
        if "geometry_dynamic_adaptive_balanced" in args.methods and cfgs and "selected_balanced" in cfgs:
            extra.append(add_adaptive_metrics(sub_det, sub_gt, cfgs["selected_balanced"], "geometry_dynamic_adaptive_balanced"))
        if "geometry_dynamic_false_new_safe" in args.methods and cfgs and "selected_false_new_safe" in cfgs:
            extra.append(add_adaptive_metrics(sub_det, sub_gt, cfgs["selected_false_new_safe"], "geometry_dynamic_false_new_safe"))
        if extra:
            summary = pd.concat([summary, pd.DataFrame(extra)], ignore_index=True)
        summary.insert(0, "mean_confidence", float(sub_det["confidence"].mean()) if len(sub_det) else 0)
        summary.insert(0, "mean_det_per_frame", len(sub_det) / max(1, sub_det[["sequence_id", "frame_id"]].drop_duplicates().shape[0]))
        summary.insert(0, "mean_gt_per_frame", len(sub_gt) / max(1, sub_gt[["sequence_id", "frame_id"]].drop_duplicates().shape[0]))
        summary.insert(0, "num_detections", len(sub_det))
        summary.insert(0, "num_gt", len(sub_gt))
        summary.insert(0, "num_frames", sub_gt[["sequence_id", "frame_id"]].drop_duplicates().shape[0])
        summary.insert(0, "sequence_id", seq)
        rows.extend(summary.to_dict("records"))
    df = pd.DataFrame(rows)
    df.to_csv(out / "sequence_level_metrics.csv", index=False)
    md = ["# Sequence-Level Metrics", "", "Dense or difficult sequences can dominate aggregate results.", ""]
    if len(df):
        md.append(df.sort_values("F1").head(20)[["sequence_id", "method", "num_frames", "num_gt", "num_detections", "F1", "false_new_tracks"]].to_string(index=False))
    (out / "sequence_level_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"status=ok output={out / 'sequence_level_metrics.csv'}")


if __name__ == "__main__":
    main()
