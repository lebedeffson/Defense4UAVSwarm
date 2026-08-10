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
    load_selected_detector_threshold,
    train_rf,
)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-root", required=True)
    p.add_argument("--detections", required=True)
    p.add_argument("--methods", nargs="+", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--iou-threshold", type=float, default=0.5)
    p.add_argument("--matching-mode", default="coarse_class")
    p.add_argument("--ignore-policy", default="exclude_ignored")
    p.add_argument("--detector-conf-threshold", type=float, default=0.05)
    p.add_argument("--detector-conf-threshold-from", default="")
    p.add_argument("--selected-configs", default="")
    p.add_argument("--evaluation-protocol", default="")
    args = p.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    conf_threshold = args.detector_conf_threshold
    if args.detector_conf_threshold_from:
        conf_threshold = load_selected_detector_threshold(args.detector_conf_threshold_from)
    det = load_detections(args.detections)
    gt, ignored = load_gt_protocol(args.dataset_root, sorted(det["sequence_id"].unique()) if not det.empty else None)
    if not det.empty and not gt.empty:
        keys = det[["sequence_id", "frame_id"]].drop_duplicates()
        gt = gt.merge(keys, on=["sequence_id", "frame_id"], how="inner")
        ignored = ignored.merge(keys, on=["sequence_id", "frame_id"], how="inner") if not ignored.empty else ignored
    det = add_single_camera_features(label_detections_protocol(det, gt, ignored, args.iou_threshold, args.matching_mode, args.ignore_policy, conf_threshold))
    cfgs = yaml.safe_load(Path(args.selected_configs).read_text(encoding="utf-8")) if args.selected_configs and Path(args.selected_configs).exists() else {}
    base_methods = [m for m in args.methods if m not in {"geometry_dynamic_adaptive_balanced", "geometry_dynamic_false_new_safe"}]
    rf_model = None
    params = None
    if "rf_learned_gate" in base_methods and not det.empty and det["eval_is_tp"].nunique() > 1:
        from defense4uavswarm.q1_visdrone import Q1Params

        rf_model, tau = train_rf(det)
        params = Q1Params(rf_threshold=tau)
    summary = evaluate_methods(det, gt, base_methods, params=params, rf_model=rf_model)
    extra = []
    if "geometry_dynamic_adaptive_balanced" in args.methods and cfgs.get("selected_balanced"):
        extra.append(add_adaptive_metrics(det, gt, cfgs["selected_balanced"], "geometry_dynamic_adaptive_balanced"))
    if "geometry_dynamic_false_new_safe" in args.methods and cfgs.get("selected_false_new_safe"):
        extra.append(add_adaptive_metrics(det, gt, cfgs["selected_false_new_safe"], "geometry_dynamic_false_new_safe"))
    if extra:
        summary = pd.concat([summary, pd.DataFrame(extra)], ignore_index=True)
    summary.to_csv(out / "main_comparison_table.csv", index=False)
    det.to_csv(out / "feature_audit.csv", index=False)
    pd.DataFrame(
        [
            {
                "dataset": "VisDrone2019-VID-val",
                "evaluation_mode": "real_detector_single_uav",
                "matching_mode": args.matching_mode,
                "ignore_policy": args.ignore_policy,
                "iou_threshold": args.iou_threshold,
                "detector_conf_threshold": conf_threshold,
                "num_sequences": gt["sequence_id"].nunique() if not gt.empty else 0,
                "num_frames": gt[["sequence_id", "frame_id"]].drop_duplicates().shape[0] if not gt.empty else 0,
                "num_detections": len(det),
                "num_ignored_detections_excluded": int(det.get("inside_ignored_region", pd.Series(dtype=bool)).astype(bool).sum()) if "inside_ignored_region" in det else 0,
            }
        ]
    ).to_csv(out / "metadata.csv", index=False)
    write_claim_safe(out / "main_comparison_claim_safe.md", summary, args.matching_mode, args.ignore_policy, conf_threshold)
    print(f"status=ok rows={len(summary)} output={out / 'main_comparison_table.csv'}")


def write_claim_safe(path: Path, summary: pd.DataFrame, matching_mode: str, ignore_policy: str, conf_threshold: float) -> None:
    lines = [
        "# Corrected Main Comparison Claim-Safe Notes",
        "",
        f"- matching_mode: `{matching_mode}`",
        f"- ignore_policy: `{ignore_policy}`",
        f"- detector_conf_threshold: `{conf_threshold}`",
        "- This is real-detector single-UAV validation, not real multi-UAV validation.",
        "- RF is a supervised reference when labels are available; no-label methods should be compared as deployable baselines.",
        "",
        "```text",
        summary[["method", "F1", "false_new_tracks"]].to_string(index=False),
        "```",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
