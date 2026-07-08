#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml

from defense4uavswarm.q1_visdrone import add_adaptive_metrics, add_single_camera_features, evaluate_methods, label_detections_protocol, load_detections, load_gt_protocol


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-root", required=True)
    p.add_argument("--detections", required=True)
    p.add_argument("--methods", nargs="+", required=True)
    p.add_argument("--iou-thresholds", nargs="+", type=float, required=True)
    p.add_argument("--matching-modes", nargs="+", required=True)
    p.add_argument("--ignore-policy", default="exclude_ignored")
    p.add_argument("--selected-configs", default="")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--full-method-sensitivity", action="store_true")
    args = p.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if not args.full_method_sensitivity:
        write_lightweight_appendix(args, out)
        return
    raw = load_detections(args.detections)
    gt, ignored = load_gt_protocol(args.dataset_root, sorted(raw["sequence_id"].unique()) if not raw.empty else None)
    keys = raw[["sequence_id", "frame_id"]].drop_duplicates()
    gt = gt.merge(keys, on=["sequence_id", "frame_id"], how="inner")
    ignored = ignored.merge(keys, on=["sequence_id", "frame_id"], how="inner") if not ignored.empty else ignored
    cfgs = yaml.safe_load(Path(args.selected_configs).read_text(encoding="utf-8")) if args.selected_configs and Path(args.selected_configs).exists() else {}
    feature_base = add_single_camera_features(raw.copy())
    feature_cols = [
        "det_id",
        "bbox_area",
        "bbox_aspect_ratio",
        "num_detections_in_frame",
        "c_i",
        "tracklet_id",
        "temporal_age",
        "k_i",
        "s_i",
        "Q_i",
        "confidence_new",
        "track_status",
    ]
    feature_base = feature_base[[c for c in feature_cols if c in feature_base]]
    rows = []
    for iou in args.iou_thresholds:
        for mode in args.matching_modes:
            labeled = label_detections_protocol(raw, gt, ignored, iou, mode, args.ignore_policy, 0.05)
            det = labeled.drop(columns=[c for c in feature_base.columns if c != "det_id" and c in labeled], errors="ignore").merge(feature_base, on="det_id", how="left")
            base_methods = [m for m in args.methods if m not in {"geometry_dynamic_adaptive_balanced", "geometry_dynamic_false_new_safe"}]
            summary = evaluate_methods(det, gt, base_methods)
            extra = []
            if "geometry_dynamic_adaptive_balanced" in args.methods and cfgs:
                extra.append(add_adaptive_metrics(det, gt, cfgs["selected_balanced"], "geometry_dynamic_adaptive_balanced"))
            if "geometry_dynamic_false_new_safe" in args.methods and cfgs:
                extra.append(add_adaptive_metrics(det, gt, cfgs["selected_false_new_safe"], "geometry_dynamic_false_new_safe"))
            if extra:
                summary = pd.concat([summary, pd.DataFrame(extra)], ignore_index=True)
            summary.insert(0, "matching_mode", mode)
            summary.insert(0, "iou_threshold", iou)
            rows.extend(summary.to_dict("records"))
    df = pd.DataFrame(rows)
    df.to_csv(out / "sensitivity_table.csv", index=False)
    md = [
        "# Q1 Sensitivity Appendix",
        "",
        "Main protocol should be interpreted with these IoU/matching-mode sensitivities.",
        "RF is marked unavailable here to keep the appendix focused on evaluation protocol sensitivity; RF is evaluated in the corrected main and label-scarcity tables.",
        "",
    ]
    if len(df):
        md.append(df[["iou_threshold", "matching_mode", "method", "F1", "false_new_tracks"]].to_string(index=False))
    (out / "sensitivity_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"status=ok output={out / 'sensitivity_table.csv'}")


def write_lightweight_appendix(args: argparse.Namespace, out: Path) -> None:
    detector = "yolov8s" if "yolov8s" in args.detections.lower() else "yolov8n" if "yolov8n" in args.detections.lower() else "detector"
    rows = []
    pr_path = Path("outputs/results/q1_audit") / f"{detector}_detector_pr" / "detector_pr_curve.csv"
    if pr_path.exists():
        pr = pd.read_csv(pr_path)
        pr = pr[pr["iou_threshold"].isin(args.iou_thresholds) & pr["matching_mode"].isin(args.matching_modes)].copy()
        for _, r in pr.iterrows():
            rows.append(
                {
                    "iou_threshold": r["iou_threshold"],
                    "matching_mode": r["matching_mode"],
                    "method": "raw_detector",
                    "F1": r["F1"],
                    "false_new_tracks": r["false_new_tracks"],
                    "conf_threshold": r["conf_threshold"],
                    "note": "raw detector PR audit",
                }
            )
    main_path = Path("outputs/results/q1_final_corrected") / f"{detector}_main" / "main_comparison_table.csv"
    if main_path.exists():
        main = pd.read_csv(main_path)
        for _, r in main[main["method"].isin(args.methods)].iterrows():
            rows.append(
                {
                    "iou_threshold": 0.5,
                    "matching_mode": "coarse_class",
                    "method": r["method"],
                    "F1": r["F1"],
                    "false_new_tracks": r["false_new_tracks"],
                    "conf_threshold": "selected_on_calibration",
                    "note": "corrected main protocol method table",
                }
            )
    df = pd.DataFrame(rows)
    df.to_csv(out / "sensitivity_table.csv", index=False)
    md = [
        "# Q1 Sensitivity Appendix",
        "",
        "Lightweight appendix mode: IoU/matching sensitivity is taken from the raw detector PR audit; corrected methods are reported at the selected main protocol.",
        "Full method-level 9-grid sensitivity is intentionally not run by default because it is much more expensive and duplicates the detector/evaluation bottleneck audit.",
        "",
    ]
    if len(df):
        md.append(df[["iou_threshold", "matching_mode", "method", "F1", "false_new_tracks", "conf_threshold"]].to_string(index=False))
    (out / "sensitivity_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"status=ok mode=lightweight output={out / 'sensitivity_table.csv'}")


if __name__ == "__main__":
    main()
