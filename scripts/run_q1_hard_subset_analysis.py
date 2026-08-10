#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml

from defense4uavswarm.q1_visdrone import add_adaptive_metrics, ensure_area_norm, evaluate_methods, load_gt


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-root", required=True)
    p.add_argument("--detections", required=True)
    p.add_argument("--feature-audit", default="")
    p.add_argument("--methods", nargs="+", required=True)
    p.add_argument("--selected-configs", required=True)
    p.add_argument("--output-dir", required=True)
    args = p.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    det = ensure_area_norm(pd.read_csv(args.feature_audit))
    gt = load_gt(args.dataset_root, sorted(det["sequence_id"].unique()))
    keys = det[["sequence_id", "frame_id"]].drop_duplicates()
    gt = gt.merge(keys, on=["sequence_id", "frame_id"], how="inner")
    cfgs = yaml.safe_load(Path(args.selected_configs).read_text(encoding="utf-8")) or {}
    subsets = hard_frame_sets(det)
    rows = []
    for subset_name, frame_keys in subsets.items():
        sub_det = det.merge(frame_keys, on=["sequence_id", "frame_id"], how="inner")
        sub_gt = gt.merge(frame_keys, on=["sequence_id", "frame_id"], how="inner")
        base_methods = [m for m in args.methods if m not in {"geometry_dynamic_adaptive_balanced", "geometry_dynamic_false_new_safe"}]
        summary = evaluate_methods(sub_det, sub_gt, base_methods)
        extra = []
        if "geometry_dynamic_adaptive_balanced" in args.methods:
            extra.append(add_adaptive_metrics(sub_det, sub_gt, cfgs["selected_balanced"], "geometry_dynamic_adaptive_balanced"))
        if "geometry_dynamic_false_new_safe" in args.methods:
            extra.append(add_adaptive_metrics(sub_det, sub_gt, cfgs["selected_false_new_safe"], "geometry_dynamic_false_new_safe"))
        if extra:
            summary = pd.concat([summary, pd.DataFrame(extra)], ignore_index=True)
        summary.insert(0, "subset", subset_name)
        rows.extend(summary.to_dict("records"))
    pd.DataFrame(rows).to_csv(out / "hard_subset_summary.csv", index=False)
    print(f"status=ok output={out / 'hard_subset_summary.csv'}")


def hard_frame_sets(det: pd.DataFrame) -> dict[str, pd.DataFrame]:
    small_thr = det["bbox_area"].quantile(0.25)
    frame = det.groupby(["sequence_id", "frame_id"]).agg(
        num_detections=("det_id", "count"),
        small_share=("bbox_area", lambda x: float((x <= small_thr).mean())),
        low_conf_share=("confidence", lambda x: float((x < 0.3).mean())),
    ).reset_index()
    out = {}
    for name, col in [("high_density", "num_detections"), ("small_object", "small_share"), ("low_confidence", "low_conf_share")]:
        thr = frame[col].quantile(0.75)
        out[name] = frame[frame[col] >= thr][["sequence_id", "frame_id"]]
    return out


if __name__ == "__main__":
    main()
