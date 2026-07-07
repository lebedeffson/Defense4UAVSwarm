#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml

from defense4uavswarm.q1_visdrone import (
    Q1Params,
    add_single_camera_features,
    add_adaptive_metrics,
    ensure_area_norm,
    evaluate_methods,
    label_detections,
    load_detections,
    load_gt,
    make_chunk_split,
    method_acceptance,
    save_model,
    select_budget_chunks,
    train_rf,
)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-root", required=True)
    p.add_argument("--detections", required=True)
    p.add_argument("--label-budgets", nargs="+", type=float, required=True)
    p.add_argument("--seeds", nargs="+", type=int, required=True)
    p.add_argument("--split-mode", default="sequence_chunks")
    p.add_argument("--methods", nargs="+", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--chunk-size", type=int, default=100)
    p.add_argument("--feature-audit", default="")
    p.add_argument("--selected-configs", default="")
    args = p.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if args.feature_audit:
        det = ensure_area_norm(pd.read_csv(args.feature_audit))
        gt = load_gt(args.dataset_root, sorted(det["sequence_id"].unique()) if not det.empty else None)
        keys = det[["sequence_id", "frame_id"]].drop_duplicates()
        gt = gt.merge(keys, on=["sequence_id", "frame_id"], how="inner")
    else:
        det_raw = load_detections(args.detections)
        gt = load_gt(args.dataset_root, sorted(det_raw["sequence_id"].unique()) if not det_raw.empty else None)
        if not det_raw.empty and not gt.empty:
            keys = det_raw[["sequence_id", "frame_id"]].drop_duplicates()
            gt = gt.merge(keys, on=["sequence_id", "frame_id"], how="inner")
        det = ensure_area_norm(add_single_camera_features(label_detections(det_raw, gt)))
    chunks = make_chunk_split(gt, args.chunk_size)
    det = det.merge(chunks[["sequence_id", "frame_id", "chunk_id", "split"]], on=["sequence_id", "frame_id"], how="left")
    gt2 = gt.merge(chunks[["sequence_id", "frame_id", "chunk_id", "split"]], on=["sequence_id", "frame_id"], how="left")
    holdout_det = det[det["split"].eq("holdout")].copy()
    holdout_gt = gt2[gt2["split"].eq("holdout")].copy()

    rows = []
    selected_cfgs = load_selected_configs(args.selected_configs)
    base_methods = [m for m in args.methods if m not in {"geometry_dynamic_adaptive_balanced", "geometry_dynamic_false_new_safe"}]
    for budget in args.label_budgets:
        for seed in args.seeds:
            chosen = select_budget_chunks(chunks, budget, seed)
            rf_model = None
            params = Q1Params()
            if budget > 0 and "rf_learned_gate" in {m.lower() for m in base_methods}:
                train_det = det[det["chunk_id"].isin(chosen)].copy()
                if not train_det.empty and train_det["eval_is_tp"].nunique() > 1:
                    rf_model, tau = train_rf(train_det)
                    params = Q1Params(rf_threshold=tau)
                    save_model(out / f"rf_budget_{budget:g}_seed_{seed}.pkl", rf_model, tau)
            summary = evaluate_methods(holdout_det, holdout_gt, base_methods, params=params, rf_model=rf_model)
            extra_rows = []
            if "geometry_dynamic_adaptive_balanced" in args.methods and selected_cfgs.get("selected_balanced"):
                extra_rows.append(add_adaptive_metrics(holdout_det, holdout_gt, selected_cfgs["selected_balanced"], "geometry_dynamic_adaptive_balanced"))
            if "geometry_dynamic_false_new_safe" in args.methods and selected_cfgs.get("selected_false_new_safe"):
                extra_rows.append(add_adaptive_metrics(holdout_det, holdout_gt, selected_cfgs["selected_false_new_safe"], "geometry_dynamic_false_new_safe"))
            if extra_rows:
                summary = pd.concat([summary, pd.DataFrame(extra_rows)], ignore_index=True)
            summary.insert(0, "seed", seed)
            summary.insert(0, "label_budget", budget)
            summary["calibration_chunks_selected"] = len(chosen)
            rows.extend(summary.to_dict("records"))

    raw = pd.DataFrame(rows)
    raw.to_csv(out / "label_budget_raw.csv", index=False)
    metric_cols = ["TP", "FP", "FN", "precision", "recall", "F1", "IDF1", "false_new_tracks", "false_new_tracks_per_100_frames", "track_breaks"]
    mean = raw.groupby(["label_budget", "method"], dropna=False)[metric_cols].agg(["mean", "std"]).reset_index()
    mean.columns = ["_".join(c).rstrip("_") for c in mean.columns.to_flat_index()]
    mean.to_csv(out / "label_budget_mean_std.csv", index=False)
    raw[["label_budget", "seed", "method", "F1", "false_new_tracks", "available"]].to_csv(out / "label_budget_plot.csv", index=False)
    chunks.to_csv(out / "chunk_split.csv", index=False)
    print(f"status=ok rows={len(raw)} output={out / 'label_budget_mean_std.csv'}")


def load_selected_configs(path: str) -> dict:
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        return {}
    return yaml.safe_load(p.read_text(encoding="utf-8")) or {}


if __name__ == "__main__":
    main()
