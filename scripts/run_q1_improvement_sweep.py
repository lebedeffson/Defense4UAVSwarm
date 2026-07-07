#!/usr/bin/env python
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import pandas as pd
import yaml

from defense4uavswarm.q1_visdrone import (
    add_adaptive_metrics,
    calibration_adaptive_stats,
    evaluate_methods,
    label_detections,
    load_detections,
    load_gt,
    make_chunk_split,
    pareto_flags,
    add_single_camera_features,
    ensure_area_norm,
)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-root", required=True)
    p.add_argument("--detections", required=True)
    p.add_argument("--feature-audit", default="")
    p.add_argument("--split-mode", default="sequence_chunks")
    p.add_argument("--calibration-only", action="store_true")
    p.add_argument("--base-method", default="geometry_dynamic_no_multiagent")
    p.add_argument("--output-dir", required=True)
    args = p.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out.parent / "splits").mkdir(parents=True, exist_ok=True)
    det, gt = load_feature_frame(args)
    det = ensure_area_norm(det)
    chunks = make_chunk_split(gt)
    det = det.merge(chunks[["sequence_id", "frame_id", "chunk_id", "split"]], on=["sequence_id", "frame_id"], how="left")
    gt2 = gt.merge(chunks[["sequence_id", "frame_id", "chunk_id", "split"]], on=["sequence_id", "frame_id"], how="left")
    cal_det = det[det["split"].eq("calibration")].copy()
    cal_gt = gt2[gt2["split"].eq("calibration")].copy()
    write_split_files(chunks, out.parent / "splits")
    stats = calibration_adaptive_stats(cal_det)
    baseline = evaluate_methods(cal_det, cal_gt, [args.base_method]).iloc[0].to_dict()
    rows = []
    configs = list(make_grid(stats))
    for i, cfg in enumerate(configs):
        cfg["config_id"] = f"cfg_{i:04d}"
        row = add_adaptive_metrics(cal_det, cal_gt, cfg, "geometry_dynamic_candidate")
        rows.append({**cfg, **row})
    sweep = pd.DataFrame(rows)
    sweep["baseline_F1"] = baseline["F1"]
    sweep["baseline_false_new_tracks"] = baseline["false_new_tracks"]
    sweep["delta_F1_vs_baseline"] = sweep["F1"] - baseline["F1"]
    sweep["delta_false_new_vs_baseline"] = sweep["false_new_tracks"] - baseline["false_new_tracks"]
    sweep["is_pareto"] = pareto_flags(sweep)
    selected_balanced, selected_safe = select_configs(sweep, baseline)
    sweep["selected_balanced"] = sweep["config_id"].eq(selected_balanced["config_id"])
    sweep["selected_false_new_safe"] = sweep["config_id"].eq(selected_safe["config_id"])
    sweep.to_csv(out / "sweep_summary.csv", index=False)
    sweep[sweep["is_pareto"]].to_csv(out / "pareto_points.csv", index=False)
    payload = clean_yaml({"calibration_stats": stats, "baseline": baseline, "selected_balanced": selected_balanced, "selected_false_new_safe": selected_safe})
    (out / "selected_configs.yaml").write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    print(f"status=ok configs={len(sweep)} pareto={int(sweep['is_pareto'].sum())} output={out}")


def load_feature_frame(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    if args.feature_audit:
        det = pd.read_csv(args.feature_audit)
    else:
        raw = load_detections(args.detections)
        gt0 = load_gt(args.dataset_root, sorted(raw["sequence_id"].unique()) if not raw.empty else None)
        keys = raw[["sequence_id", "frame_id"]].drop_duplicates()
        gt0 = gt0.merge(keys, on=["sequence_id", "frame_id"], how="inner")
        det = add_single_camera_features(label_detections(raw, gt0))
    gt = load_gt(args.dataset_root, sorted(det["sequence_id"].unique()) if not det.empty else None)
    keys = det[["sequence_id", "frame_id"]].drop_duplicates()
    gt = gt.merge(keys, on=["sequence_id", "frame_id"], how="inner")
    return det, gt


def write_split_files(chunks: pd.DataFrame, out: Path) -> None:
    cal = sorted(chunks[chunks["split"].eq("calibration")]["chunk_id"].unique())
    hold = sorted(chunks[chunks["split"].eq("holdout")]["chunk_id"].unique())
    (out / "calibration_chunks.json").write_text(json.dumps(cal, indent=2), encoding="utf-8")
    (out / "holdout_chunks.json").write_text(json.dumps(hold, indent=2), encoding="utf-8")


def default_geometry_config(stats: dict) -> dict:
    return {
        **stats,
        "base_threshold": 0.45,
        "density_gain": 0.0,
        "min_threshold": 0.40,
        "max_threshold": 0.60,
        "small_q_floor": 0.60,
        "medium_q_floor": 0.60,
        "large_q_floor": 0.60,
        "temporal_support_threshold": 0.30,
        "aggregator": "min",
        "alpha": 0.35,
        "beta": 0.35,
        "delta": 0.30,
        "use_recovery": True,
        "recovery_min_age": 2,
        "recovery_conf": 0.10,
    }


def clean_yaml(obj):
    if isinstance(obj, dict):
        return {str(k): clean_yaml(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [clean_yaml(v) for v in obj]
    if hasattr(obj, "item"):
        try:
            return obj.item()
        except Exception:
            pass
    return obj


def make_grid(stats: dict):
    qfloor_sets = [(0.65, 0.60, 0.50), (0.75, 0.65, 0.55), (0.85, 0.70, 0.60)]
    weight_sets = [(0.35, 0.35, 0.30), (0.40, 0.30, 0.30), (0.30, 0.40, 0.20)]
    for base, gain, min_thr, qset, temporal_thr, agg, weights in itertools.product(
        [0.45, 0.50, 0.55],
        [0.00, 0.05, 0.08, 0.10],
        [0.35, 0.40],
        qfloor_sets,
        [0.20, 0.30, 0.40],
        ["min", "geometric_mean"],
        weight_sets,
    ):
        a, b, d = weights
        yield {
            **stats,
            "base_threshold": base,
            "density_gain": gain,
            "min_threshold": min_thr,
            "max_threshold": 0.60,
            "small_q_floor": qset[0],
            "medium_q_floor": qset[1],
            "large_q_floor": qset[2],
            "temporal_support_threshold": temporal_thr,
            "aggregator": agg,
            "alpha": a,
            "beta": b,
            "delta": d,
            "use_recovery": True,
            "recovery_min_age": 2,
            "recovery_conf": 0.10,
        }


def select_configs(sweep: pd.DataFrame, baseline: dict) -> tuple[dict, dict]:
    denom = {c: max(1e-9, float(sweep[c].max() - sweep[c].min())) for c in ["F1", "false_new_tracks", "FP", "FN"]}
    score = (
        0.45 * ((sweep["F1"] - sweep["F1"].min()) / denom["F1"])
        - 0.30 * ((sweep["false_new_tracks"] - sweep["false_new_tracks"].min()) / denom["false_new_tracks"])
        - 0.15 * ((sweep["FP"] - sweep["FP"].min()) / denom["FP"])
        - 0.10 * ((sweep["FN"] - sweep["FN"].min()) / denom["FN"])
    )
    constrained = sweep[(sweep["F1"] >= baseline["F1"] - 0.002) & (sweep["false_new_tracks"] <= baseline["false_new_tracks"])].copy()
    if constrained.empty:
        constrained = sweep.copy()
    balanced = constrained.assign(score=score.loc[constrained.index]).sort_values(["score", "F1"], ascending=False).iloc[0].to_dict()
    f1_oriented = sweep[sweep["false_new_tracks"] <= 1.10 * baseline["false_new_tracks"]].copy()
    if f1_oriented.empty:
        f1_oriented = sweep.copy()
    safe = constrained.sort_values(["false_new_tracks", "F1"], ascending=[True, False]).iloc[0].to_dict()
    if f1_oriented["F1"].max() > safe["F1"] and f1_oriented.sort_values("F1", ascending=False).iloc[0]["false_new_tracks"] <= baseline["false_new_tracks"]:
        safe = f1_oriented.sort_values(["F1", "false_new_tracks"], ascending=[False, True]).iloc[0].to_dict()
    return balanced, safe


if __name__ == "__main__":
    main()
