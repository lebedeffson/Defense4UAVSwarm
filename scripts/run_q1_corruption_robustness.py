#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from defense4uavswarm.q1_visdrone import Q1Params, add_single_camera_features, evaluate_methods, label_detections, load_detections, load_gt, train_rf


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-root", required=True)
    p.add_argument("--model", default="")
    p.add_argument("--detections", required=True)
    p.add_argument("--corruptions", nargs="+", required=True)
    p.add_argument("--methods", nargs="+", required=True)
    p.add_argument("--rf-train-clean-only", action="store_true")
    p.add_argument("--output-dir", required=True)
    args = p.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    clean_raw = load_detections(args.detections)
    gt = load_gt(args.dataset_root, sorted(clean_raw["sequence_id"].unique()) if not clean_raw.empty else None)
    if not clean_raw.empty and not gt.empty:
        keys = clean_raw[["sequence_id", "frame_id"]].drop_duplicates()
        gt = gt.merge(keys, on=["sequence_id", "frame_id"], how="inner")
    clean = add_single_camera_features(label_detections(clean_raw, gt))
    rf_model, rf_tau = (None, 0.55)
    if "rf_learned_gate" in {m.lower() for m in args.methods}:
        rf_model, rf_tau = train_rf(clean)
    rows = []
    for corruption in ["clean", *args.corruptions]:
        det = apply_detection_corruption(clean, corruption)
        summary = evaluate_methods(det, gt, args.methods, params=Q1Params(rf_threshold=rf_tau), rf_model=rf_model)
        summary.insert(0, "corruption", corruption)
        rows.extend(summary.to_dict("records"))
    pd.DataFrame(rows).to_csv(out / "corruption_summary.csv", index=False)
    (out / "rf_fixed_clean_note.md").write_text(
        "RF was trained on clean VisDrone detections and applied unchanged to corrupted detector streams.\n"
        "Corruption is applied as a deterministic detector-output perturbation proxy, not as a claim of full image-level corruption benchmarking.\n",
        encoding="utf-8",
    )
    print(f"status=ok output={out / 'corruption_summary.csv'}")


def apply_detection_corruption(det: pd.DataFrame, name: str) -> pd.DataFrame:
    d = det.copy()
    rng = np.random.default_rng(abs(hash(name)) % (2**32))
    if name == "clean":
        return d
    if name == "brightness_down":
        d["confidence"] *= 0.90
    elif name == "brightness_up":
        d["confidence"] = (d["confidence"] * 1.05).clip(0, 1)
    elif name == "contrast_low":
        d["confidence"] *= 0.85
    elif name == "gaussian_noise":
        d["confidence"] = (d["confidence"] + rng.normal(-0.05, 0.08, len(d))).clip(0, 1)
    elif name == "motion_blur":
        d["confidence"] *= 0.82
        d["x1"] += rng.normal(0, 3, len(d))
        d["x2"] += rng.normal(0, 3, len(d))
    elif name == "combined_corruption":
        d["confidence"] = (d["confidence"] * 0.75 + rng.normal(-0.03, 0.08, len(d))).clip(0, 1)
        keep = rng.random(len(d)) >= 0.08
        d = d[keep].copy()
    d["bbox"] = d[["x1", "y1", "x2", "y2"]].values.tolist()
    return add_single_camera_features(d)


if __name__ == "__main__":
    main()
