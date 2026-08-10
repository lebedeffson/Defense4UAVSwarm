#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from defense4uavswarm.v8_sim import evaluate_experiment


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--gt-2d", required=True)
    p.add_argument("--detections", required=True)
    p.add_argument("--split", default="calibration")
    p.add_argument("--holdout-split", default="holdout")
    p.add_argument("--features", nargs="+", default=[])
    p.add_argument("--output-dir", required=True)
    args = p.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    result = evaluate_experiment(args.manifest, args.gt_2d, args.detections, ["s_naive", "s2_tnorm_soft", "s2_tnorm_temporal", "s2_learned_fp_gate"], args.holdout_split, "configs/selected_s2_temporal.yaml")
    frame = pd.DataFrame(result["summary"])
    rf = frame[frame["scenario"].eq("S2_learned_fp_gate")].copy()
    rf.insert(0, "model", "random_forest")
    rf.insert(1, "threshold", 0.55)
    rf.to_csv(out / "rf_holdout_summary.csv", index=False)
    frame.to_csv(out / "rf_comparison_summary.csv", index=False)
    print(f"status=ok output={out / 'rf_holdout_summary.csv'}")


if __name__ == "__main__":
    main()
