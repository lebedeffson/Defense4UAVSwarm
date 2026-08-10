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
    p.add_argument("--base-detections", required=True)
    p.add_argument("--scenarios", nargs="+", default=["s2_tnorm_soft", "s2_tnorm_temporal", "s2_learned_fp_gate"])
    p.add_argument("--sync-delay-frames", nargs="+", type=int, default=[0, 1, 2, 3])
    p.add_argument("--pose-noise-translation-m", nargs="+", type=float, default=[0, 0.5, 1.0, 2.0])
    p.add_argument("--pose-noise-yaw-deg", nargs="+", type=float, default=[0, 1, 3, 5])
    p.add_argument("--agent-dropout-prob", nargs="+", type=float, default=[0.0, 0.1, 0.3, 0.5])
    p.add_argument("--output-dir", required=True)
    args = p.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    scenarios = [s for s in args.scenarios if s.lower() != "s2_learned_fp_gate"]
    if len(scenarios) != len(args.scenarios):
        (out / "rf_robustness_note.md").write_text(
            "# RF Robustness Note\n\n"
            "`S2_learned_fp_gate` is omitted from the repeated robustness sweep to avoid retraining a supervised model for every synthetic perturbation point. "
            "Use `outputs/results/v8_custom_swarm/rf_baseline/rf_holdout_summary.csv` for the direct RF comparison.\n",
            encoding="utf-8",
        )

    sync_rows = []
    for delay in args.sync_delay_frames:
        result = evaluate_experiment(args.manifest, args.gt_2d, args.base_detections, scenarios, "holdout", "configs/selected_s2_temporal.yaml", modifiers={"sync_delay_frames": delay})
        for row in result["summary"]:
            sync_rows.append({"delay_frames": delay, **pick(row), "s_i_mean": float(result["features"]["s_i"].mean()) if len(result["features"]) else 0.0, "comment": "synthetic sync-delay sensitivity"})
    pd.DataFrame(sync_rows).to_csv(out / "sync_delay_sensitivity.csv", index=False)

    pose_rows = []
    for trans in args.pose_noise_translation_m:
        for yaw in args.pose_noise_yaw_deg:
            result = evaluate_experiment(args.manifest, args.gt_2d, args.base_detections, scenarios, "holdout", "configs/selected_s2_temporal.yaml", modifiers={"pose_noise_translation_m": trans, "pose_noise_yaw_deg": yaw})
            for row in result["summary"]:
                pose_rows.append({"translation_noise_m": trans, "yaw_noise_deg": yaw, **pick(row), "s_i_mean": float(result["features"]["s_i"].mean()) if len(result["features"]) else 0.0, "support_count_mean": float(result["features"]["support_count"].mean()) if len(result["features"]) else 0.0, "comment": "synthetic pose-noise sensitivity"})
    pd.DataFrame(pose_rows).to_csv(out / "pose_noise_sensitivity.csv", index=False)

    dropout_rows = []
    for prob in args.agent_dropout_prob:
        result = evaluate_experiment(args.manifest, args.gt_2d, args.base_detections, scenarios, "holdout", "configs/selected_s2_temporal.yaml", modifiers={"agent_dropout_prob": prob})
        for row in result["summary"]:
            dropout_rows.append({"dropout_prob": prob, "available_agents_mean": row.get("num_agents", 0), **pick(row), "comment": "synthetic agent-dropout sensitivity"})
    pd.DataFrame(dropout_rows).to_csv(out / "agent_dropout_sensitivity.csv", index=False)
    print(f"status=ok output={out}")


def pick(row: dict) -> dict:
    return {
        "scenario": row["scenario"],
        "FP": row["FP"],
        "FN": row["FN"],
        "F1": row["F1"],
        "false_new_tracks": row["false_new_tracks"],
        "track_breaks": row["track_breaks"],
    }


if __name__ == "__main__":
    main()
