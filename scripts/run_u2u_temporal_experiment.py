#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


SCENARIOS = [
    "S_naive",
    "S2_tnorm_soft",
    "S2_tnorm_temporal",
    "S2_learned_fp_gate",
    "S2_ema_confidence_gate",
    "S2_bayesian_persistence_gate",
    "S2_support_count_gate",
]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--detections", default="")
    p.add_argument("--scenarios", nargs="+", default=[s.lower() for s in SCENARIOS])
    p.add_argument("--agents", type=int, default=3)
    p.add_argument("--split", default="holdout")
    p.add_argument("--selected-params", default="configs/selected_s2_temporal.yaml")
    p.add_argument("--output-dir", required=True)
    args = p.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        write_blocked(out, "missing_manifest", args)
        return
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status", "ok").startswith("blocked") or not manifest.get("frames"):
        write_blocked(out, manifest.get("status", "blocked_empty_manifest"), args)
        return
    if args.detections and not Path(args.detections).exists():
        write_blocked(out, "missing_detections", args)
        return
    write_blocked(out, "blocked_evaluator_not_connected_to_dataset_detections_yet", args)


def write_blocked(out: Path, status: str, args: argparse.Namespace) -> None:
    rows = []
    for scenario in SCENARIOS:
        rows.append(
            {
                "dataset": "U2UData",
                "scenario": scenario,
                "agents": args.agents,
                "TP": "",
                "FP": "",
                "FN": "",
                "precision": "",
                "recall": "",
                "F1": "",
                "IDF1": "",
                "false_new_tracks": "",
                "false_new_tracks_per_100_frames": "",
                "track_breaks": "",
                "runtime_fps": "",
                "requires_training": scenario == "S2_learned_fp_gate",
                "uses_temporal_memory": "temporal" in scenario.lower() or "ema" in scenario.lower() or "bayesian" in scenario.lower(),
                "uses_inter_agent_consistency": scenario not in {"S_naive", "S2_ema_confidence_gate", "S2_bayesian_persistence_gate"},
                "status": status,
            }
        )
    pd.DataFrame(rows).to_csv(out / "main_comparison_table.csv", index=False)
    for name, fields in {
        "agent_count_ablation.csv": ["dataset", "num_agents", "scenario", "TP", "FP", "FN", "F1", "IDF1", "false_new_tracks", "false_new_tracks_per_100_frames", "track_breaks", "support_count_mean", "s_i_mean", "runtime_ms", "status"],
        "sync_delay_sensitivity.csv": ["delay_frames", "scenario", "FP", "FN", "F1", "false_new_tracks", "s_i_mean", "track_breaks", "comment", "status"],
        "pose_noise_sensitivity.csv": ["translation_noise_m", "yaw_noise_deg", "scenario", "FP", "FN", "F1", "false_new_tracks", "s_i_mean", "support_count_mean", "comment", "status"],
        "agent_dropout_sensitivity.csv": ["dropout_prob", "scenario", "available_agents_mean", "FP", "FN", "F1", "false_new_tracks", "track_breaks", "comment", "status"],
        "bootstrap_ci.csv": ["metric", "mean", "ci95_low", "ci95_high", "bootstrap_unit", "status"],
        "runtime_summary.csv": ["scenario", "runtime_fps", "runtime_ms", "status"],
        "rf_holdout_summary.csv": ["model", "threshold", "FP", "FN", "F1", "status"],
    }.items():
        pd.DataFrame([{field: status if field == "status" else "" for field in fields}]).to_csv(out / name, index=False)
    (out / "LIMITATIONS_Q1.md").write_text(
        f"# Q1 Closure Limitations\n\nStatus: `{status}`.\n\n"
        "No U2UData metrics are claimed unless a local U2UData/U2UData+ manifest and detections are available.\n"
        "Do not describe this blocked run as real swarm validation.\n",
        encoding="utf-8",
    )
    print(f"status={status}")


if __name__ == "__main__":
    main()
