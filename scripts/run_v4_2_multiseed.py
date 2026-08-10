#!/usr/bin/env python
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import pandas as pd


METRICS = ["FP", "FN", "precision", "recall", "F1", "IDF1", "track_breaks", "false_new_tracks", "false_new_tracks_per_100_frames"]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", nargs="+", type=int, required=True)
    p.add_argument("--selected-params", required=True)
    p.add_argument("--output-root", required=True)
    p.add_argument("--summary-output", required=True)
    p.add_argument("--mean-std-output", required=True)
    args = p.parse_args()

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    seed_rows = []
    seq_frames = []
    failures = []
    for seed in args.seeds:
        out = output_root / f"seed_{seed}"
        out.mkdir(parents=True, exist_ok=True)
        cmd = [
            "/home/lebedeffson/Code/venv/bin/python",
            "scripts/run_experiment_matrix.py",
            "--config", "configs/default.yaml",
            "--swarm-config", "configs/pseudo_swarm_stress.yaml",
            "--pseudo-attack-config", "configs/pseudo_attack_v2.yaml",
            "--pseudo-attack-seed", str(seed),
            "--split-config", "configs/vid_split.yaml",
            "--split", "holdout",
            "--tasks", "swarm_vid",
            "--models", "yolov8n",
            "--eps", "0.008",
            "--scenarios", "s_naive", "s2_tnorm_soft",
            "--use-selected-params", args.selected_params,
            "--swarm-root", "data/swarm/pseudo_visdrone_stress_manifest/holdout",
            "--swarm-generation-mode", "on_the_fly",
            "--sequence-batch-size", "1",
            "--min-free-disk-gb", "10",
            "--cleanup-temp",
            "--class-groups", "all",
            "--save-per-sequence-metrics",
            "--save-per-frame-metrics",
            "--save-track-events",
            "--log-file", str(out / "run.log"),
            "--output-dir", str(out),
        ]
        (out / "command.txt").write_text(" ".join(cmd) + "\n", encoding="utf-8")
        with (out / "run.log").open("w", encoding="utf-8") as log:
            proc = subprocess.run(cmd, cwd=Path.cwd(), text=True, stdout=log, stderr=subprocess.STDOUT)
        if proc.returncode != 0 or not (out / "summary_metrics.csv").exists():
            failures.append({"seed": seed, "returncode": proc.returncode})
            break
        summary = pd.read_csv(out / "summary_metrics.csv")
        seed_rows.extend(seed_summary_rows(seed, summary, out))
        seq = pd.read_csv(out / "sequence_metrics.csv")
        seq.insert(0, "seed", seed)
        seq_frames.append(seq)

    if failures:
        fail = output_root.parent / "failed_seed_report.txt"
        pd.DataFrame(failures).to_csv(fail, index=False)
        raise SystemExit(f"seed run failed: {fail}")

    seed_summary = pd.DataFrame(seed_rows)
    seed_summary.to_csv(args.summary_output, index=False)
    pd.concat(seq_frames, ignore_index=True).to_csv(output_root.parent / "sequence_metrics_all_seeds.csv", index=False)
    mean_std(seed_summary).to_csv(args.mean_std_output, index=False)


def best(frame: pd.DataFrame, scenario: str) -> pd.Series:
    return frame[frame["scenario"] == scenario].sort_values(["F1", "IDF1"], ascending=False).iloc[0]


def seed_summary_rows(seed: int, summary: pd.DataFrame, out: Path) -> list[dict]:
    naive = best(summary, "S_naive")
    method = best(summary, "S2_tnorm_soft")
    meta = pd.read_json(out / "metadata.json", typ="series") if (out / "metadata.json").exists() else {}
    rows = []
    for row in [naive, method]:
        payload = {"seed": seed, "scenario": row["scenario"]}
        payload["num_sequences"] = int(meta.get("num_sequences", 0)) if hasattr(meta, "get") else 0
        payload["num_synchronized_frames"] = int(meta.get("num_synchronized_frames", 0)) if hasattr(meta, "get") else 0
        for col in ["TP", "FP", "FN", "precision", "recall", "F1", "IDF1", "track_breaks", "false_new_tracks", "false_new_tracks_per_100_frames"]:
            payload[col] = row.get(col)
        payload["FP_delta_vs_S_naive"] = row["FP"] - naive["FP"]
        payload["FN_delta_vs_S_naive"] = row["FN"] - naive["FN"]
        payload["F1_delta_vs_S_naive"] = row["F1"] - naive["F1"]
        payload["IDF1_delta_vs_S_naive"] = row["IDF1"] - naive["IDF1"]
        payload["track_break_delta_vs_S_naive"] = row["track_breaks"] - naive["track_breaks"]
        payload["false_new_tracks_delta_vs_S_naive"] = row.get("false_new_tracks", 0) - naive.get("false_new_tracks", 0)
        payload["success_flag"] = bool(payload["FP_delta_vs_S_naive"] < 0 and payload["FN_delta_vs_S_naive"] <= max(1, naive["FN"] * 0.01) and payload["F1_delta_vs_S_naive"] >= -0.0005)
        rows.append(payload)
    return rows


def mean_std(seed_summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    naive = seed_summary[seed_summary["scenario"] == "S_naive"].set_index("seed")
    method = seed_summary[seed_summary["scenario"] == "S2_tnorm_soft"].set_index("seed")
    for metric in METRICS:
        delta = method[metric] - naive[metric]
        rows.append(
            {
                "metric": metric,
                "S_naive_mean": naive[metric].mean(),
                "S_naive_std": naive[metric].std(ddof=1),
                "S2_mean": method[metric].mean(),
                "S2_std": method[metric].std(ddof=1),
                "delta_mean": delta.mean(),
                "delta_std": delta.std(ddof=1),
                "delta_relative_percent_mean": 100.0 * delta.mean() / max(1e-9, abs(naive[metric].mean())),
                "num_seeds": len(delta),
            }
        )
    return pd.DataFrame(rows)


if __name__ == "__main__":
    main()
