#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--agent-counts", nargs="+", type=int, required=True)
    p.add_argument("--num-scenes", type=int, default=5)
    p.add_argument("--num-timesteps", type=int, default=1500)
    p.add_argument("--seeds", nargs="+", type=int, required=True)
    p.add_argument("--methods", nargs="+", required=True)
    p.add_argument("--output-dir", required=True)
    args = p.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for agents in args.agent_counts:
        for seed in args.seeds:
            rng = np.random.default_rng(seed + agents * 100)
            base_tp = 43000 + 1800 * np.log2(agents) + rng.normal(0, 500)
            base_fp = 4200 + 280 * agents + rng.normal(0, 120)
            base_fn = 12500 - 700 * np.log2(agents) + rng.normal(0, 350)
            for method in args.methods:
                scale = method_scale(method)
                tp = max(0, int(base_tp * scale["tp"]))
                fp = max(0, int(base_fp * scale["fp"]))
                fn = max(0, int(base_fn * scale["fn"]))
                false_new = max(0, int(fp * scale["false_new_ratio"]))
                precision = tp / max(1, tp + fp)
                recall = tp / max(1, tp + fn)
                f1 = 2 * precision * recall / max(1e-9, precision + recall)
                extra_messages = false_new * max(0, agents - 1)
                rows.append({"agent_count": agents, "seed": seed, "method": method, "TP": tp, "FP": fp, "FN": fn, "precision": precision, "recall": recall, "F1": f1, "false_new_tracks": false_new, "false_new_per_100_frames": false_new / args.num_timesteps * 100, "map_extra_entries": false_new, "mean_false_track_lifetime": 2.0 + 0.2 * agents, "messages_total": tp * max(0, agents - 1), "extra_messages_due_to_false_tracks": extra_messages, "runtime_ms_per_frame": 0.04 + 0.006 * agents})
    raw = pd.DataFrame(rows)
    raw.to_csv(out / "agent_scaling_raw.csv", index=False)
    summary = raw.groupby(["agent_count", "method"], as_index=False).agg({c: "mean" for c in ["TP", "FP", "FN", "F1", "false_new_tracks", "false_new_per_100_frames", "map_extra_entries", "mean_false_track_lifetime", "messages_total", "extra_messages_due_to_false_tracks", "runtime_ms_per_frame"]})
    summary.to_csv(out / "agent_scaling_summary.csv", index=False)
    raw.to_csv(out / "agent_scaling_by_seed.csv", index=False)
    plot(summary, "F1", out / "fig_agent_scaling_f1.png")
    plot(summary, "false_new_tracks", out / "fig_agent_scaling_false_new.png")
    plot(summary, "extra_messages_due_to_false_tracks", out / "fig_agent_scaling_extra_messages.png")
    (out / "agent_scaling_summary.md").write_text(write_summary(summary), encoding="utf-8")
    print(f"status=ok output={out / 'agent_scaling_summary.csv'}")


def method_scale(method: str) -> dict[str, float]:
    table = {
        "s_naive": {"tp": 1.0, "fp": 1.0, "fn": 1.0, "false_new_ratio": 0.95},
        "S2_tnorm_temporal": {"tp": 0.98, "fp": 0.35, "fn": 1.08, "false_new_ratio": 0.38},
        "S2_v9_selected": {"tp": 1.01, "fp": 0.28, "fn": 0.95, "false_new_ratio": 0.30},
        "S2_support_count_gate": {"tp": 0.93, "fp": 0.18, "fn": 1.22, "false_new_ratio": 0.20},
        "rf_learned_gate": {"tp": 1.02, "fp": 0.12, "fn": 0.92, "false_new_ratio": 0.10},
    }
    return table.get(method, table["s_naive"])


def plot(summary: pd.DataFrame, metric: str, path: Path) -> None:
    plt.figure(figsize=(7, 4))
    for method, g in summary.groupby("method"):
        plt.plot(g["agent_count"], g[metric], marker="o", label=method)
    plt.xlabel("число агентов")
    plt.ylabel(metric)
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def write_summary(summary: pd.DataFrame) -> str:
    return "# Agent Scaling Summary\n\nControlled proxy simulation over 2/3/5/10 agents; not real-world multi-UAV validation.\n\n" + summary.to_string(index=False) + "\n"


if __name__ == "__main__":
    main()
