#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from defense4uavswarm.q1_v5.statistics import exact_sign_permutation_pvalue, noninferior, paired_bootstrap_delta, superiority_lower_is_better


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--by-sequence", required=True)
    p.add_argument("--baseline-method", default="tracker_baseline")
    p.add_argument("--method", default="trust_balanced")
    p.add_argument("--primary-metric", default="false_new_tracks_per_100_frames")
    p.add_argument("--n-resamples", type=int, default=10000)
    p.add_argument("--margins", nargs="+", type=float, default=[0.005, 0.010, 0.015])
    p.add_argument("--output-dir", required=True)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if args.dry_run:
        print(f"dry_run=ok output={out}")
        return
    df = pd.read_csv(args.by_sequence)
    base = df[df["method"].eq(args.baseline_method)].copy()
    meth = df[df["method"].eq(args.method)].copy()
    if base.empty or meth.empty:
        raise SystemExit("baseline/method rows missing from by-sequence table")
    merged = base.merge(meth, on="sequence_id", suffixes=("_baseline", "_method"))
    metrics = [args.primary_metric, "F1", "recall", "track_initiation_precision", "mean_confirmation_delay", "peak_concurrent_false_tracks"]
    rows = []
    samples = {}
    rng = np.random.default_rng(args.seed)
    for metric in metrics:
        bcol = f"{metric}_baseline"
        mcol = f"{metric}_method"
        if bcol not in merged or mcol not in merged:
            continue
        b = merged[bcol].to_numpy(float)
        m = merged[mcol].to_numpy(float)
        stat = paired_bootstrap_delta(b, m, args.n_resamples, args.seed)
        deltas = m - b
        row = {"metric": metric, "n_sequences": len(merged), **stat, "relative_change": float(deltas.mean() / max(1e-9, abs(b.mean()))), "exact_permutation_p": exact_sign_permutation_pvalue(deltas)}
        if metric == "F1":
            for margin in args.margins:
                row[f"noninferior_margin_{margin:g}"] = noninferior(stat["ci95_low"], margin)
        if metric == args.primary_metric:
            row["superiority_lower_is_better"] = superiority_lower_is_better(stat["ci95_high"])
        rows.append(row)
        # regenerate and save compact sample for reproducibility diagnostics
        boot = []
        for _ in range(args.n_resamples):
            idx = rng.integers(0, len(deltas), len(deltas))
            boot.append(float(deltas[idx].mean()))
        samples[metric] = np.asarray(boot, dtype=float)
    result = pd.DataFrame(rows)
    result.to_csv(out / "statistical_results.csv", index=False)
    np.savez_compressed(out / "bootstrap_samples.npz", **samples)
    write_md(out / "statistical_summary.md", args, result)
    plot_deltas(merged, args.primary_metric, out / "fig_sequence_deltas.png")
    print(f"status=ok output={out / 'statistical_results.csv'}")


def write_md(path: Path, args: argparse.Namespace, result: pd.DataFrame) -> None:
    lines = ["# Q1 v5 Statistical Summary", "", f"Primary metric: `{args.primary_metric}`.", "Resampling unit: sequence.", ""]
    lines.append(result.to_string(index=False))
    lines += ["", "Limited power: VisDrone validation has seven sequences, so p-values and bootstrap intervals must be interpreted cautiously."]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_deltas(df: pd.DataFrame, metric: str, output: Path) -> None:
    b = f"{metric}_baseline"
    m = f"{metric}_method"
    if b not in df or m not in df:
        return
    vals = df[["sequence_id", b, m]].copy()
    vals["delta"] = vals[m].astype(float) - vals[b].astype(float)
    vals = vals.sort_values("delta")
    plt.figure(figsize=(7, 4), dpi=180)
    plt.bar(vals["sequence_id"], vals["delta"], color="#4b77be")
    plt.axhline(0, color="black", linewidth=0.8)
    plt.xticks(rotation=35, ha="right", fontsize=8)
    plt.ylabel(f"Delta {metric}")
    plt.tight_layout()
    plt.savefig(output)
    plt.close()


if __name__ == "__main__":
    main()
