#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--inputs", nargs="+", required=True)
    p.add_argument("--primary-comparisons", nargs="+", required=True)
    p.add_argument("--metrics", nargs="+", default=["F1", "false_new_tracks"])
    p.add_argument("--bootstrap-samples", type=int, default=1000)
    p.add_argument("--output-dir", required=True)
    args = p.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    frames = [pd.read_csv(x) for x in args.inputs if Path(x).exists()]
    data = pd.concat(frames, ignore_index=True)
    data = data[data.get("available", True).fillna(False).astype(bool)] if "available" in data else data
    tests = []
    boot = []
    effects = []
    rng = np.random.default_rng(2026)
    for comp in args.primary_comparisons:
        a_tracker, a_trust, b_tracker, b_trust = comp.split(":")
        a = select(data, a_tracker, a_trust)
        b = select(data, b_tracker, b_trust)
        merged = a.merge(b, on="sequence_id", suffixes=("_a", "_b"))
        if merged.empty:
            tests.append({"comparison": comp, "status": "unavailable"})
            continue
        for metric in args.metrics:
            av = merged[f"{metric}_a"].astype(float).to_numpy()
            bv = merged[f"{metric}_b"].astype(float).to_numpy()
            delta = bv - av
            ci = bootstrap_ci(delta, args.bootstrap_samples, rng)
            try:
                wil = stats.wilcoxon(delta).pvalue if len(delta) > 0 and np.any(delta != 0) else 1.0
            except Exception:
                wil = np.nan
            try:
                t_p = stats.ttest_rel(bv, av).pvalue
            except Exception:
                t_p = np.nan
            tests.append({"comparison": comp, "metric": metric, "n_sequences": len(delta), "mean_delta": float(delta.mean()), "p_value_wilcoxon": wil, "p_value_ttest": t_p})
            boot.append({"comparison": comp, "metric": metric, "delta_mean": float(delta.mean()), "ci95_low": ci[0], "ci95_high": ci[1]})
            effects.append({"comparison": comp, "metric": metric, "cohens_dz": float(delta.mean() / (delta.std(ddof=1) + 1e-9)) if len(delta) > 1 else 0.0})
    tests_df = pd.DataFrame(tests)
    if "p_value_wilcoxon" in tests_df:
        tests_df["p_value_wilcoxon_holm"] = holm(tests_df["p_value_wilcoxon"])
        tests_df["significant_after_correction"] = tests_df["p_value_wilcoxon_holm"].astype(float) < 0.05
    tests_df.to_csv(out / "statistical_tests.csv", index=False)
    pd.DataFrame(boot).to_csv(out / "bootstrap_ci.csv", index=False)
    pd.DataFrame(effects).to_csv(out / "effect_sizes.csv", index=False)
    make_plots(data, out)
    (out / "statistics_summary.md").write_text(summary_text(tests_df, pd.DataFrame(boot)), encoding="utf-8")
    print(f"status=ok output={out}")


def select(df: pd.DataFrame, tracker: str, trust: str) -> pd.DataFrame:
    d = df[df["tracker"].astype(str).eq(tracker) & df["trust_mode"].astype(str).eq(trust)].copy()
    if "false_new_per_100_frames" not in d and "false_new_tracks_per_100_frames" in d:
        d["false_new_per_100_frames"] = d["false_new_tracks_per_100_frames"]
    cols = ["sequence_id", "F1", "false_new_tracks"]
    if "false_new_per_100_frames" in d:
        cols.append("false_new_per_100_frames")
    return d[cols]


def bootstrap_ci(delta: np.ndarray, n: int, rng) -> tuple[float, float]:
    if len(delta) == 0:
        return (np.nan, np.nan)
    vals = [rng.choice(delta, size=len(delta), replace=True).mean() for _ in range(n)]
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))


def holm(pvals: pd.Series) -> list[float]:
    vals = [(i, float(p)) for i, p in pvals.items() if not pd.isna(p)]
    vals.sort(key=lambda x: x[1])
    out = {i: np.nan for i in pvals.index}
    m = len(vals)
    prev = 0.0
    for rank, (i, p) in enumerate(vals):
        adj = min(1.0, max(prev, (m - rank) * p))
        out[i] = adj
        prev = adj
    return [out[i] for i in pvals.index]


def make_plots(data: pd.DataFrame, out: Path) -> None:
    ok = data[data["tracker"].eq("bytetrack")]
    if ok.empty:
        return
    for metric, name in [("F1", "fig_boxplot_f1_yolov8s.png"), ("false_new_tracks", "fig_boxplot_false_new_yolov8s.png")]:
        plt.figure(figsize=(8, 4))
        labels = []
        vals = []
        for trust, g in ok.groupby("trust_mode"):
            labels.append(trust)
            vals.append(g[metric].astype(float).to_numpy())
        plt.boxplot(vals, labels=labels, vert=True)
        plt.xticks(rotation=25, ha="right")
        plt.ylabel(metric)
        plt.tight_layout()
        plt.savefig(out / name, dpi=160)
        plt.close()
    plt.figure(figsize=(7, 4))
    base = ok[ok["trust_mode"].eq("none")][["sequence_id", "false_new_tracks"]]
    for trust, g in ok[~ok["trust_mode"].eq("none")].groupby("trust_mode"):
        m = base.merge(g[["sequence_id", "false_new_tracks"]], on="sequence_id", suffixes=("_base", "_trust"))
        plt.bar(trust, (m["false_new_tracks_trust"] - m["false_new_tracks_base"]).mean())
    plt.ylabel("mean false_new delta")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(out / "fig_effect_size_false_new_yolov8s.png", dpi=160)
    plt.close()


def summary_text(tests: pd.DataFrame, boot: pd.DataFrame) -> str:
    lines = ["# Statistical Analysis Summary", "", "Sequence-level n is small; interpret p-values together with effect sizes and CI95.", ""]
    if len(tests):
        lines += ["## Tests", "", tests.to_string(index=False), ""]
    if len(boot):
        lines += ["## Bootstrap CI95", "", boot.to_string(index=False), ""]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
