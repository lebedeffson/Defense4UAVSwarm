#!/usr/bin/env python
from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--by-sequence", required=True)
    p.add_argument("--method-a", required=True, help="tracker:trust_mode, for example bytetrack:none")
    p.add_argument("--method-b", required=True, help="tracker:trust_mode, for example bytetrack:geometry_dynamic_no_multiagent")
    p.add_argument("--metric", default="false_new_tracks")
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--target-power", type=float, default=0.80)
    p.add_argument("--bootstrap-samples", type=int, default=1000)
    p.add_argument("--output-dir", required=True)
    args = p.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(args.by_sequence)
    a = select_method(data, args.method_a, args.metric)
    b = select_method(data, args.method_b, args.metric)
    paired = a.merge(b, on="sequence_id", suffixes=("_a", "_b"))
    if paired.empty:
        raise ValueError("No paired sequence rows found")

    paired["delta"] = paired[f"{args.metric}_b"].astype(float) - paired[f"{args.metric}_a"].astype(float)
    deltas = paired["delta"].to_numpy(dtype=float)
    mean_delta = float(np.mean(deltas))
    std_delta = float(np.std(deltas, ddof=1)) if len(deltas) > 1 else float("nan")
    effect_size = mean_delta / std_delta if std_delta and not math.isnan(std_delta) else float("nan")
    required_n = approximate_paired_n(effect_size, args.alpha, args.target_power)
    ci_low, ci_high, boot = bootstrap_required_n(deltas, args.bootstrap_samples, args.alpha, args.target_power)
    stable = stability_flag(required_n, ci_low, ci_high, len(deltas))

    result = pd.DataFrame(
        [
            {
                "method_a": args.method_a,
                "method_b": args.method_b,
                "metric": args.metric,
                "n_sequences_observed": len(deltas),
                "mean_delta": mean_delta,
                "std_delta": std_delta,
                "paired_effect_size_dz": effect_size,
                "alpha": args.alpha,
                "target_power": args.target_power,
                "estimated_required_sequences": required_n,
                "bootstrap_ci_low": ci_low,
                "bootstrap_ci_high": ci_high,
                "estimate_stable": stable,
            }
        ]
    )
    result.to_csv(out / "power_analysis.csv", index=False)
    paired.to_csv(out / "paired_sequence_deltas.csv", index=False)
    plot_power(effect_size, len(deltas), args.alpha, out / "fig_power_analysis_ru.png")
    write_summary(out / "power_analysis_summary.md", result.iloc[0], paired, boot)
    print(f"status=ok n={len(deltas)} required={required_n} stable={stable} output={out}")


def select_method(data: pd.DataFrame, spec: str, metric: str) -> pd.DataFrame:
    tracker, trust_mode = spec.split(":", 1)
    df = data.copy()
    if "available" in df:
        df = df[df["available"].fillna(False).astype(bool)]
    hit = df[df["tracker"].astype(str).eq(tracker) & df["trust_mode"].astype(str).eq(trust_mode)]
    if hit.empty:
        raise ValueError(f"No rows for method {spec}")
    return hit[["sequence_id", metric]].copy()


def approximate_paired_n(effect_size: float, alpha: float, target_power: float) -> float:
    if effect_size == 0 or math.isnan(effect_size):
        return float("inf")
    z_alpha = stats.norm.ppf(1.0 - alpha / 2.0)
    z_power = stats.norm.ppf(target_power)
    return float(math.ceil(((z_alpha + z_power) / abs(effect_size)) ** 2))


def bootstrap_required_n(deltas: np.ndarray, samples: int, alpha: float, target_power: float) -> tuple[float, float, np.ndarray]:
    if len(deltas) < 2:
        return (float("nan"), float("nan"), np.array([]))
    rng = np.random.default_rng(2026)
    vals = []
    for _ in range(samples):
        boot = rng.choice(deltas, size=len(deltas), replace=True)
        std = float(np.std(boot, ddof=1))
        if std <= 1e-9:
            continue
        eff = float(np.mean(boot)) / std
        if abs(eff) <= 1e-9:
            continue
        vals.append(approximate_paired_n(eff, alpha, target_power))
    arr = np.asarray(vals, dtype=float)
    if len(arr) == 0:
        return (float("nan"), float("nan"), arr)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return (float("nan"), float("nan"), arr)
    return (float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5)), arr)


def stability_flag(required_n: float, ci_low: float, ci_high: float, observed_n: int) -> str:
    if not np.isfinite(required_n) or not np.isfinite(ci_low) or not np.isfinite(ci_high):
        return "no"
    width = ci_high - ci_low
    if observed_n < 10:
        return "no"
    if width > max(10.0, 1.5 * required_n):
        return "no"
    return "yes"


def plot_power(effect_size: float, observed_n: int, alpha: float, path: Path) -> None:
    xs = np.arange(3, 61)
    if effect_size == 0 or math.isnan(effect_size):
        ys = np.zeros_like(xs, dtype=float)
    else:
        z_alpha = stats.norm.ppf(1.0 - alpha / 2.0)
        ys = 1.0 - stats.norm.cdf(z_alpha - np.sqrt(xs) * abs(effect_size))
    plt.figure(figsize=(7, 4))
    plt.plot(xs, ys, color="#1f77b4", linewidth=2)
    plt.axhline(0.80, color="#d62728", linestyle="--", label="power = 0.80")
    plt.axvline(observed_n, color="#555555", linestyle=":", label=f"observed n={observed_n}")
    plt.xlabel("Число независимых последовательностей")
    plt.ylabel("Приблизительная мощность")
    plt.title("Power analysis для false-new tracks")
    plt.ylim(0, 1.02)
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def write_summary(path: Path, row: pd.Series, paired: pd.DataFrame, boot: np.ndarray) -> None:
    required = row["estimated_required_sequences"]
    stable = str(row["estimate_stable"])
    if stable == "yes":
        sentence = (
            f"Approximate paired-power analysis estimates about {required:.0f} independent sequences "
            f"for 80% power at alpha={row['alpha']}."
        )
        answer_15_20 = "Yes, if the estimate falls in that range."
    else:
        sentence = (
            "Power estimate is unstable because only 7 independent VisDrone sequences are available; "
            "report it as an exploratory calculation, not as a firm sample-size requirement."
        )
        answer_15_20 = "No. The estimate should not be presented as a stable 15-20 sequence requirement."
    lines = [
        "# Q1 Power Analysis Summary",
        "",
        "Comparison: ByteTrack vs ByteTrack + geometry trust.",
        f"Metric: `{row['metric']}`.",
        "",
        "Sequence-level deltas are method_b - method_a:",
        "",
        "```text",
        paired[["sequence_id", f"{row['metric']}_a", f"{row['metric']}_b", "delta"]].to_string(index=False),
        "```",
        "",
        f"mean_delta_false_new: {row['mean_delta']:.6f}",
        f"std_delta_false_new: {row['std_delta']:.6f}",
        f"paired_effect_size: {row['paired_effect_size_dz']:.6f}",
        f"estimated_required_sequences: {required}",
        f"bootstrap_n_ci95: [{row['bootstrap_ci_low']}, {row['bootstrap_ci_high']}]",
        f"estimate_stable: {stable}",
        "",
        "Required answers:",
        f"1. Can we robustly claim 15-20 sequences? {answer_15_20}",
        "2. Calculation: normal approximation for paired mean delta using observed paired effect size.",
        "3. Stability: unstable if based on only 7 sequences or wide bootstrap CI.",
        f"4. Safe article sentence: {sentence}",
    ]
    if len(boot):
        lines += ["", f"Bootstrap finite samples: {len(boot)}"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
