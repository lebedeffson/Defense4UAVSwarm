#!/usr/bin/env python
from __future__ import annotations

import argparse
import numpy as np
import pandas as pd


EPS = 1e-9


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--baseline-scenario", default="s_naive")
    p.add_argument("--method-scenario", default="s2_tnorm_soft")
    p.add_argument("--unit", default="seed_sequence")
    p.add_argument("--n-bootstrap", type=int, default=1000)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    df = pd.read_csv(args.input)
    df["scenario_key"] = df["scenario"].str.lower()
    sort_cols = [c for c in ["seed", "sequence_id", "scenario_key", "F1", "IDF1"] if c in df.columns]
    if {"seed", "sequence_id", "scenario_key", "F1"}.issubset(df.columns):
        df = (
            df.sort_values(sort_cols, ascending=[True, True, True, False, False][: len(sort_cols)])
            .groupby(["seed", "sequence_id", "scenario_key"], as_index=False)
            .head(1)
            .reset_index(drop=True)
        )
    units = df[["seed", "sequence_id"]].drop_duplicates().reset_index(drop=True)
    rng = np.random.default_rng(args.seed)
    rows = []
    samples = []
    for _ in range(args.n_bootstrap):
        idx = rng.integers(0, len(units), size=len(units))
        sample_units = units.iloc[idx]
        sample = pd.concat(
            [
                df[(df.seed == u.seed) & (df.sequence_id == u.sequence_id)]
                for u in sample_units.itertuples(index=False)
            ],
            ignore_index=True,
        )
        b = aggregate(sample[sample.scenario_key == args.baseline_scenario.lower()])
        m = aggregate(sample[sample.scenario_key == args.method_scenario.lower()])
        for name, vals in [("S_naive", b), ("S2_tnorm_soft", m)]:
            for metric, value in vals.items():
                samples.append({"metric": metric, "scenario_or_delta": name, "value": value})
        for metric in ["FP", "FN", "F1", "IDF1", "track_breaks", "false_new_tracks", "false_new_tracks_per_100_frames"]:
            samples.append({"metric": f"{metric}_delta", "scenario_or_delta": "delta", "value": m[metric] - b[metric]})
    sample_df = pd.DataFrame(samples)
    for (metric, scenario), group in sample_df.groupby(["metric", "scenario_or_delta"], sort=False):
        values = group["value"].dropna()
        if values.empty:
            continue
        rows.append(
            {
                "metric": metric,
                "scenario_or_delta": scenario,
                "mean": values.mean(),
                "ci95_low": values.quantile(0.025),
                "ci95_high": values.quantile(0.975),
                "n_bootstrap": args.n_bootstrap,
                "unit": args.unit,
                "bootstrap_seed": args.seed,
            }
        )
    pd.DataFrame(rows).to_csv(args.output, index=False)


def aggregate(frame: pd.DataFrame) -> dict:
    tp = float(frame["TP"].sum())
    fp = float(frame["FP"].sum())
    fn = float(frame["FN"].sum())
    precision = tp / (tp + fp + EPS)
    recall = tp / (tp + fn + EPS)
    f1 = 2 * precision * recall / (precision + recall + EPS)
    frames = float(frame.get("num_frames", pd.Series([1])).sum())
    false_new = float(frame.get("false_new_tracks", pd.Series([0])).sum())
    return {
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "precision": precision,
        "recall": recall,
        "F1": f1,
        "IDF1": f1,
        "track_breaks": float(frame.get("track_breaks", pd.Series([0])).sum()),
        "false_new_tracks": false_new,
        "false_new_tracks_per_100_frames": false_new / max(1.0, frames) * 100.0,
    }


if __name__ == "__main__":
    main()
