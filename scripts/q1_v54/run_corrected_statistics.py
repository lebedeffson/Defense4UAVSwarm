#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/q1_v54/bytetrack.yaml")
    p.add_argument("--loso", default="outputs/results/q1_v542/loso/outer_test_by_sequence.csv")
    p.add_argument("--output-dir", default="outputs/results/q1_v542/statistics")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--n-resamples", type=int, default=10000)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    out = Path(args.output_dir)
    if out.exists() and any(out.iterdir()) and not args.overwrite and not args.dry_run:
        raise SystemExit(f"Output exists; use --overwrite: {out}")
    out.mkdir(parents=True, exist_ok=True)
    if args.dry_run:
        print(f"dry_run=ok output={out}")
        return
    df = pd.read_csv(args.loso)
    df = df[df["F1"].notna()].copy()
    rows = []
    deltas = []
    samples_payload: dict[str, np.ndarray] = {}
    rng = np.random.default_rng(args.seed)
    for tracker, group in df.groupby("tracker", sort=False):
        occupancy_col = "decision_complete_online_observed_false_track_occupancy_per_100_frames"
        if occupancy_col not in group.columns:
            occupancy_col = "observed_false_track_occupancy_per_100_frames"
        if occupancy_col not in group.columns:
            occupancy_col = "observed_false_track_occupancy_rows"
        f1_col = "decision_complete_online_F1" if "decision_complete_online_F1" in group.columns else "F1"
        base = group[group["method"].eq("tracker_baseline")][["outer_test_sequence", f1_col, occupancy_col, "false_new_tracks_per_100_frames"]]
        for method, mg in group.groupby("method", sort=False):
            if method == "tracker_baseline":
                continue
            paired = mg.merge(base, on="outer_test_sequence", suffixes=("_method", "_baseline"))
            if paired.empty:
                continue
            f1_delta = paired[f"{f1_col}_method"].astype(float).to_numpy() - paired[f"{f1_col}_baseline"].astype(float).to_numpy()
            occ_delta = paired[f"{occupancy_col}_method"].astype(float).to_numpy() - paired[f"{occupancy_col}_baseline"].astype(float).to_numpy()
            f1_samples = paired_bootstrap_mean(f1_delta, args.n_resamples, rng)
            occ_samples = paired_bootstrap_mean(occ_delta, args.n_resamples, rng)
            f1_ci = ci95(f1_samples)
            occ_ci = ci95(occ_samples)
            h1 = f1_ci[0] > -0.01
            h2_tested = bool(h1)
            h2 = bool(h1 and occ_ci[1] < 0)
            key = sanitize(f"{tracker}_{method}")
            samples_payload[f"{key}_f1_delta"] = f1_samples
            samples_payload[f"{key}_occupancy_delta"] = occ_samples
            rows.append(
                {
                    "tracker": tracker,
                    "method": method,
                    "hypothesis": "H1_F1_noninferiority",
                    "metric": f1_col,
                    "tested_after_h1": True,
                    "n_sequences": len(f1_delta),
                    "mean_delta": float(f1_delta.mean()),
                    "median_delta": float(np.median(f1_delta)),
                    "ci95_low": float(f1_ci[0]),
                    "ci95_high": float(f1_ci[1]),
                    "margin": -0.01,
                    "status": "PASS" if h1 else "FAIL",
                }
            )
            rows.append(
                {
                    "tracker": tracker,
                    "method": method,
                    "hypothesis": "H2_occupancy_superiority",
                    "metric": occupancy_col,
                    "tested_after_h1": h2_tested,
                    "n_sequences": len(occ_delta),
                    "mean_delta": float(occ_delta.mean()),
                    "median_delta": float(np.median(occ_delta)),
                    "ci95_low": float(occ_ci[0]),
                    "ci95_high": float(occ_ci[1]),
                    "margin": 0.0,
                    "status": "PASS" if h2 else "NOT_TESTED" if not h2_tested else "FAIL",
                }
            )
            for row in paired.itertuples(index=False):
                deltas.append(
                    {
                        "tracker": tracker,
                        "method": method,
                        "outer_test_sequence": row.outer_test_sequence,
                        "delta_F1": getattr(row, f"{f1_col}_method") - getattr(row, f"{f1_col}_baseline"),
                        "delta_occupancy": getattr(row, f"{occupancy_col}_method") - getattr(row, f"{occupancy_col}_baseline"),
                        "occupancy_metric": occupancy_col,
                    }
                )
    pd.DataFrame(rows).to_csv(out / "statistical_results.csv", index=False)
    pd.DataFrame(deltas).to_csv(out / "paired_sequence_deltas.csv", index=False)
    np.savez(out / "bootstrap_samples.npz", **samples_payload)
    (out / "statistical_claim_safe.md").write_text(write_report(pd.DataFrame(rows)), encoding="utf-8")
    (out / "run_metadata.json").write_text(json.dumps({"status": "success", "git_commit": git(["rev-parse", "HEAD"]), "branch": git(["branch", "--show-current"]), "n_resamples": args.n_resamples, "seed": args.seed}, indent=2), encoding="utf-8")
    print(f"status=ok output={out} rows={len(rows)}")


def paired_bootstrap_mean(values: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    if values.size == 0:
        return np.asarray([])
    idx = rng.integers(0, values.size, size=(n, values.size))
    return values[idx].mean(axis=1)


def ci95(samples: np.ndarray) -> tuple[float, float]:
    return float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))


def sanitize(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", text)


def write_report(df: pd.DataFrame) -> str:
    return "# Corrected Sequence-Level Statistics\n\n" + df.to_string(index=False) + "\n"


def git(args: list[str]) -> str:
    try:
        return subprocess.check_output(["git", *args], text=True).strip()
    except Exception:
        return "unavailable"


if __name__ == "__main__":
    main()
