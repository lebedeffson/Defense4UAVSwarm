#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np
import pandas as pd

from scripts.q1_v55.common import prepare_output, read_config, results_root, write_metadata


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/q1_v55/bytetrack_risk_prioritized.yaml")
    p.add_argument("--output-dir")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    cfg = read_config(args.config)
    root = results_root(cfg, args.output_dir)
    out = root / "loso"
    # Statistics live in loso per requested result structure.
    prepare_output(out, overwrite=True, dry_run=args.dry_run)
    if args.dry_run:
        print(f"dry_run=ok output={out}")
        return
    df = pd.read_csv(out / "outer_test_by_sequence.csv")
    if "F1" not in df.columns or "tracker_baseline" not in set(df.get("method", pd.Series(dtype=str)).astype(str)):
        rows = [
            {"hypothesis": "H1_F1_noninferiority", "metric": "online_current_frame_F1", "mean_delta": float("nan"), "ci95_low": float("nan"), "ci95_high": float("nan"), "margin": -0.01, "status": "NOT_RUN_GATE_B_FAIL"},
            {"hypothesis": "H2_occupancy_superiority", "metric": "online_false_track_occupancy_per_100_frames", "mean_delta": float("nan"), "ci95_low": float("nan"), "ci95_high": float("nan"), "margin": 0.0, "status": "NOT_TESTED_GATE_B_FAIL"},
        ]
        pd.DataFrame(rows).to_csv(out / "statistical_results.csv", index=False)
        pd.DataFrame(columns=["outer_test_sequence", "delta_F1", "delta_occupancy_per_100_frames"]).to_csv(out / "paired_sequence_deltas.csv", index=False)
        np.savez(out / "bootstrap_samples.npz")
        write_metadata(out / "statistics_run_metadata.json", {"h1": rows[0]["status"], "h2": rows[1]["status"]})
        print(f"status=NOT_RUN_GATE_B_FAIL output={out}")
        return
    base = df[df["method"].eq("tracker_baseline")][["outer_test_sequence", "F1", "observed_false_track_occupancy_per_100_frames"]].rename(columns={"F1": "baseline_F1", "observed_false_track_occupancy_per_100_frames": "baseline_occupancy"})
    method = df[df["method"].eq("risk_prioritized_two_stage_quarantine")].merge(base, on="outer_test_sequence", how="inner")
    selected = method.sort_values(["outer_fold"]).copy()
    f1_delta = selected["F1"].astype(float).to_numpy() - selected["baseline_F1"].astype(float).to_numpy()
    occ_delta = selected["observed_false_track_occupancy_per_100_frames"].astype(float).to_numpy() - selected["baseline_occupancy"].astype(float).to_numpy()
    rng = np.random.default_rng(int(cfg.get("v22", {}).get("seed", 2026)))
    n = int(cfg.get("v22", {}).get("bootstrap_resamples", 10000))
    f1_samples = bootstrap(f1_delta, n, rng)
    occ_samples = bootstrap(occ_delta, n, rng)
    h1 = float(np.quantile(f1_samples, 0.025)) > -0.01
    h2 = bool(h1 and float(np.quantile(occ_samples, 0.975)) < 0)
    rows = [
        {"hypothesis": "H1_F1_noninferiority", "metric": "online_current_frame_F1", "mean_delta": float(f1_delta.mean()), "ci95_low": float(np.quantile(f1_samples, 0.025)), "ci95_high": float(np.quantile(f1_samples, 0.975)), "margin": -0.01, "status": "PASS" if h1 else "FAIL"},
        {"hypothesis": "H2_occupancy_superiority", "metric": "online_false_track_occupancy_per_100_frames", "mean_delta": float(occ_delta.mean()), "ci95_low": float(np.quantile(occ_samples, 0.025)), "ci95_high": float(np.quantile(occ_samples, 0.975)), "margin": 0.0, "status": "PASS" if h2 else "NOT_TESTED" if not h1 else "FAIL"},
    ]
    pd.DataFrame(rows).to_csv(out / "statistical_results.csv", index=False)
    selected.assign(delta_F1=f1_delta, delta_occupancy_per_100_frames=occ_delta).to_csv(out / "paired_sequence_deltas.csv", index=False)
    np.savez(out / "bootstrap_samples.npz", f1_delta=f1_samples, occupancy_delta=occ_samples)
    write_metadata(out / "statistics_run_metadata.json", {"h1": rows[0]["status"], "h2": rows[1]["status"]})
    print(f"status=ok h1={rows[0]['status']} h2={rows[1]['status']} output={out}")


def bootstrap(values: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    idx = rng.integers(0, len(values), size=(n, len(values)))
    return values[idx].mean(axis=1)


if __name__ == "__main__":
    main()
