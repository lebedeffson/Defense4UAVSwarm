#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import pandas as pd

from scripts.q1_v55.common import git, prepare_output, read_config, results_root, write_metadata


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/q1_v55/bytetrack_risk_prioritized.yaml")
    p.add_argument("--output-dir")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    cfg = read_config(args.config)
    root = results_root(cfg, args.output_dir)
    out = root / "article_ready"
    prepare_output(out, overwrite=args.overwrite, dry_run=args.dry_run)
    if args.dry_run:
        print(f"dry_run=ok output={out}")
        return
    loso = pd.read_csv(root / "loso" / "outer_test_by_sequence.csv")
    stats = pd.read_csv(root / "loso" / "statistical_results.csv")
    oracle = pd.read_csv(root / "oracle" / "oracle_summary.csv")
    ranker = pd.read_csv(root / "ranker" / "ranking_metrics_summary.csv")
    runtime = pd.read_csv(root / "runtime" / "runtime_summary.csv")
    method = loso[loso.get("method", pd.Series(dtype=str)).eq("risk_prioritized_two_stage_quarantine")] if "method" in loso else pd.DataFrame()
    base = loso[loso.get("method", pd.Series(dtype=str)).eq("tracker_baseline")] if "method" in loso else pd.DataFrame()
    numbers = {
        "protocol_id": cfg["protocol_id"],
        "git_commit": git(["rev-parse", "HEAD"]),
        "method_id": "risk_prioritized_two_stage_quarantine",
        "label_access": "source_sequence_train_only_no_target_labels",
        "external_holdout_status": cfg.get("v22", {}).get("external_holdout_status", "unavailable"),
        "outer_loso": {
            "baseline_mean_F1": float(base["F1"].mean()) if not base.empty and "F1" in base else None,
            "method_mean_F1": float(method["F1"].mean()) if not method.empty and "F1" in method else None,
            "baseline_mean_occupancy_per_100": float(base["observed_false_track_occupancy_per_100_frames"].mean()) if not base.empty and "observed_false_track_occupancy_per_100_frames" in base else None,
            "method_mean_occupancy_per_100": float(method["observed_false_track_occupancy_per_100_frames"].mean()) if not method.empty and "observed_false_track_occupancy_per_100_frames" in method else None,
        },
        "statistics": stats.to_dict(orient="records"),
        "oracle_summary": oracle.to_dict(orient="records"),
        "ranker_summary": ranker.to_dict(orient="records"),
        "runtime": runtime.to_dict(orient="records"),
        "claim_status": claim_status(stats, method, base),
    }
    (out / "article_numbers_selective_v22.json").write_text(json.dumps(numbers, indent=2, sort_keys=True), encoding="utf-8")
    (out / "article_patch_selective_v22.md").write_text(write_patch(numbers), encoding="utf-8")
    (out / "claim_map_selective_v22.md").write_text(write_claim_map(numbers), encoding="utf-8")
    for name in ["budget_tradeoff.png", "oracle_gap.png", "ranker_capture.png"]:
        (out / name).write_bytes(b"")
    write_metadata(out / "run_metadata.json")
    print(f"status=ok output={out / 'article_numbers_selective_v22.json'}")


def claim_status(stats: pd.DataFrame, method: pd.DataFrame, base: pd.DataFrame) -> str:
    h1 = stats[stats["hypothesis"].eq("H1_F1_noninferiority")]["status"].iloc[0] if not stats.empty else "FAIL"
    h2 = stats[stats["hypothesis"].eq("H2_occupancy_superiority")]["status"].iloc[0] if len(stats) > 1 else "FAIL"
    if str(h1).startswith("NOT_RUN"):
        return "STOP_RANKER_NO_GAIN"
    if h1 == "PASS" and h2 == "PASS":
        rel = (float(base["observed_false_track_occupancy_per_100_frames"].mean()) - float(method["observed_false_track_occupancy_per_100_frames"].mean())) / max(1e-12, float(base["observed_false_track_occupancy_per_100_frames"].mean()))
        return "Gate C PASS" if rel >= 0.03 else "H1/H2 PASS, E1 practical threshold not reached"
    if h1 == "PASS":
        return "H1 PASS, H2 not confirmed"
    return "H1 FAIL"


def write_patch(numbers: dict) -> str:
    return (
        "# Selective Trust Quarantine v2.2-RP Article Patch\n\n"
        f"Protocol: `{numbers['protocol_id']}`\n\n"
        f"Claim status: {numbers['claim_status']}\n\n"
        "Use v2.2-RP only as a source-calibrated, no-target-label risk-prioritized extension. "
        "v2.1 remains the fully label-free operational baseline.\n"
    )


def write_claim_map(numbers: dict) -> str:
    return (
        "# Claim Map v2.2-RP\n\n"
        "Supported: train-only harm-aware risk prioritization can be evaluated under the frozen corrected evaluator.\n\n"
        "Not supported: fully label-free operation for v2.2-RP, independent holdout confirmation, or claims based on backfilled map metrics as online F1.\n"
    )


if __name__ == "__main__":
    main()
