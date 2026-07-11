#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from scripts.q1_v55.common import git, prepare_output, read_config, results_root, write_metadata


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/q1_v55/bytetrack_risk_prioritized.yaml")
    p.add_argument("--output-dir")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    cfg = read_config(args.config)
    out = results_root(cfg, args.output_dir) / "protocol"
    prepare_output(out, overwrite=args.overwrite, dry_run=args.dry_run)
    if args.dry_run:
        print(f"dry_run=ok output={out}")
        return
    v22 = cfg.get("v22", {})
    lock = {
        "protocol_id": cfg.get("protocol_id"),
        "base_commit": cfg.get("base_commit"),
        "seed": int(v22.get("seed", 2026)),
        "bootstrap_resamples": int(v22.get("bootstrap_resamples", 10000)),
        "h1_margin": float(v22.get("h1_margin", -0.01)),
        "primary_metric": "online_current_frame_F1",
        "contamination_metric": "online_false_track_occupancy_per_100_frames",
        "oracle_budgets": v22.get("oracle_budgets", [0.01, 0.03, 0.05, 0.10, 0.15, 0.20]),
        "stage1_budgets": v22.get("stage1_budgets", [0.05, 0.10, 0.15]),
        "stage2_budgets": v22.get("stage2_budgets", [0.01, 0.03, 0.05]),
        "stage2_horizons": v22.get("stage2_horizons", [3, 5, 10]),
        "alpha_stage1_true": float(v22.get("alpha_stage1_true", 0.10)),
        "alpha_stage2_true": float(v22.get("alpha_stage2_true", 0.05)),
        "runtime_target_median_ms": float(v22.get("runtime_target_median_ms", 2.0)),
        "runtime_target_p95_ms": float(v22.get("runtime_target_p95_ms", 3.0)),
        "external_holdout_status": str(v22.get("external_holdout_status", "unavailable")),
    }
    (out / "protocol_lock.json").write_text(json.dumps(lock, indent=2, sort_keys=True), encoding="utf-8")
    dev = _development_sequences()
    (out / "development_sequences.txt").write_text("\n".join(dev) + "\n", encoding="utf-8")
    (out / "external_holdout_sequences.txt").write_text("# unavailable\n", encoding="utf-8")
    checksums = {}
    for path in [Path(args.config), Path(cfg["feature_audit"])]:
        checksums[path.as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    (out / "input_checksums.json").write_text(json.dumps(checksums, indent=2, sort_keys=True), encoding="utf-8")
    ref = Path("outputs/results/q1_selective_quarantine_v21/article_ready/article_numbers_selective_v21.json")
    if ref.exists():
        (out / "v21_reference_numbers.json").write_text(ref.read_text(encoding="utf-8"), encoding="utf-8")
    else:
        (out / "v21_reference_numbers.json").write_text(json.dumps({"status": "missing"}, indent=2), encoding="utf-8")
    env = {"python": platform.python_version(), "platform": platform.platform(), "commit": git(["rev-parse", "HEAD"])}
    (out / "environment.json").write_text(json.dumps(env, indent=2), encoding="utf-8")
    write_metadata(out / "run_metadata.json")
    print(f"status=ok output={out}")


def _development_sequences() -> list[str]:
    p = Path("outputs/results/q1_selective_quarantine_v21/loso/bytetrack/outer_folds.json")
    if not p.exists():
        return []
    data = json.loads(p.read_text(encoding="utf-8"))
    return sorted({str(row["outer_test_sequence"]) for row in data})


if __name__ == "__main__":
    main()
