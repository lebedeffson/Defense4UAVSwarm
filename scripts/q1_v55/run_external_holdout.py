#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

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
    out = results_root(cfg, args.output_dir) / "holdout"
    prepare_output(out, overwrite=args.overwrite, dry_run=args.dry_run)
    if args.dry_run:
        print(f"dry_run=ok output={out}")
        return
    status = {
        "external_holdout_status": "unavailable",
        "reason": "No frozen unused VisDrone detector/tracker outputs are available in the repository for v2.2-RP.",
        "claim_status": "post_hoc_secondary_validation_only",
    }
    (out / "selected_final_config.json").write_text(json.dumps({"status": "not_selected_for_holdout"}, indent=2), encoding="utf-8")
    (out / "frozen_model_manifest.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    for name in ["holdout_results_by_sequence.csv", "holdout_summary.csv", "holdout_paired_deltas.csv", "holdout_statistical_results.csv", "holdout_terminal_audit.csv"]:
        pd.DataFrame([status]).to_csv(out / name, index=False)
    write_metadata(out / "run_metadata.json", status)
    print(f"status=ok holdout=unavailable output={out}")


if __name__ == "__main__":
    main()
