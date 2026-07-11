#!/usr/bin/env python
from __future__ import annotations

import argparse
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
    out = results_root(cfg, args.output_dir) / "baselines"
    prepare_output(out, overwrite=args.overwrite, dry_run=args.dry_run)
    if args.dry_run:
        print(f"dry_run=ok output={out}")
        return
    note = {
        "status": "not_run",
        "reason": "Matched-safety baselines are gated by Gate C or separate author instruction in the v2.2-RP protocol.",
    }
    pd.DataFrame([note]).to_csv(out / "matched_safety_baselines_status.csv", index=False)
    write_metadata(out / "run_metadata.json", note)
    print(f"status=ok matched_safety_baselines=not_run output={out}")


if __name__ == "__main__":
    main()
