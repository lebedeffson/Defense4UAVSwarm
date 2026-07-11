#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import pandas as pd

from defense4uavswarm.q1_v5.episode_harm import EpisodeHarmInputs, build_episode_harm_table
from scripts.q1_v55.common import gate_config, load_frozen_inputs, matching_config, prepare_output, read_config, results_root, write_metadata


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/q1_v55/bytetrack_risk_prioritized.yaml")
    p.add_argument("--output-dir")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    cfg = read_config(args.config)
    out = results_root(cfg, args.output_dir) / "episode_harm"
    prepare_output(out, overwrite=args.overwrite, dry_run=args.dry_run)
    if args.dry_run:
        print(f"dry_run=ok output={out}")
        return
    det, gt, ignored, manifest = load_frozen_inputs(cfg)
    table = build_episode_harm_table(EpisodeHarmInputs(det, gt, ignored, manifest, matching_config(cfg), gate_config(cfg), tracker=str(cfg.get("tracker", "bytetrack"))))
    table.to_parquet(out / "episode_harm_table.parquet", index=False)
    table.to_csv(out / "episode_harm_table.csv", index=False)
    summary = {
        "episodes": [len(table)],
        "false_episode_rate": [float(table["is_false_episode"].mean()) if len(table) else 0.0],
        "false_rows": [int(table["unmatched_false_rows"].sum()) if len(table) else 0],
    }
    pd.DataFrame(summary).to_csv(out / "episode_harm_summary.csv", index=False)
    write_metadata(out / "run_metadata.json", {"rows": len(table)})
    print(f"status=ok output={out / 'episode_harm_table.parquet'} rows={len(table)}")


if __name__ == "__main__":
    main()
