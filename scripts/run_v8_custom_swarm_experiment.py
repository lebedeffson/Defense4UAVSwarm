#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from defense4uavswarm.v8_sim import evaluate_experiment, write_csv, write_main_outputs


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--gt-2d", required=True)
    p.add_argument("--gt-3d", default="")
    p.add_argument("--detections", required=True)
    p.add_argument("--scenarios", nargs="+", default=["s_naive", "s2_tnorm_soft", "s2_tnorm_temporal", "s2_ema_confidence_gate", "s2_support_count_gate"])
    p.add_argument("--selected-params", default="configs/selected_s2_temporal.yaml")
    p.add_argument("--split", default="holdout")
    p.add_argument("--agent-counts", nargs="+", type=int, default=None)
    p.add_argument("--output-dir", required=True)
    args = p.parse_args()

    out = Path(args.output_dir)
    result = evaluate_experiment(args.manifest, args.gt_2d, args.detections, args.scenarios, args.split, args.selected_params)
    write_main_outputs(result, out)
    if args.agent_counts:
        rows = []
        for count in args.agent_counts:
            sub = evaluate_experiment(args.manifest, args.gt_2d, args.detections, args.scenarios, args.split, args.selected_params, agent_count=count)
            for row in sub["summary"]:
                row = dict(row)
                row["num_agents"] = count
                row["support_count_mean"] = float(sub["features"]["support_count"].mean()) if len(sub["features"]) else 0.0
                row["s_i_mean"] = float(sub["features"]["s_i"].mean()) if len(sub["features"]) else 0.0
                row["runtime_ms"] = row.get("runtime_ms_per_frame", 0.0)
                rows.append(row)
        pd.DataFrame(rows).to_csv(out / "agent_count_ablation.csv", index=False)
    print(f"status=ok output={out / 'main_comparison_table.csv'}")


if __name__ == "__main__":
    main()
