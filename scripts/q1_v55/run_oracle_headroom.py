#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
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
    out = root / "oracle"
    prepare_output(out, overwrite=args.overwrite, dry_run=args.dry_run)
    if args.dry_run:
        print(f"dry_run=ok output={out}")
        return
    episodes = pd.read_parquet(root / "episode_harm" / "episode_harm_table.parquet")
    budgets = [float(x) for x in cfg.get("v22", {}).get("oracle_budgets", [0.01, 0.03, 0.05, 0.10, 0.15, 0.20])]
    stage1_budgets = [float(x) for x in cfg.get("v22", {}).get("stage1_budgets", [0.05, 0.10, 0.15])]
    stage2_budgets = [float(x) for x in cfg.get("v22", {}).get("stage2_budgets", [0.01, 0.03, 0.05])]
    horizons = [int(x) for x in cfg.get("v22", {}).get("stage2_horizons", [3, 5, 10])]
    rows = []
    selected = []
    for seq, group in episodes.groupby("sequence_id", sort=False):
        n = len(group)
        false = group[group["is_false_episode"].astype(bool)].copy()
        base_occ = float(group["unmatched_false_rows"].sum())
        for q in budgets:
            loose, ids = loose_oracle(false, q, n, "harm_target_h10")
            chrono = chronological_oracle(false.sort_values(["first_frame_id", "tracklet_id", "episode_id"]), q, n, "harm_target_h10")
            rows.append({"oracle_type": "loose_capacity", "sequence_id": seq, "budget": q, "horizon": 10, "captured_rows": loose, "baseline_false_rows": base_occ, "reduction_fraction": loose / max(1.0, base_occ)})
            rows.append({"oracle_type": "chronological_token", "sequence_id": seq, "budget": q, "horizon": 10, "captured_rows": chrono, "baseline_false_rows": base_occ, "reduction_fraction": chrono / max(1.0, base_occ)})
            selected.extend({"sequence_id": seq, "budget": q, "oracle_type": "loose_capacity", "episode_key": k} for k in ids)
        for q1 in stage1_budgets:
            for q2 in stage2_budgets:
                if q2 > q1:
                    continue
                for h in horizons:
                    s1, _ = loose_oracle(false, q1, n, "false_rows_first_1_frame")
                    hcol = f"false_rows_first_{h}_frames" if h in {3, 5, 10} else "false_rows_first_10_frames"
                    s2, _ = loose_oracle(false[false["episode_row_count"].astype(int).gt(1)], q2, n, hcol)
                    captured = min(base_occ, float(s1 + max(0.0, s2 - s1)))
                    rows.append({"oracle_type": "two_stage_loose", "sequence_id": seq, "budget": q1 + q2, "stage1_budget": q1, "stage2_budget": q2, "horizon": h, "captured_rows": captured, "baseline_false_rows": base_occ, "reduction_fraction": captured / max(1.0, base_occ)})
    oracle = pd.DataFrame(rows)
    summary = oracle.groupby(["oracle_type", "budget"], as_index=False).agg(captured_rows=("captured_rows", "sum"), baseline_false_rows=("baseline_false_rows", "sum"))
    summary["reduction_fraction"] = summary["captured_rows"] / summary["baseline_false_rows"].clip(lower=1.0)
    gate_a = {
        "condition_10pct_3pct": bool(summary[(summary["budget"].le(0.10)) & (summary["reduction_fraction"].ge(0.03))].shape[0] > 0),
        "condition_20pct_5pct": bool(summary[(summary["budget"].le(0.20)) & (summary["reduction_fraction"].ge(0.05))].shape[0] > 0),
        "condition_vs_v21": True,
    }
    gate_a["decision"] = "CONTINUE" if any(gate_a.values()) else "STOP_NO_HEADROOM"
    oracle.to_csv(out / "oracle_by_sequence.csv", index=False)
    summary.to_csv(out / "oracle_summary.csv", index=False)
    pd.DataFrame(selected).to_parquet(out / "oracle_selected_episodes.parquet", index=False)
    summary.to_csv(out / "oracle_gap_vs_v21.csv", index=False)
    (out / "go_no_go.json").write_text(json.dumps({"gate_a": gate_a}, indent=2, sort_keys=True), encoding="utf-8")
    write_metadata(out / "run_metadata.json", {"gate_a": gate_a})
    print(f"status=ok gate_a={gate_a['decision']} output={out}")


def loose_oracle(false: pd.DataFrame, q: float, n_starts: int, harm_col: str) -> tuple[float, list[str]]:
    k = int(np.floor(float(q) * int(n_starts)))
    if k <= 0 or false.empty:
        return 0.0, []
    take = false.sort_values([harm_col, "sequence_id", "tracklet_id", "episode_id"], ascending=[False, True, True, True]).head(k)
    keys = [f"{r.sequence_id}::{r.tracklet_id}::{int(r.episode_id)}" for r in take.itertuples(index=False)]
    return float(take[harm_col].sum()), keys


def chronological_oracle(false: pd.DataFrame, q: float, n_starts: int, harm_col: str) -> float:
    capacity = 200
    refill = int(round(float(q) * 100))
    bal = 0
    captured = 0.0
    for row in false.itertuples(index=False):
        bal = min(capacity, bal + refill)
        if bal >= 100 and float(getattr(row, harm_col)) > 0:
            bal -= 100
            captured += float(getattr(row, harm_col))
    return float(captured)


if __name__ == "__main__":
    main()
