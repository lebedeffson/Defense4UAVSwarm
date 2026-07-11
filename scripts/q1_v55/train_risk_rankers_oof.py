#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import pandas as pd

from defense4uavswarm.q1_v5.risk_ranker import oof_predictions
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
    out = root / "ranker"
    prepare_output(out, overwrite=args.overwrite, dry_run=args.dry_run)
    if args.dry_run:
        print(f"dry_run=ok output={out}")
        return
    episodes = pd.read_parquet(root / "episode_harm" / "episode_harm_table.parquet")
    p1, p2, cards = oof_predictions(episodes, seed=int(cfg.get("v22", {}).get("seed", 2026)))
    p1.to_parquet(out / "stage1_oof_predictions.parquet", index=False)
    p2.to_parquet(out / "stage2_oof_predictions.parquet", index=False)
    cards.to_csv(out / "coefficients_by_fold.csv", index=False)
    (out / "thresholds_by_fold.json").write_text(json.dumps({}, indent=2), encoding="utf-8")
    (out / "feature_schema.json").write_text(json.dumps({"stage1": "see src/defense4uavswarm/q1_v5/risk_ranker.py", "stage2": "see src/defense4uavswarm/q1_v5/risk_ranker.py"}, indent=2), encoding="utf-8")
    merged = episodes.merge(p1[["sequence_id", "tracklet_id", "episode_id", "risk_score"]].rename(columns={"risk_score": "stage1_risk_score"}), on=["sequence_id", "tracklet_id", "episode_id"], how="left")
    budgets = [0.01, 0.03, 0.05, 0.10]
    rows = []
    for seq, group in merged.groupby("sequence_id", sort=False):
        false_total = float(group["unmatched_false_rows"].sum())
        for q in budgets:
            k = max(1, int(len(group) * q))
            top = group.sort_values("stage1_risk_score", ascending=False).head(k)
            rows.append(
                {
                    "sequence_id": seq,
                    "budget": q,
                    "ranker": "stage1_risk",
                    "precision_false_at_budget": float(top["is_false_episode"].astype(bool).mean()) if len(top) else 0.0,
                    "captured_false_occupancy_fraction": float(top["unmatched_false_rows"].sum() / max(1.0, false_total)),
                    "true_episode_admission_fraction": float((~top["is_false_episode"].astype(bool)).sum() / max(1, (~group["is_false_episode"].astype(bool)).sum())),
                    "NDCG_at_budget": float(top["unmatched_false_rows"].sum() / max(1.0, group.sort_values("unmatched_false_rows", ascending=False).head(k)["unmatched_false_rows"].sum())),
                    "oracle_gap_closed_fraction": float(top["unmatched_false_rows"].sum() / max(1.0, group.sort_values("unmatched_false_rows", ascending=False).head(k)["unmatched_false_rows"].sum())),
                }
            )
    metrics = pd.DataFrame(rows)
    summary = metrics.groupby(["budget", "ranker"], as_index=False).mean(numeric_only=True)
    gate_b = {
        "max_oracle_gap_closed_fraction": float(summary["oracle_gap_closed_fraction"].max()) if not summary.empty else 0.0,
        "decision": "CONTINUE" if (not summary.empty and float(summary["oracle_gap_closed_fraction"].max()) >= 0.25) else "STOP_RANKER_NO_GAIN",
    }
    metrics.to_csv(out / "ranking_metrics_by_sequence.csv", index=False)
    summary.to_csv(out / "ranking_metrics_summary.csv", index=False)
    (out / "go_no_go.json").write_text(json.dumps({"gate_b": gate_b}, indent=2, sort_keys=True), encoding="utf-8")
    (out / "model_cards").mkdir(exist_ok=True)
    for row in cards.itertuples(index=False):
        (out / "model_cards" / f"{row.excluded_sequence}.json").write_text(json.dumps(row._asdict(), indent=2, sort_keys=True), encoding="utf-8")
    write_metadata(out / "run_metadata.json", {"gate_b": gate_b})
    print(f"status=ok gate_b={gate_b['decision']} output={out}")


if __name__ == "__main__":
    main()
