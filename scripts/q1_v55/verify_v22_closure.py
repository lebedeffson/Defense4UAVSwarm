#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import pandas as pd

from scripts.q1_v55.common import read_config, results_root, write_metadata


REQUIRED = [
    "protocol/protocol_lock.json",
    "episode_harm/episode_harm_table.parquet",
    "oracle/go_no_go.json",
    "ranker/go_no_go.json",
    "loso/outer_test_by_sequence.csv",
    "loso/statistical_results.csv",
    "loso/budget_audit.csv",
    "runtime/runtime_summary.csv",
    "article_ready/article_numbers_selective_v22.json",
]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/q1_v55/bytetrack_risk_prioritized.yaml")
    p.add_argument("--output-dir")
    args = p.parse_args()
    cfg = read_config(args.config)
    root = results_root(cfg, args.output_dir)
    out = root / "closure"
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for rel in REQUIRED:
        path = root / rel
        rows.append({"check": f"exists:{rel}", "status": "PASS" if path.exists() else "FAIL", "detail": path.as_posix()})
    stats = pd.read_csv(root / "loso" / "statistical_results.csv") if (root / "loso" / "statistical_results.csv").exists() else pd.DataFrame()
    if not stats.empty:
        h1 = stats[stats["hypothesis"].eq("H1_F1_noninferiority")]["status"].iloc[0]
        h2 = stats[stats["hypothesis"].eq("H2_occupancy_superiority")]["status"].iloc[0]
        rows.append({"check": "H2_not_pass_when_H1_fail", "status": "PASS" if h1 == "PASS" or h2 != "PASS" else "FAIL", "detail": f"h1={h1} h2={h2}"})
        rows.append({"check": "gate_b_stop_is_explicit", "status": "PASS" if h1 != "NOT_RUN_GATE_B_FAIL" or (root / "ranker" / "go_no_go.json").exists() else "FAIL", "detail": str(h1)})
    budget = pd.read_csv(root / "loso" / "budget_audit.csv") if (root / "loso" / "budget_audit.csv").exists() else pd.DataFrame()
    if not budget.empty and "conservation_residual" in budget:
        rows.append({"check": "budget_conservation_residual", "status": "PASS" if budget["conservation_residual"].abs().max() <= 1e-9 else "FAIL", "detail": str(float(budget["conservation_residual"].abs().max()))})
    else:
        rows.append({"check": "budget_conservation_residual", "status": "PASS", "detail": "not_applicable_before_online_integration"})
    article = json.loads((root / "article_ready" / "article_numbers_selective_v22.json").read_text(encoding="utf-8")) if (root / "article_ready" / "article_numbers_selective_v22.json").exists() else {}
    rows.append({"check": "article_protocol_id", "status": "PASS" if article.get("protocol_id") == cfg.get("protocol_id") else "FAIL", "detail": str(article.get("protocol_id"))})
    df = pd.DataFrame(rows)
    df.to_csv(out / "closure_checks.csv", index=False)
    status = "PASS" if df["status"].eq("PASS").all() else "FAIL"
    (out / "closure_summary.json").write_text(json.dumps({"status": status, "checks": rows}, indent=2), encoding="utf-8")
    write_metadata(out / "run_metadata.json", {"closure_status": status})
    print(f"status={status} output={out / 'closure_checks.csv'}")
    if status != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
