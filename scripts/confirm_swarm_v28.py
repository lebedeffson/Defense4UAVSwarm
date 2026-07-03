#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml


METRICS = ["TP", "FP", "FN", "precision", "recall", "F1", "IDF1", "IDSW", "track_breaks"]


def best_row(frame: pd.DataFrame, scenario: str) -> pd.Series | None:
    rows = frame[frame["scenario"] == scenario].copy()
    if rows.empty:
        return None
    return rows.sort_values(["F1", "IDF1"], ascending=False).iloc[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True)
    args = parser.parse_args()
    root = Path(args.results)
    summary = pd.read_csv(root / "summary_metrics.csv")
    naive = best_row(summary, "S_naive")
    if naive is None:
        raise SystemExit("S_naive not found")

    scenarios = ["S_naive", "S2_tnorm_soft", "S3_tnorm_xai_soft", "S2_tnorm_hard", "S3_tnorm_xai_hard"]
    rows = []
    for scenario in scenarios:
        row = best_row(summary, scenario)
        if row is None:
            continue
        out = {"scenario": scenario}
        for metric in METRICS:
            out[metric] = row.get(metric)
        out["FP_delta_vs_S_naive"] = row["FP"] - naive["FP"]
        out["FN_delta_vs_S_naive"] = row["FN"] - naive["FN"]
        out["F1_delta_vs_S_naive"] = row["F1"] - naive["F1"]
        out["IDF1_delta_vs_S_naive"] = row["IDF1"] - naive["IDF1"]
        out["IDSW_delta_vs_S_naive"] = row["IDSW"] - naive["IDSW"]
        out["track_break_delta_vs_S_naive"] = row["track_breaks"] - naive["track_breaks"]
        out["selected_candidate"] = scenario == "S2_tnorm_soft"
        rows.append(out)
    delta = pd.DataFrame(rows)
    delta.to_csv(root / "full_calibration_delta_vs_s_naive.csv", index=False)

    candidate = delta[delta["scenario"] == "S2_tnorm_soft"]
    if candidate.empty:
        selected = {
            "selection_status": "not_selected",
            "holdout_allowed": False,
            "reason": "s2_tnorm_soft_missing",
            "source_stage": "v2.8_full_calibration",
        }
    else:
        cand = candidate.iloc[0]
        fn_delta_pct = cand["FN_delta_vs_S_naive"] / max(1, naive["FN"])
        strict = cand["FN_delta_vs_S_naive"] <= 0 and cand["FP_delta_vs_S_naive"] < 0 and cand["F1_delta_vs_S_naive"] >= 0
        soft = fn_delta_pct <= 0.02 and cand["FP_delta_vs_S_naive"] < 0 and cand["F1_delta_vs_S_naive"] >= -0.001
        selected = {
            "selection_status": "selected" if strict else ("tradeoff_selected" if soft else "not_selected"),
            "holdout_allowed": bool(strict or soft),
            "selected_scenario": "S2_tnorm_soft",
            "selected_t_norm": "min",
            "selected_reweight_mode": "multiplicative_floor",
            "selected_q_floor": 0.6,
            "selected_q_hard_min": 0.0,
            "selected_new_track_threshold": 0.4,
            "selected_existing_track_threshold": 0.05,
            "source_stage": "v2.8_full_calibration",
            "reason": "full_calibration_confirms_fp_reduction_without_recall_loss" if strict else (
                "full_calibration_tradeoff_candidate" if soft else "compact_calibration_did_not_generalize"
            ),
            "FP_delta_vs_S_naive": int(cand["FP_delta_vs_S_naive"]),
            "FN_delta_vs_S_naive": int(cand["FN_delta_vs_S_naive"]),
            "F1_delta_vs_S_naive": float(cand["F1_delta_vs_S_naive"]),
        }
    (root / "selected_params.yaml").write_text(yaml.safe_dump(selected, sort_keys=False), encoding="utf-8")
    print(yaml.safe_dump(selected, sort_keys=False))


if __name__ == "__main__":
    main()
