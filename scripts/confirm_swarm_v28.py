#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
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
    split = "calibration"
    metadata_path = root / "metadata.json"
    if metadata_path.exists():
        split = json.loads(metadata_path.read_text(encoding="utf-8")).get("split", split)
    delta_name = "holdout_delta_vs_s_naive.csv" if split == "holdout" else "full_calibration_delta_vs_s_naive.csv"
    delta.to_csv(root / delta_name, index=False)

    candidate = delta[delta["scenario"] == "S2_tnorm_soft"]
    raw_candidate = best_row(summary, "S2_tnorm_soft")
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
        source_stage = _source_stage(root, split)
        selected = {
            "selection_status": "selected" if strict else ("tradeoff_selected" if soft else "not_selected"),
            "holdout_allowed": bool(strict or soft),
            "success_flag": bool(strict or soft) if split == "holdout" else None,
            "selected_scenario": "S2_tnorm_soft",
            "selected_t_norm": _value(raw_candidate, "t_norm", "min"),
            "selected_reweight_mode": _value(raw_candidate, "reweight_mode", "multiplicative_floor"),
            "selected_q_floor": _value(raw_candidate, "q_floor", 0.6),
            "selected_q_hard_min": _value(raw_candidate, "q_hard_min", 0.0),
            "selected_new_track_threshold": _value(raw_candidate, "new_track_threshold", 0.4),
            "selected_existing_track_threshold": _value(raw_candidate, "existing_track_threshold", 0.05),
            "source_stage": source_stage,
            "reason": (
                "holdout_confirms_fp_reduction_without_recall_loss"
                if split == "holdout" and strict
                else "holdout_tradeoff_candidate"
                if split == "holdout" and soft
                else "full_calibration_confirms_fp_reduction_without_recall_loss"
                if strict
                else "full_calibration_tradeoff_candidate"
                if soft
                else "compact_calibration_did_not_generalize"
            ),
            "FP_delta_vs_S_naive": int(cand["FP_delta_vs_S_naive"]),
            "FN_delta_vs_S_naive": int(cand["FN_delta_vs_S_naive"]),
            "F1_delta_vs_S_naive": float(cand["F1_delta_vs_S_naive"]),
        }
    (root / "selected_params.yaml").write_text(yaml.safe_dump(selected, sort_keys=False), encoding="utf-8")
    print(yaml.safe_dump(selected, sort_keys=False))


def _value(row: pd.Series, key: str, default):
    value = row.get(key, default)
    if pd.isna(value):
        return default
    return value.item() if hasattr(value, "item") else value


def _source_stage(root: Path, split: str) -> str:
    root_text = str(root).lower()
    if "swarm_v3" in root_text or "newgate" in root_text:
        return "v3.0_newtrack_gate_holdout" if split == "holdout" else "v3.0_newtrack_gate_calibration"
    return "v2.9_holdout" if split == "holdout" else "v2.8_full_calibration"


if __name__ == "__main__":
    main()
