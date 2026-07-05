#!/usr/bin/env python
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from train_fp_classifier_baseline import features, predict_score


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", nargs="+", type=int, required=True)
    p.add_argument("--learned-fp-model", required=True)
    p.add_argument("--learned-fp-threshold", required=True)
    p.add_argument("--seed-results-root", default="outputs/results/v4_2_review_response/seeds")
    p.add_argument("--output-root", required=True)
    args = p.parse_args()
    out = Path(args.output_root)
    out.mkdir(parents=True, exist_ok=True)
    model_payload = pickle.load(open(args.learned_fp_model, "rb"))
    threshold = yaml.safe_load(open(args.learned_fp_threshold, "r", encoding="utf-8"))
    clf = model_payload["model"]
    model_name = model_payload["model_name"]
    tau = float(threshold["classifier_threshold"])
    base_conf = float(threshold.get("base_confidence_threshold", 0.6))
    rows = []
    for seed in args.seeds:
        seed_dir = Path(args.seed_results_root) / f"seed_{seed}"
        summary = pd.read_csv(seed_dir / "summary_metrics.csv")
        for scenario in ["S_naive", "S2_tnorm_soft"]:
            row = summary[summary.scenario.eq(scenario)].sort_values(["F1", "IDF1"], ascending=False).iloc[0].to_dict()
            row = {k: row.get(k) for k in ["scenario", "TP", "FP", "FN", "precision", "recall", "F1", "IDF1", "false_new_tracks", "false_new_tracks_per_100_frames"]}
            row["seed"] = seed
            rows.append(row)
        audit = pd.read_csv(seed_dir / "swarm_feature_audit.csv")
        score = predict_score(clf, model_name, audit)
        expected_gt = int(summary[summary.scenario.eq("S_naive")].sort_values(["F1", "IDF1"], ascending=False).iloc[0]["TP"] + summary[summary.scenario.eq("S_naive")].sort_values(["F1", "IDF1"], ascending=False).iloc[0]["FN"])
        accepted_base = audit["confidence"].astype(float).to_numpy() >= base_conf
        is_new = audit["track_status"].eq("new_candidate").to_numpy()
        y_tp = audit["eval_is_tp"].astype(bool).to_numpy()
        accepted = accepted_base & ~(is_new & (score >= tau))
        learned = metrics(accepted, y_tp, expected_gt, is_new, audit)
        learned.update({"seed": seed, "scenario": "S2_learned_fp_gate", "model": model_name, "classifier_threshold": tau})
        rows.append(learned)
    result = pd.DataFrame(rows)
    result = add_deltas(result)
    result.to_csv(out / "learned_fp_tracking_summary.csv", index=False)
    mean_std(result).to_csv(out / "learned_fp_mean_std_summary.csv", index=False)


def metrics(accepted: np.ndarray, y_tp: np.ndarray, expected_gt: int, is_new: np.ndarray, audit: pd.DataFrame) -> dict:
    tp = int((accepted & y_tp).sum())
    fp = int((accepted & ~y_tp).sum())
    fn = max(0, expected_gt - tp)
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-9, precision + recall)
    frames = audit[["sequence_id", "frame_id"]].drop_duplicates().shape[0]
    false_new = int((accepted & is_new & ~y_tp).sum())
    return {
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "precision": precision,
        "recall": recall,
        "F1": f1,
        "IDF1": f1,
        "false_new_tracks": false_new,
        "false_new_tracks_per_100_frames": false_new / max(1, frames) * 100.0,
    }


def add_deltas(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for seed, group in out.groupby("seed"):
        base = group[group.scenario.eq("S_naive")].iloc[0]
        mask = out["seed"].eq(seed)
        out.loc[mask, "FP_delta_vs_S_naive"] = out.loc[mask, "FP"] - base["FP"]
        out.loc[mask, "FN_delta_vs_S_naive"] = out.loc[mask, "FN"] - base["FN"]
        out.loc[mask, "F1_delta_vs_S_naive"] = out.loc[mask, "F1"] - base["F1"]
        out.loc[mask, "false_new_tracks_delta_vs_S_naive"] = out.loc[mask, "false_new_tracks"] - base["false_new_tracks"]
    return out


def mean_std(frame: pd.DataFrame) -> pd.DataFrame:
    metrics = ["FP", "FN", "F1", "IDF1", "false_new_tracks", "false_new_tracks_per_100_frames"]
    rows = []
    piv = frame.pivot(index="seed", columns="scenario")
    for metric in metrics:
        row = {"metric": metric}
        for scenario in ["S_naive", "S2_tnorm_soft", "S2_learned_fp_gate"]:
            row[f"{scenario}_mean"] = piv[metric][scenario].mean()
            row[f"{scenario}_std"] = piv[metric][scenario].std(ddof=1)
        row["S2_tnorm_soft_delta_mean"] = (piv[metric]["S2_tnorm_soft"] - piv[metric]["S_naive"]).mean()
        row["S2_learned_fp_gate_delta_mean"] = (piv[metric]["S2_learned_fp_gate"] - piv[metric]["S_naive"]).mean()
        rows.append(row)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    main()
