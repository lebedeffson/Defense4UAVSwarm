#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from defense4uavswarm.q1_visdrone import Q1Params, compute_metrics, feature_matrix, load_gt_protocol, make_chunk_split, train_rf


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-root", required=True)
    p.add_argument("--detections", required=True)
    p.add_argument("--feature-audit", default="outputs/results/q1_final_corrected/yolov8s_main/feature_audit.csv")
    p.add_argument("--label-budgets", nargs="+", type=float, required=True)
    p.add_argument("--seeds", nargs="+", type=int, required=True)
    p.add_argument("--safety-thresholds", nargs="+", type=float, required=True)
    p.add_argument("--output-dir", required=True)
    args = p.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    det = pd.read_csv(args.feature_audit)
    gt, _ = load_gt_protocol(args.dataset_root, sorted(det["sequence_id"].unique()))
    keys = det[["sequence_id", "frame_id"]].drop_duplicates()
    gt = gt.merge(keys, on=["sequence_id", "frame_id"], how="inner")
    chunks = make_chunk_split(gt)
    det = det.merge(chunks[["sequence_id", "frame_id", "chunk_id", "split"]], on=["sequence_id", "frame_id"], how="left")
    gt = gt.merge(chunks[["sequence_id", "frame_id", "chunk_id", "split"]], on=["sequence_id", "frame_id"], how="left")
    hold_det = det[det["split"].eq("holdout")].copy()
    hold_gt = gt[gt["split"].eq("holdout")].copy()
    rows = []
    selected = []
    for budget in args.label_budgets:
        for seed in args.seeds:
            chosen = select_chunks(chunks, budget, seed)
            train_det = det[det["chunk_id"].isin(chosen)].copy()
            if train_det.empty or train_det["eval_is_tp"].nunique() < 2:
                continue
            rf, tau = train_rf(train_det)
            rf_accept = rf.predict_proba(feature_matrix(hold_det))[:, 1] < tau
            rf_row = compute_metrics(hold_det, rf_accept, len(hold_gt), hold_gt[["sequence_id", "frame_id"]].drop_duplicates().shape[0], "rf")
            rf_row.update({"label_budget": budget, "seed": seed, "safety_threshold": np.nan})
            rows.append(rf_row)
            best = None
            for thr in args.safety_thresholds:
                q = np.minimum(hold_det["c_i"].astype(float), hold_det["k_i"].astype(float))
                accept = rf_accept & (q >= thr)
                row = compute_metrics(hold_det, accept, len(hold_gt), hold_gt[["sequence_id", "frame_id"]].drop_duplicates().shape[0], "rf_plus_trust_veto")
                row.update({"label_budget": budget, "seed": seed, "safety_threshold": thr})
                rows.append(row)
                score = row["F1"] - 0.0001 * row["false_new_tracks"]
                if best is None or score > best["score"]:
                    best = {"label_budget": budget, "seed": seed, "selected_threshold": thr, "score": score}
            if best:
                selected.append(best)
    raw = pd.DataFrame(rows)
    raw.to_csv(out / "rf_trust_hybrid_raw.csv", index=False)
    summary = raw.groupby(["label_budget", "method"], dropna=False, as_index=False).agg({"F1": ["mean", "std"], "false_new_tracks": ["mean", "std"]})
    summary.columns = ["_".join(c).rstrip("_") for c in summary.columns.to_flat_index()]
    summary.to_csv(out / "rf_trust_hybrid_summary.csv", index=False)
    pd.DataFrame(selected).to_csv(out / "rf_trust_hybrid_selected_thresholds.csv", index=False)
    plt.figure(figsize=(7, 4))
    for method, g in raw.groupby("method"):
        plt.scatter(g["false_new_tracks"], g["F1"], label=method, alpha=0.7)
    plt.xlabel("false_new")
    plt.ylabel("F1")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / "fig_rf_hybrid_tradeoff.png", dpi=160)
    plt.close()
    (out / "rf_trust_hybrid_summary.md").write_text("# RF + Trust Hybrid Summary\n\n" + summary.to_string(index=False) + "\n", encoding="utf-8")
    print(f"status=ok output={out / 'rf_trust_hybrid_summary.csv'}")


def select_chunks(chunks: pd.DataFrame, budget: float, seed: int) -> set[str]:
    import random

    cal = sorted(chunks[chunks["split"].eq("calibration")]["chunk_id"].unique())
    n = max(1, int(round(len(cal) * budget)))
    return set(random.Random(seed).sample(cal, min(n, len(cal))))


if __name__ == "__main__":
    main()
