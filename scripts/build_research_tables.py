#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def best_s2(df: pd.DataFrame) -> pd.DataFrame:
    s2 = df[df["scenario"] == "S2"].copy()
    if s2.empty:
        return s2
    return s2.sort_values("F1", ascending=False).groupby(["model_name", "eps", "class_group"], as_index=False).head(1)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--matrix", default="outputs/results/research_matrix.csv")
    p.add_argument("--ablation", default="outputs/results/ablation_summary.csv")
    p.add_argument("--out-dir", default="outputs/results")
    args = p.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.matrix)
    s2b = best_s2(df)

    rows = []
    for eps in [0.004, 0.008]:
        sub = df[(df["eps"].isin([0.0, eps])) & (df["class_group"] == "all")]
        for model in sorted(sub["model_name"].dropna().unique()):
            s0 = sub[(sub.model_name == model) & (sub.scenario == "S0")]["F1"].max()
            s1 = sub[(sub.model_name == model) & (sub.scenario == "S1") & (sub.eps == eps)]["F1"].max()
            sn = sub[(sub.model_name == model) & (sub.scenario == "S_naive") & (sub.eps == eps)]["F1"].max()
            s2 = s2b[(s2b.model_name == model) & (s2b.eps == eps) & (s2b.class_group == "all")]["F1"].max()
            rows.append({"eps": eps, "model_name": model, "S0_F1": s0, "S1_F1": s1, "S_naive_F1": sn, "S2_best_F1": s2, "gain_vs_S1": s2 - s1, "gain_vs_S_naive": s2 - sn})
    pd.DataFrame(rows).to_csv(out / "table1_model_robustness.csv", index=False)

    rows = []
    for group in ["all", "vru", "vehicles"]:
        sub = df[df["class_group"] == group]
        s0 = sub[sub.scenario == "S0"]["F1"].max()
        s1 = sub[sub.scenario == "S1"]["F1"].max()
        s2 = s2b[s2b.class_group == group]
        rows.append({
            "class_group": group,
            "S0_F1": s0,
            "S1_F1": s1,
            "S2_best_F1": s2["F1"].max() if len(s2) else None,
            "FP_reduction": sub[sub.scenario == "S1"]["FP"].mean() - s2["FP"].mean() if len(s2) else None,
            "FN_change": s2["FN"].mean() - sub[sub.scenario == "S1"]["FN"].mean() if len(s2) else None,
            "ASR_reduction": sub[sub.scenario == "S1"]["ASR"].mean() - s2["ASR"].mean() if len(s2) else None,
        })
    pd.DataFrame(rows).to_csv(out / "table2_class_robustness.csv", index=False)

    df[df["scenario"] == "S2"].groupby("t_norm")[["F1", "IDF1", "ASR", "latency_ms"]].mean().reset_index().to_csv(out / "table3_tnorm_comparison.csv", index=False)

    ab = pd.read_csv(args.ablation) if Path(args.ablation).exists() else pd.DataFrame()
    if len(ab):
        ab.groupby("filter_variant")[["F1", "IDF1", "FP", "FN", "ASR", "delta_F1_vs_S1", "delta_IDF1_vs_S1", "delta_ASR_vs_S1"]].mean().reset_index().to_csv(out / "table4_ablation_features.csv", index=False)

    rows = []
    for eps, sub in df[df["class_group"] == "all"].groupby("eps"):
        if eps == 0:
            continue
        s2 = s2b[(s2b.eps == eps) & (s2b.class_group == "all")]
        rows.append({
            "eps": eps,
            "S1_F1": sub[sub.scenario == "S1"]["F1"].max(),
            "S_naive_F1": sub[sub.scenario == "S_naive"]["F1"].max(),
            "S2_F1": s2["F1"].max() if len(s2) else None,
            "S1_ASR": sub[sub.scenario == "S1"]["ASR"].mean(),
            "S2_ASR": s2["ASR"].mean() if len(s2) else None,
        })
    pd.DataFrame(rows).to_csv(out / "table5_eps_sensitivity.csv", index=False)


if __name__ == "__main__":
    main()
