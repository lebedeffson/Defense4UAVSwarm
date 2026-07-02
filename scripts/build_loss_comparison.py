#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--inputs", nargs="+", default=["outputs/results/class_only", "outputs/results/proxy_conf_box"])
    p.add_argument("--out", default="outputs/results/loss_comparison.csv")
    p.add_argument("--summary-out", default="outputs/results/loss_comparison_summary.csv")
    args = p.parse_args()

    frames = []
    for item in args.inputs:
        root = Path(item)
        research = read(root / "research_matrix.csv")
        if research.empty:
            continue
        asr = read(root / "asr_breakdown.csv")
        if not asr.empty:
            keys = ["fgsm_loss", "model_name", "scenario", "eps", "class_group", "t_norm"]
            if "fgsm_loss" not in asr:
                asr["fgsm_loss"] = research["fgsm_loss"].dropna().iloc[0]
            cols = keys + ["ASR_frame", "asr_miss", "asr_fp", "asr_break", "asr_idsw"]
            research = research.merge(asr[[c for c in cols if c in asr]], on=[k for k in keys if k in research and k in asr], how="left")
        keep = [
            "fgsm_loss",
            "model_name",
            "eps",
            "scenario",
            "class_group",
            "t_norm",
            "F1",
            "precision",
            "recall",
            "IDF1",
            "MOTA",
            "IDSW",
            "FP",
            "FN",
            "ASR_frame",
            "asr_miss",
            "asr_fp",
            "asr_break",
            "asr_idsw",
            "latency_ms",
        ]
        frames.append(research[[c for c in keep if c in research]])
    if not frames:
        raise RuntimeError("No research_matrix.csv inputs found")
    out = pd.concat(frames, ignore_index=True)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)

    rows = []
    all_group = out[out["class_group"] == "all"]
    for (loss, model, eps), g in all_group.groupby(["fgsm_loss", "model_name", "eps"]):
        s0 = all_group[(all_group["fgsm_loss"] == loss) & (all_group["model_name"] == model) & (all_group["scenario"] == "S0")]
        s1 = g[g["scenario"] == "S1"]
        s2 = g[g["scenario"] == "S2"].sort_values("F1", ascending=False)
        if s0.empty or s1.empty or s2.empty:
            continue
        rows.append(
            {
                "fgsm_loss": loss,
                "model_name": model,
                "eps": eps,
                "S0_F1": float(s0.iloc[0]["F1"]),
                "S1_F1": float(s1.iloc[0]["F1"]),
                "S2_best_F1": float(s2.iloc[0]["F1"]),
                "S1_IDF1": float(s1.iloc[0]["IDF1"]) if pd.notna(s1.iloc[0]["IDF1"]) else None,
                "S2_best_IDF1": float(s2.iloc[0]["IDF1"]) if pd.notna(s2.iloc[0]["IDF1"]) else None,
                "S2_vs_S1_F1": float(s2.iloc[0]["F1"] - s1.iloc[0]["F1"]),
                "best_t_norm": s2.iloc[0]["t_norm"],
            }
        )
    pd.DataFrame(rows).to_csv(args.summary_out, index=False)
    print(out_path)
    print(args.summary_out)


if __name__ == "__main__":
    main()
