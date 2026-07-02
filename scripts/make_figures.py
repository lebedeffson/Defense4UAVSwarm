#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--summary", default="outputs/results/summary_metrics.csv")
    p.add_argument("--out-dir", default="outputs/figures")
    args = p.parse_args()
    df = pd.read_csv(args.summary)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    atk = df[df["eps"].notna()]
    for metric in ["F1", "IDF1", "MOTA", "ASR"]:
        if metric in atk:
            plt.figure()
            for name, g in atk.groupby(["scenario", "t_norm"], dropna=False):
                plt.plot(g["eps"], g[metric], marker="o", label=str(name))
            plt.xlabel("eps")
            plt.ylabel(metric)
            plt.legend()
            plt.tight_layout()
            plt.savefig(out / f"{metric.lower()}_vs_eps.png", dpi=200)
            plt.close()

    metrics = [m for m in ["F1", "IDF1", "MOTA", "ASR"] if m in atk]
    if metrics:
        fig, axes = plt.subplots(2, 2, figsize=(10, 7))
        for ax, metric in zip(axes.ravel(), metrics):
            for name, g in atk.groupby(["scenario", "t_norm"], dropna=False):
                ax.plot(g["eps"], g[metric], marker="o", label=str(name))
            ax.set_xlabel("eps")
            ax.set_ylabel(metric)
        axes.ravel()[0].legend(fontsize=7)
        plt.tight_layout()
        plt.savefig(out / "metrics_vs_eps.png", dpi=200)
        plt.close(fig)

    s2 = df[df["scenario"] == "S2"]
    if len(s2):
        plt.figure()
        for name, g in s2.groupby("t_norm"):
            plt.plot(g["eps"], g["F1"], marker="o", label=name)
        plt.xlabel("eps")
        plt.ylabel("F1")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out / "t_norm_comparison.png", dpi=200)
        plt.close()

    matrix_path = Path(args.summary).with_name("research_matrix.csv")
    if matrix_path.exists():
        mdf = pd.read_csv(matrix_path)
        for metric, filename in [("F1", "f1_vs_eps_by_model.png"), ("ASR", "asr_vs_eps_by_model.png")]:
            plt.figure()
            data = mdf[(mdf["class_group"] == "all") & (mdf["scenario"].isin(["S1", "S_naive", "S2"]))]
            if metric in data:
                for key, g in data.groupby(["model_name", "scenario", "t_norm"], dropna=False):
                    plt.plot(g["eps"], g[metric], marker="o", label=str(key))
                plt.xlabel("eps")
                plt.ylabel(metric)
                plt.legend(fontsize=7)
                plt.tight_layout()
                plt.savefig(out / filename, dpi=200)
                plt.close()

        plt.figure()
        data = mdf[mdf["scenario"].isin(["S1", "S2"])]
        if len(data):
            pivot = data.groupby(["class_group", "scenario"])["F1"].max().unstack()
            pivot.plot(kind="bar")
            plt.ylabel("F1")
            plt.tight_layout()
            plt.savefig(out / "class_group_comparison.png", dpi=200)
            plt.close()

        ablation_path = matrix_path.with_name("ablation_summary.csv")
        if ablation_path.exists():
            adf = pd.read_csv(ablation_path)
            plt.figure()
            data = adf[adf["class_group"] == "all"]
            if len(data):
                data.groupby("filter_variant")["F1"].mean().sort_values().plot(kind="barh")
                plt.xlabel("F1")
                plt.tight_layout()
                plt.savefig(out / "ablation_variants.png", dpi=200)
                plt.close()


if __name__ == "__main__":
    main()
