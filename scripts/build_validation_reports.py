#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from defense4uavswarm.class_groups import filter_by_group, load_class_groups, normalized_names
from defense4uavswarm.config import load_config
from defense4uavswarm.matrix import task_dataset
from defense4uavswarm.metrics.asr import attack_success_breakdown


def read(path: Path) -> pd.DataFrame | None:
    return pd.read_csv(path) if path.exists() else None


def group_for_class(name: str, cfg: dict) -> str:
    groups = load_class_groups(cfg["class_groups_file"])
    aliases = groups.get("aliases", {})
    name = aliases.get(str(name).lower(), str(name).lower())
    for group, names in groups["class_groups"].items():
        if group != "all" and name in names:
            return group
    return "other"


def class_counts(df: pd.DataFrame, cfg: dict, prefix: str) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["class_name", prefix])
    tmp = df.copy()
    if "accepted" in tmp:
        tmp = tmp[tmp["accepted"] == True]
    tmp["class_name_norm"] = normalized_names(tmp, load_class_groups(cfg["class_groups_file"]).get("aliases", {}))
    out = tmp.groupby("class_name_norm").size().reset_index(name=prefix)
    return out.rename(columns={"class_name_norm": "class_name"})


def merge_counts(gt, s0, s1, s2, cfg) -> pd.DataFrame:
    parts = [
        class_counts(gt, cfg, "num_gt"),
        class_counts(s0, cfg, "num_pred_s0"),
        class_counts(s1, cfg, "num_pred_s1"),
        class_counts(s2, cfg, "num_pred_s2"),
    ]
    out = parts[0]
    for part in parts[1:]:
        out = out.merge(part, on="class_name", how="outer")
    out = out.fillna(0)
    out["class_group"] = out["class_name"].map(lambda x: group_for_class(x, cfg))
    for col in ["num_gt", "num_pred_s0", "num_pred_s1", "num_pred_s2"]:
        out[col] = out[col].astype(int)
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--tasks", nargs="+", default=["vid"])
    p.add_argument("--models", nargs="+", default=["yolov8n", "yolov8s"])
    p.add_argument("--eps", nargs="+", type=float, default=[0.004, 0.008])
    p.add_argument("--class-groups", nargs="+", default=["all", "vru", "vehicles"])
    p.add_argument("--limit-sequences", type=int, default=3)
    p.add_argument("--output-dir", default=None)
    args = p.parse_args()
    cfg = load_config(args.config)
    if args.output_dir:
        cfg["outputs"]["results_dir"] = args.output_dir
    results = Path(cfg["outputs"]["results_dir"])
    asr_rows, class_rows = [], []
    for task in args.tasks:
        _, ds = task_dataset(cfg, task)
        sequences = ds.sequence_ids()[: args.limit_sequences]
        gt = ds.all_annotations(sequences)
        for model in args.models:
            s0 = read(results / f"{task}_{model}_s0_baseline.csv")
            if s0 is None:
                continue
            fgsm_loss = s0["fgsm_loss"].dropna().iloc[0] if "fgsm_loss" in s0 and len(s0["fgsm_loss"].dropna()) else "class_only"
            for eps in args.eps:
                s1 = read(results / f"{task}_{model}_s1_fgsm_eps_{eps}.csv")
                if s1 is None:
                    continue
                if "fgsm_loss" in s1 and len(s1["fgsm_loss"].dropna()):
                    fgsm_loss = s1["fgsm_loss"].dropna().iloc[0]
                scenarios = [("S1", None, s1), ("S_naive", "confidence", read(results / f"{task}_{model}_s_naive_eps_{eps}.csv"))]
                for path in sorted(results.glob(f"{task}_{model}_s2_*_eps_{eps}.csv")):
                    t_norm = path.name.split("_s2_", 1)[1].split("_eps_", 1)[0]
                    scenarios.append(("S2", t_norm, read(path)))
                first_s2 = next((df for sc, _, df in scenarios if sc == "S2" and df is not None), None)
                dist = merge_counts(gt, s0, s1, first_s2, cfg)
                dist.insert(0, "eps", eps)
                dist.insert(0, "fgsm_loss", fgsm_loss)
                dist.insert(0, "model_name", model)
                dist.insert(0, "task", task)
                class_rows.append(dist)
                for scenario, t_norm, frame in scenarios:
                    if frame is None:
                        continue
                    for group in args.class_groups:
                        b = attack_success_breakdown(
                            filter_by_group(gt, cfg, group),
                            filter_by_group(s0, cfg, group),
                            filter_by_group(frame, cfg, group),
                        )
                        asr_rows.append(
                            {
                                "model_name": model,
                                "fgsm_loss": fgsm_loss,
                                "scenario": scenario,
                                "eps": eps,
                                "class_group": group,
                                "t_norm": t_norm,
                                **b,
                            }
                        )
    if asr_rows:
        pd.DataFrame(asr_rows).to_csv(results / "asr_breakdown.csv", index=False)
    if class_rows:
        pd.concat(class_rows, ignore_index=True).to_csv(results / "class_distribution.csv", index=False)
    print(results / "asr_breakdown.csv")
    print(results / "class_distribution.csv")


if __name__ == "__main__":
    main()
