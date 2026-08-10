#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import yaml

from defense4uavswarm.q1_visdrone import add_adaptive_metrics, evaluate_methods, load_gt_protocol


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-root", required=True)
    p.add_argument("--detections", required=True)
    p.add_argument("--methods", nargs="+", required=True)
    p.add_argument("--feature-audit", default="outputs/results/q1_final_corrected/yolov8s_main/feature_audit.csv")
    p.add_argument("--selected-configs", default="outputs/results/q1_improvement/yolov8s_sweep/selected_configs.yaml")
    p.add_argument("--output-dir", required=True)
    args = p.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    det = pd.read_csv(args.feature_audit)
    gt, _ = load_gt_protocol(args.dataset_root, sorted(det["sequence_id"].unique()))
    keys = det[["sequence_id", "frame_id"]].drop_duplicates()
    gt = gt.merge(keys, on=["sequence_id", "frame_id"], how="inner")
    cfgs = yaml.safe_load(Path(args.selected_configs).read_text(encoding="utf-8"))
    frame_counts = det.groupby(["sequence_id", "frame_id"])["det_id"].count().reset_index(name="density")
    frame_counts["density_bin"] = pd.qcut(frame_counts["density"], 3, labels=["low_density", "medium_density", "high_density"], duplicates="drop")
    area_q = det["bbox_area"].quantile([1 / 3, 2 / 3]).tolist()
    det["size_bin"] = pd.cut(det["bbox_area"], [-1, area_q[0], area_q[1], float("inf")], labels=["small", "medium", "large"])
    density = eval_groups(det.merge(frame_counts, on=["sequence_id", "frame_id"]), gt, args.methods, cfgs, "density_bin")
    size = eval_groups(det, gt, args.methods, cfgs, "size_bin")
    cross = eval_groups(det.merge(frame_counts, on=["sequence_id", "frame_id"]), gt, args.methods, cfgs, ["density_bin", "size_bin"])
    density.to_csv(out / "density_metrics.csv", index=False)
    size.to_csv(out / "size_metrics.csv", index=False)
    cross.to_csv(out / "density_size_cross_metrics.csv", index=False)
    plot(density, "density_bin", out / "fig_density_tradeoff.png")
    plot(size, "size_bin", out / "fig_size_tradeoff.png")
    (out / "density_size_summary.md").write_text("# Density / Size Analysis\n\n" + density.to_string(index=False) + "\n\n" + size.to_string(index=False) + "\n", encoding="utf-8")
    print(f"status=ok output={out}")


def eval_groups(det: pd.DataFrame, gt: pd.DataFrame, methods: list[str], cfgs: dict, group_cols) -> pd.DataFrame:
    if isinstance(group_cols, str):
        group_cols = [group_cols]
    rows = []
    for key, sub_det in det.groupby(group_cols, observed=False):
        sub_gt = gt.merge(sub_det[["sequence_id", "frame_id"]].drop_duplicates(), on=["sequence_id", "frame_id"], how="inner")
        base = [m for m in methods if m not in {"geometry_dynamic_adaptive_balanced", "geometry_dynamic_false_new_safe"}]
        summary = evaluate_methods(sub_det, sub_gt, base)
        extra = []
        if "geometry_dynamic_adaptive_balanced" in methods:
            extra.append(add_adaptive_metrics(sub_det, sub_gt, cfgs["selected_balanced"], "geometry_dynamic_adaptive_balanced"))
        if "geometry_dynamic_false_new_safe" in methods:
            extra.append(add_adaptive_metrics(sub_det, sub_gt, cfgs["selected_false_new_safe"], "geometry_dynamic_false_new_safe"))
        if extra:
            summary = pd.concat([summary, pd.DataFrame(extra)], ignore_index=True)
        if not isinstance(key, tuple):
            key = (key,)
        for col, val in zip(group_cols, key):
            summary[col] = val
        rows.extend(summary.to_dict("records"))
    return pd.DataFrame(rows)


def plot(df: pd.DataFrame, group_col: str, path: Path) -> None:
    plt.figure(figsize=(8, 4))
    for method, g in df.groupby("method"):
        plt.plot(g[group_col].astype(str), g["F1"], marker="o", label=method)
    plt.ylabel("F1")
    plt.xticks(rotation=25, ha="right")
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


if __name__ == "__main__":
    main()
