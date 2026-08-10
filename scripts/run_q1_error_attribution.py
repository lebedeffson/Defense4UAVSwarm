#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from defense4uavswarm.q1_visdrone import adaptive_geometry_acceptance


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-root", required=True)
    p.add_argument("--detections", required=True)
    p.add_argument("--method", default="geometry_dynamic_adaptive_balanced")
    p.add_argument("--feature-audit", default="outputs/results/q1_final_corrected/yolov8s_main/feature_audit.csv")
    p.add_argument("--selected-configs", default="outputs/results/q1_improvement/yolov8s_sweep/selected_configs.yaml")
    p.add_argument("--output-dir", required=True)
    args = p.parse_args()
    import yaml

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    det = pd.read_csv(args.feature_audit)
    cfgs = yaml.safe_load(Path(args.selected_configs).read_text(encoding="utf-8"))
    cfg = cfgs["selected_balanced"] if args.method.endswith("adaptive_balanced") else cfgs.get("selected_false_new_safe", cfgs["selected_balanced"])
    accepted = adaptive_geometry_acceptance(det, cfg)
    det = det.copy()
    det["accepted"] = accepted
    det["limiting_feature"] = det.apply(reason, axis=1)
    rejected = det[~det["accepted"]].copy()
    rejected.groupby("limiting_feature").size().reset_index(name="count").to_csv(out / "rejected_candidates_by_reason.csv", index=False)
    det[(~det["accepted"]) & det["eval_is_tp"].astype(bool)].groupby("limiting_feature").size().reset_index(name="count").to_csv(out / "false_negative_attribution.csv", index=False)
    det[det["accepted"] & (~det["eval_is_tp"].astype(bool))].groupby("limiting_feature").size().reset_index(name="count").to_csv(out / "false_positive_attribution.csv", index=False)
    dist = det.groupby("limiting_feature").size().reset_index(name="count")
    dist.to_csv(out / "limiting_feature_distribution.csv", index=False)
    plot_pie(dist, out / "fig_limiting_feature_pie.png")
    plot_bar(out / "false_negative_attribution.csv", out / "false_positive_attribution.csv", out / "fig_error_attribution_bar.png")
    dominant = dist.sort_values("count", ascending=False).iloc[0]["limiting_feature"] if len(dist) else "none"
    (out / "error_attribution_summary.md").write_text(f"# Error Attribution Summary\n\nDominant limiting feature: `{dominant}`.\n\nSingle-camera VisDrone setting uses temporal/geometric proxy, not inter-agent consistency.\n", encoding="utf-8")
    print(f"status=ok output={out}")


def reason(row) -> str:
    c = float(row.get("c_i", row.get("confidence", 0)))
    k = float(row.get("k_i", 0))
    age = float(row.get("temporal_age", 0))
    vals = {
        "detector_confidence_min": c,
        "kinematic_consistency_min": k,
        "temporal_memory_min": min(1.0, age / 3.0),
    }
    ordered = sorted(vals.items(), key=lambda x: x[1])
    if len(ordered) > 1 and abs(ordered[0][1] - ordered[1][1]) < 0.05:
        return "mixed_or_tie"
    return ordered[0][0]


def plot_pie(dist: pd.DataFrame, path: Path) -> None:
    plt.figure(figsize=(5, 5))
    plt.pie(dist["count"], labels=dist["limiting_feature"], autopct="%1.1f%%")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def plot_bar(fn_path: Path, fp_path: Path, path: Path) -> None:
    frames = []
    for label, p in [("FN", fn_path), ("FP", fp_path)]:
        if p.exists():
            d = pd.read_csv(p)
            d["type"] = label
            frames.append(d)
    if not frames:
        return
    df = pd.concat(frames)
    piv = df.pivot_table(index="limiting_feature", columns="type", values="count", aggfunc="sum").fillna(0)
    piv.plot(kind="bar", figsize=(7, 4))
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


if __name__ == "__main__":
    main()
