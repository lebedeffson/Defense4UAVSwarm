#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from defense4uavswarm.v8_sim import add_features, metric_summary, read_json


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="")
    p.add_argument("--dataset-root", default="")
    p.add_argument("--detections", default="")
    p.add_argument("--setting", required=True)
    p.add_argument("--feature-modes", nargs="+", required=True)
    p.add_argument("--seeds", nargs="+", type=int, default=[11])
    p.add_argument("--matching-mode", default="coarse_class")
    p.add_argument("--ignore-policy", default="exclude_ignored")
    p.add_argument("--iou-threshold", type=float, default=0.5)
    p.add_argument("--output-dir", required=True)
    args = p.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if args.setting != "controlled_multiagent":
        write_skipped(out, args.setting)
        print(f"status=skipped setting={args.setting} output={out}")
        return

    manifest = read_json("data/custom_uav_swarm_v8/manifest.json")
    gt_all = pd.DataFrame(read_json("data/custom_uav_swarm_v8/gt_2d_boxes.json")["boxes"])
    det_all = pd.DataFrame(read_json("outputs/results/v8_custom_swarm/detections/combined_stress_detections.json")["detections"])
    gt = gt_all[gt_all["split"].eq("holdout")].copy()
    det = det_all[det_all["split"].eq("holdout")].copy()
    rows = []
    for seed in args.seeds:
        feat = add_features(seed_jitter(det, seed), {})
        for mode in args.feature_modes:
            accepted = accept_mode(feat, mode)
            row = metric_summary(gt, feat, accepted, mode, len(gt), frame_count(gt), manifest, runtime_base=0.001)
            row.update({"feature_mode": mode, "seed": seed, "setting": args.setting})
            rows.append(row)
    raw = pd.DataFrame(rows)
    raw.to_csv(out / "feature_ablation_raw.csv", index=False)
    summary = raw.groupby("feature_mode", as_index=False)[["F1", "FP", "FN", "precision", "recall", "false_new_tracks", "false_new_tracks_per_100_frames"]].mean()
    summary.to_csv(out / "feature_ablation_summary.csv", index=False)
    plot_metric(summary, "false_new_tracks", out / "fig_feature_ablation_false_new_ru.png")
    plot_metric(summary, "F1", out / "fig_feature_ablation_f1_ru.png")
    write_claim(out / "feature_ablation_claim_safe.md", summary)
    print(f"status=ok rows={len(raw)} output={out}")


def seed_jitter(det: pd.DataFrame, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    d = det.copy()
    d["confidence"] = (d["confidence"].astype(float) + rng.normal(0, 0.003, len(d))).clip(0, 1)
    return d


def frame_count(gt: pd.DataFrame) -> int:
    return int(gt[["scene_id", "frame_id"]].drop_duplicates().shape[0])


def accept_mode(d: pd.DataFrame, mode: str) -> pd.Series:
    m = mode.lower()
    c = d["c_i"].astype(float)
    k = d["k_i"].astype(float)
    s = d["s_i"].astype(float)
    t = (d["temporal_age"].astype(float) / 3.0).clip(0, 1)
    parts = {
        "c_only": c,
        "c_k": np.minimum(c, k),
        "c_temporal": np.minimum(c, np.maximum(t, 0.35)),
        "c_geometric": np.minimum(c, s),
        "c_k_temporal": np.minimum.reduce([c, k, np.maximum(t, 0.35)]),
        "c_k_geometric": np.minimum.reduce([c, k, s]),
        "c_temporal_geometric": np.minimum.reduce([c, np.maximum(t, 0.35), s]),
        "full": np.minimum.reduce([c, k, np.maximum(t, 0.35), s]),
    }
    q = pd.Series(parts.get(m, parts["full"]), index=d.index)
    conf_rw = d["confidence"].astype(float) * (0.6 + 0.4 * q)
    recovery = (d["temporal_age"].astype(float) >= 2) & (d["confidence"].astype(float) >= 0.10) if "temporal" in m or m == "full" else False
    return (conf_rw >= 0.50) | recovery


def plot_metric(summary: pd.DataFrame, metric: str, path: Path) -> None:
    plt.figure(figsize=(8, 4))
    s = summary.sort_values(metric, ascending=False if metric == "F1" else True)
    plt.bar(s["feature_mode"], s[metric])
    plt.xticks(rotation=35, ha="right")
    plt.ylabel(metric)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def write_claim(path: Path, summary: pd.DataFrame) -> None:
    best = summary.sort_values(["F1", "false_new_tracks"], ascending=[False, True]).iloc[0]
    ctemp = get(summary, "c_temporal")
    full = get(summary, "full")
    geom = "not available"
    if ctemp is not None and full is not None:
        geom = f"full - c_temporal: F1 {full['F1'] - ctemp['F1']:+.6f}, false_new {full['false_new_tracks'] - ctemp['false_new_tracks']:+.1f}"
    lines = [
        "# Feature Ablation Claim-Safe Notes",
        "",
        "Controlled simulation feature ablation; no VisDrone inter-agent consistency is claimed.",
        "",
        "```text",
        summary[["feature_mode", "F1", "false_new_tracks"]].to_string(index=False),
        "```",
        "",
        f"Best compromise by F1 then false_new: `{best['feature_mode']}`.",
        f"Geometric contribution diagnostic: {geom}.",
        "Safe claim: feature channels are ablated in the controlled setting; geometry can be described as a safety channel only if its delta is small.",
        "Forbidden claim: do not infer real-world inter-agent geometry from VisDrone single-camera results.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def get(df: pd.DataFrame, mode: str) -> pd.Series | None:
    hit = df[df["feature_mode"].eq(mode)]
    return None if hit.empty else hit.iloc[0]


def write_skipped(out: Path, setting: str) -> None:
    text = f"# Feature Ablation Claim-Safe Notes\n\nSetting `{setting}` skipped because final clean inter-agent feature ablation is only supported for controlled_multiagent in this script.\n"
    (out / "feature_ablation_claim_safe.md").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
