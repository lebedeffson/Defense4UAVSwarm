#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from defense4uavswarm.v8_sim import add_features, metric_summary, read_json
from defense4uavswarm.v9_geomdyn import load_config, prepare_v9_features, v9_acceptance


DEFAULT_DETECTIONS = "outputs/results/v8_custom_swarm/detections/combined_stress_detections.json"
DEFAULT_GT = "data/custom_uav_swarm_v8/gt_2d_boxes.json"
DEFAULT_MANIFEST = "data/custom_uav_swarm_v8/manifest.json"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="outputs/results/custom_uav_swarm")
    p.add_argument("--methods", nargs="+", required=True)
    p.add_argument("--seeds", nargs="+", type=int, required=True)
    p.add_argument("--agent-counts", nargs="+", type=int, default=[3])
    p.add_argument("--output-dir", required=True)
    args = p.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    source = resolve_source(Path(args.input))
    manifest = read_json(source["manifest"])
    gt_all = pd.DataFrame(read_json(source["gt"])["boxes"])
    det_all = pd.DataFrame(read_json(source["detections"])["detections"])
    det_all = det_all[det_all["split"].eq("holdout")].copy()
    gt_all = gt_all[gt_all["split"].eq("holdout")].copy()
    cfg = load_config()

    rows: list[dict[str, Any]] = []
    for agent_count in args.agent_counts:
        agents = sorted(det_all["agent_id"].unique())[:agent_count]
        gt = gt_all[gt_all["agent_id"].isin(agents)].copy()
        det_base = det_all[det_all["agent_id"].isin(agents)].copy()
        for seed in args.seeds:
            det = deterministic_seed_replay(det_base, seed)
            feat = add_features(det, {})
            v9_feat = prepare_v9_features(det, manifest, cfg)
            for method in args.methods:
                accepted = acceptance_for(method, feat, v9_feat, cfg)
                if accepted is None:
                    continue
                source_feat = v9_feat if method.lower() == "s2_v9_selected" else feat
                row = metric_summary(gt, source_feat, accepted, method, len(gt), frame_count(gt), manifest, runtime_base=0.0)
                row.pop("runtime_ms_per_frame", None)
                row.update(
                    {
                        "method": method,
                        "seed": seed,
                        "agent_count": agent_count,
                        "false_new_per_100_frames": row["false_new_tracks_per_100_frames"],
                        "map_extra_entries": row["false_new_tracks"],
                        "extra_messages_due_to_false_tracks": row["false_new_tracks"] * max(0, agent_count - 1),
                        "notes": f"controlled replay from {source['label']}; not cooperative perception SOTA",
                    }
                )
                rows.append(row)
    raw = pd.DataFrame(rows)
    raw.to_csv(out / "simple_fusion_raw.csv", index=False)
    by_seed = raw.copy()
    by_seed.to_csv(out / "simple_fusion_by_seed.csv", index=False)
    summary = summarize(raw)
    summary.to_csv(out / "simple_fusion_summary.csv", index=False)
    plot_tradeoff(summary, out / "fig_simple_fusion_tradeoff_ru.png")
    write_claim_safe(out / "simple_fusion_claim_safe.md", summary, source["label"])
    print(f"status=ok rows={len(raw)} output={out}")


def resolve_source(input_path: Path) -> dict[str, str]:
    candidates = [
        {
            "label": str(input_path),
            "manifest": str(input_path / "manifest.json"),
            "gt": str(input_path / "gt_2d_boxes.json"),
            "detections": str(input_path / "detections.json"),
        },
        {"label": "custom_uav_swarm_v8", "manifest": DEFAULT_MANIFEST, "gt": DEFAULT_GT, "detections": DEFAULT_DETECTIONS},
    ]
    for c in candidates:
        if Path(c["manifest"]).exists() and Path(c["gt"]).exists() and Path(c["detections"]).exists():
            return c
    raise FileNotFoundError("Cannot find controlled simulation manifest/gt/detections")


def deterministic_seed_replay(det: pd.DataFrame, seed: int) -> pd.DataFrame:
    # Keep the same simulation universe; seed only applies a tiny deterministic confidence jitter
    # for repeated diagnostic runs, not a new dataset.
    rng = np.random.default_rng(seed)
    d = det.copy()
    jitter = rng.normal(0.0, 0.003, len(d))
    d["confidence"] = (d["confidence"].astype(float) + jitter).clip(0, 1)
    return d


def frame_count(gt: pd.DataFrame) -> int:
    return int(gt[["scene_id", "frame_id"]].drop_duplicates().shape[0]) if len(gt) else 0


def fusion_scores(det: pd.DataFrame) -> pd.DataFrame:
    mean = np.zeros(len(det), dtype=float)
    maxv = np.zeros(len(det), dtype=float)
    support = np.zeros(len(det), dtype=int)
    for _, group in det.groupby(["scene_id", "frame_id", "class_id"], sort=False):
        idx = group.index.to_numpy()
        worlds = np.asarray(group["world_center"].tolist(), dtype=float)
        conf = group["confidence"].to_numpy(dtype=float)
        agents = group["agent_id"].to_numpy()
        for local_i, row_idx in enumerate(idx):
            dist = np.linalg.norm(worlds[:, :2] - worlds[local_i, :2], axis=1)
            mask = (dist <= 10.0) & (agents != agents[local_i])
            vals = np.r_[conf[local_i], conf[mask]]
            mean[row_idx] = float(vals.mean())
            maxv[row_idx] = float(vals.max())
            support[row_idx] = int(mask.sum())
    return pd.DataFrame({"fusion_mean_conf": mean, "fusion_max_conf": maxv, "fusion_support": support}, index=det.index)


def acceptance_for(method: str, feat: pd.DataFrame, v9_feat: pd.DataFrame, cfg: dict[str, Any]) -> pd.Series | None:
    m = method.lower()
    if m == "naive_union":
        return feat["confidence"].astype(float) >= 0.30
    if m == "iou_fusion_only":
        return (feat["support_count"].astype(float) >= 1) & (feat["confidence"].astype(float) >= 0.30)
    if m in {"mean_confidence_fusion", "max_confidence_fusion"}:
        scores = fusion_scores(feat)
        col = "fusion_mean_conf" if m.startswith("mean") else "fusion_max_conf"
        return (scores[col] >= 0.42) & (scores["fusion_support"] >= 1)
    if m == "bayesian_existence_filter":
        return bayesian_logodds(feat)
    if m == "s2_tnorm_temporal":
        q = np.minimum.reduce([feat["c_i"], feat["k_i"], feat["s_i"]])
        conf_rw = feat["confidence"] * (0.6 + 0.4 * q)
        return (conf_rw >= 0.50) | ((feat["temporal_age"] >= 2) & (feat["confidence"] >= 0.10))
    if m == "s2_v9_selected":
        return v9_acceptance(v9_feat, "s2_v9_selected", cfg)[0]
    return None


def bayesian_logodds(det: pd.DataFrame) -> pd.Series:
    accepted = pd.Series(False, index=det.index)
    for _, group in det.groupby(["scene_id", "agent_id", "class_id", "_track_key"], sort=False):
        logodds = 0.0
        for idx, row in group.sort_values("frame_id").iterrows():
            if float(row.confidence) >= 0.50:
                logodds += 0.9
            elif float(row.confidence) >= 0.30:
                logodds += 0.35
            else:
                logodds -= 0.25
            accepted.loc[idx] = logodds >= 1.1
    return accepted


def summarize(raw: pd.DataFrame) -> pd.DataFrame:
    metrics = ["TP", "FP", "FN", "F1", "precision", "recall", "false_new_tracks", "false_new_per_100_frames", "map_extra_entries", "extra_messages_due_to_false_tracks"]
    return raw.groupby(["method", "agent_count"], as_index=False)[metrics].mean()


def plot_tradeoff(summary: pd.DataFrame, path: Path) -> None:
    plt.figure(figsize=(7, 4.5))
    for _, r in summary.iterrows():
        plt.scatter(r["false_new_tracks"], r["F1"], s=70)
        plt.annotate(str(r["method"]), (r["false_new_tracks"], r["F1"]), xytext=(4, 4), textcoords="offset points", fontsize=7)
    plt.xlabel("Ложные новые треки")
    plt.ylabel("F1")
    plt.title("Simple fusion baselines: F1 / false-new")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def write_claim_safe(path: Path, summary: pd.DataFrame, source: str) -> None:
    proposed = row(summary, "S2_v9_selected")
    lines = ["# Simple Fusion Baselines Claim-Safe Notes", "", f"Actual controlled input source: `{source}`.", ""]
    lines += ["```text", summary[["method", "F1", "false_new_tracks"]].to_string(index=False), "```", ""]
    for name in ["naive_union", "mean_confidence_fusion", "max_confidence_fusion", "bayesian_existence_filter"]:
        r = row(summary, name)
        if proposed is not None and r is not None:
            lines.append(f"- Proposed vs {name}: F1 delta={proposed['F1'] - r['F1']:+.6f}, false_new delta={proposed['false_new_tracks'] - r['false_new_tracks']:+.1f}.")
    lines += [
        "",
        "Safe claim: the proposed layer is compared with simple fusion baselines and provides a conservative operating point in this controlled replay.",
        "Forbidden claim: do not call these baselines cooperative perception SOTA and do not claim replacement for CoBEVFusion/V2X-Real.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def row(df: pd.DataFrame, method: str) -> pd.Series | None:
    hit = df[df["method"].astype(str).eq(method)]
    return None if hit.empty else hit.iloc[0]


if __name__ == "__main__":
    main()
