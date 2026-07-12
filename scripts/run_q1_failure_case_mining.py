#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from defense4uavswarm.q1_visdrone import Q1Params, adaptive_geometry_acceptance, compute_metrics, method_acceptance


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-root", required=True)
    p.add_argument("--detections", required=True)
    p.add_argument("--methods", nargs="+", required=True)
    p.add_argument("--matching-mode", default="coarse_class")
    p.add_argument("--ignore-policy", default="exclude_ignored")
    p.add_argument("--iou-threshold", type=float, default=0.5)
    p.add_argument("--feature-audit", default="outputs/results/q1_final_corrected/yolov8s_main/feature_audit.csv")
    p.add_argument("--output-dir", required=True)
    args = p.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    det = pd.read_csv(args.feature_audit)
    det = add_bins(det)
    accept = {m: acceptance(det, m) for m in args.methods}

    size = grouped_metrics(det, accept, "size_bin")
    density = grouped_metrics(det, accept, "density_bin")
    conf = grouped_metrics(det, accept, "confidence_bin")
    motion = grouped_metrics(det, accept, "motion_bin")
    size.to_csv(out / "failure_by_size.csv", index=False)
    density.to_csv(out / "failure_by_density.csv", index=False)
    conf.to_csv(out / "failure_by_confidence.csv", index=False)
    motion.to_csv(out / "failure_by_motion.csv", index=False)
    plot_delta(size, "size_bin", out / "fig_failure_size_ru.png")
    plot_delta(density, "density_bin", out / "fig_failure_density_ru.png")
    plot_delta(conf, "confidence_bin", out / "fig_failure_confidence_ru.png")
    plot_delta(motion, "motion_bin", out / "fig_failure_motion_ru.png")
    write_claim(out / "failure_case_claim_safe.md", size, density, conf, motion)
    print(f"status=ok output={out}")


def add_bins(det: pd.DataFrame) -> pd.DataFrame:
    d = det.copy()
    area = ((d["x2"] - d["x1"]).clip(lower=1) * (d["y2"] - d["y1"]).clip(lower=1)).astype(float)
    d["bbox_area_diag"] = area
    q1, q2 = area.quantile([1 / 3, 2 / 3]).to_list()
    d["size_bin"] = np.where(area <= q1, "small", np.where(area <= q2, "medium", "large"))
    dens = d["num_detections_in_frame"].astype(float)
    dq1, dq2 = dens.quantile([1 / 3, 2 / 3]).to_list()
    d["density_bin"] = np.where(dens <= dq1, "low_density", np.where(dens <= dq2, "medium_density", "high_density"))
    c = d["confidence"].astype(float)
    d["confidence_bin"] = np.where(c < 0.30, "low_confidence", np.where(c < 0.50, "medium_confidence", "high_confidence"))
    tracklet_group = ["sequence_id"] + (["agent_id"] if "agent_id" in d.columns else []) + ["tracklet_id"]
    speed_proxy = d.groupby(tracklet_group)["frame_id"].transform("nunique").astype(float)
    sq1, sq2 = speed_proxy.quantile([1 / 3, 2 / 3]).to_list()
    d["motion_bin"] = np.where(speed_proxy <= sq1, "fast_proxy", np.where(speed_proxy <= sq2, "medium_proxy", "slow_proxy"))
    return d


def acceptance(det: pd.DataFrame, method: str) -> pd.Series:
    if method == "geometry_dynamic_adaptive_balanced":
        import yaml

        cfg_path = Path("outputs/results/q1_improvement/yolov8s_sweep/selected_configs.yaml")
        cfgs = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
        if not isinstance(cfgs, dict) or "selected_balanced" not in cfgs:
            raise ValueError(f"Missing selected_balanced in adaptive geometry config: {cfg_path}")
        return adaptive_geometry_acceptance(det, cfgs["selected_balanced"])
    return method_acceptance(det, method, Q1Params())


def grouped_metrics(det: pd.DataFrame, accept: dict[str, pd.Series], group_col: str) -> pd.DataFrame:
    rows = []
    total_frames = max(1, det[["sequence_id", "frame_id"]].drop_duplicates().shape[0])
    for group_value, sub in det.groupby(group_col):
        # Candidate-level diagnostic: expected positives are labeled TP candidates in the bin.
        expected_gt = int(sub["eval_is_tp"].astype(bool).sum())
        for method, acc in accept.items():
            row = compute_metrics(sub, acc.loc[sub.index], expected_gt, total_frames, method)
            row[group_col] = group_value
            row["diagnostic_scope"] = "candidate-level bin; absent GT objects outside detector candidates are not counted"
            rows.append(row)
    df = pd.DataFrame(rows)
    return rename_candidate_metrics(add_delta(df, group_col))


def add_delta(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    out = df.copy()
    out["candidate_conditional_F1_delta_vs_bytetrack"] = pd.NA
    out["false_new_delta_vs_bytetrack"] = pd.NA
    for _, g in out.groupby(group_col):
        b = g[g["method"].eq("bytetrack")]
        if b.empty:
            continue
        base = b.iloc[0]
        idx = g.index
        out.loc[idx, "candidate_conditional_F1_delta_vs_bytetrack"] = out.loc[idx, "F1"].astype(float) - float(base["F1"])
        out.loc[idx, "false_new_delta_vs_bytetrack"] = out.loc[idx, "false_new_tracks"].astype(float) - float(base["false_new_tracks"])
    return out


def rename_candidate_metrics(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(
        columns={
            "precision": "candidate_conditional_precision",
            "recall": "candidate_conditional_recall",
            "F1": "candidate_conditional_F1",
            "IDF1": "candidate_conditional_IDF1",
        }
    )


def plot_delta(df: pd.DataFrame, group_col: str, path: Path) -> None:
    hit = df[df["method"].eq("geometry_dynamic_no_multiagent")]
    if hit.empty:
        return
    plt.figure(figsize=(7, 4))
    plt.bar(hit[group_col].astype(str), hit["false_new_delta_vs_bytetrack"].astype(float))
    plt.ylabel("false_new delta vs ByteTrack")
    plt.xticks(rotation=25, ha="right")
    plt.title(group_col)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def write_claim(path: Path, size: pd.DataFrame, density: pd.DataFrame, conf: pd.DataFrame, motion: pd.DataFrame) -> None:
    def worst(df: pd.DataFrame, col: str) -> str:
        hit = df[df["method"].eq("geometry_dynamic_no_multiagent")].copy()
        if hit.empty:
            return "unavailable"
        return str(hit.sort_values("candidate_conditional_F1_delta_vs_bytetrack").iloc[0][col])

    lines = [
        "# Failure Case Mining Claim-Safe Notes",
        "",
        "Diagnostics are candidate-level bins from saved VisDrone detections. They do not reassign absent GT objects to bins.",
        "Tracklet persistence is a post-hoc diagnostic computed from the complete recorded tracklet. It is not an online feature.",
        f"Most negative F1 delta by size: `{worst(size, 'size_bin')}`.",
        f"Most negative F1 delta by confidence: `{worst(conf, 'confidence_bin')}`.",
        f"Most negative F1 delta by density: `{worst(density, 'density_bin')}`.",
        "Motion analysis: proxy only. Tracklet length is used as a motion/persistence proxy; true FP objects do not have GT motion.",
        "",
        "Safe claim: use these as failure diagnostics, not as formal robustness proof.",
        "Forbidden claim: do not claim speed robustness from this proxy analysis.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
