#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from defense4uavswarm.q1_visdrone import (
    Q1Params,
    add_single_camera_features,
    compute_metrics,
    label_detections_protocol,
    load_detections,
    load_gt_protocol,
    method_acceptance,
)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-root", required=True)
    p.add_argument("--detections", required=True)
    p.add_argument("--methods", nargs="+", required=True)
    p.add_argument("--matching-mode", default="coarse_class")
    p.add_argument("--ignore-policy", default="exclude_ignored")
    p.add_argument("--iou-threshold", type=float, default=0.5)
    p.add_argument("--detector-conf-threshold", type=float, default=0.1)
    p.add_argument("--output-dir", required=True)
    args = p.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    det_raw = load_detections(args.detections)
    gt, ignored = load_gt_protocol(args.dataset_root, sorted(det_raw["sequence_id"].unique()) if not det_raw.empty else None)
    keys = det_raw[["sequence_id", "frame_id"]].drop_duplicates()
    gt = gt.merge(keys, on=["sequence_id", "frame_id"], how="inner")
    ignored = ignored.merge(keys, on=["sequence_id", "frame_id"], how="inner") if not ignored.empty else ignored
    speed = compute_gt_speed_bins(gt)
    det = add_single_camera_features(label_detections_protocol(det_raw, gt, ignored, args.iou_threshold, args.matching_mode, args.ignore_policy, args.detector_conf_threshold))
    det["speed_bin"] = det["matched_gt_id"].astype(str).map(speed["object_speed_bin"]).fillna("unmatched_fp_proxy")

    rows = []
    for method in args.methods:
        if method == "geometry_dynamic_no_multiagent":
            accepted = evaluate_acceptance(det, method)
        elif method == "bytetrack":
            accepted = evaluate_acceptance(det, method)
        else:
            continue
        for speed_bin in ["slow", "medium", "fast"]:
            gt_ids = set(speed[speed["object_speed_bin"].eq(speed_bin)]["object_id"].astype(str))
            sub_det = det[det["matched_gt_id"].astype(str).isin(gt_ids) | (~det["eval_is_tp"].astype(bool))].copy()
            sub_acc = accepted.loc[sub_det.index]
            expected_gt = int(gt[gt["object_id"].astype(str).isin(gt_ids)].shape[0])
            row = compute_metrics(sub_det, sub_acc, expected_gt, max(1, gt[["sequence_id", "frame_id"]].drop_duplicates().shape[0]), method)
            row["speed_bin"] = speed_bin
            row["metric_scope"] = "TP/FN by GT object speed; FP false-new are unmatched proxy candidates"
            rows.append(row)

    metrics = pd.DataFrame(rows)
    metrics.to_csv(out / "motion_metrics.csv", index=False)
    plot_motion(metrics, out / "fig_motion_error_ru.png")
    write_summary(out / "motion_error_summary.md", metrics, speed)
    print(f"status=ok rows={len(metrics)} output={out}")


def compute_gt_speed_bins(gt: pd.DataFrame) -> pd.DataFrame:
    g = gt.copy()
    g["cx"] = (g["x1"].astype(float) + g["x2"].astype(float)) / 2.0
    g["cy"] = (g["y1"].astype(float) + g["y2"].astype(float)) / 2.0
    rows = []
    for (seq, obj), group in g.sort_values("frame_id").groupby(["sequence_id", "object_id"], sort=False):
        frames = group["frame_id"].astype(float).to_numpy()
        cx = group["cx"].to_numpy(dtype=float)
        cy = group["cy"].to_numpy(dtype=float)
        if len(group) < 2:
            speed = 0.0
        else:
            dist = np.sqrt(np.diff(cx) ** 2 + np.diff(cy) ** 2)
            dt = np.maximum(1.0, np.diff(frames))
            speed = float(np.median(dist / dt))
        rows.append({"sequence_id": seq, "object_id": str(obj), "speed_px_per_frame": speed})
    out = pd.DataFrame(rows)
    q1, q2 = out["speed_px_per_frame"].quantile([1 / 3, 2 / 3]).to_list()
    out["object_speed_bin"] = np.where(out["speed_px_per_frame"] <= q1, "slow", np.where(out["speed_px_per_frame"] <= q2, "medium", "fast"))
    return out


def evaluate_acceptance(det: pd.DataFrame, method: str) -> pd.Series:
    return method_acceptance(det, method, Q1Params())


def plot_motion(metrics: pd.DataFrame, path: Path) -> None:
    if metrics.empty:
        return
    piv = metrics.pivot_table(index="speed_bin", columns="method", values="F1", aggfunc="first").reindex(["slow", "medium", "fast"])
    ax = piv.plot(kind="bar", figsize=(7, 4))
    ax.set_title("F1 по скорости объектов (proxy-анализ)")
    ax.set_xlabel("Группа скорости")
    ax.set_ylabel("F1")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def write_summary(path: Path, metrics: pd.DataFrame, speed: pd.DataFrame) -> None:
    lines = [
        "# Motion Error Analysis Summary",
        "",
        "Status: done with limitations.",
        "",
        "Motion groups are computed from VisDrone GT object center displacement in pixels per frame.",
        "TP/FN are grouped by matched GT object speed. FP/false-new tracks have no true object speed, so FP values are retained as unmatched candidate proxies.",
        "This is a diagnostic proxy, not a stable motion benchmark.",
        "",
        f"GT objects with speed estimate: {len(speed)}",
        "",
        "```text",
        metrics[["method", "speed_bin", "F1", "FP", "FN", "false_new_tracks"]].to_string(index=False) if len(metrics) else "no metrics",
        "```",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
