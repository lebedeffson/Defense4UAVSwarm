#!/usr/bin/env python
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from defense4uavswarm.config import load_config
from defense4uavswarm.datasets.visdrone import VisDroneDetDataset
from defense4uavswarm.metrics.detection import match_frame


def f1(tp: int, fp: int, fn: int) -> float:
    return 2 * tp / max(1, 2 * tp + fp + fn)


def accepted(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["accepted"] == True] if "accepted" in df else df


def write_threshold_summary(results: Path) -> None:
    src = results / "threshold_selection.csv"
    if not src.exists():
        return
    df = pd.read_csv(src)
    rows = []
    group_cols = [c for c in ["task", "model_name", "mode", "t_norm"] if c in df]
    for key, g in df.groupby(group_cols, dropna=False):
        selected = g[g["selected"] == True]
        item = dict(zip(group_cols, key if isinstance(key, tuple) else (key,)))
        item["class_group"] = "all"
        item["candidate_tau_values"] = ",".join(str(x) for x in sorted(g["tau"].dropna().unique()))
        item["selected_tau"] = float(selected.iloc[0]["tau"]) if len(selected) else None
        item["selected_F1"] = float(selected.iloc[0]["F1"]) if len(selected) else None
        rows.append(item)
    pd.DataFrame(rows).to_csv(results / "threshold_summary.csv", index=False)


def write_metric_debug_sample(cfg: dict, results: Path, model: str, eps: float, n: int) -> None:
    ds = VisDroneDetDataset(cfg["dataset"]["root"], "val")
    s0 = pd.read_csv(results / f"det_{model}_s0_baseline.csv")
    s1 = pd.read_csv(results / f"det_{model}_s1_fgsm_eps_{eps}.csv")
    gt = ds.all_annotations(ds.sequence_ids()[:n])
    s0_frames = {k: v for k, v in s0.groupby(["sequence_id", "frame_id"])}
    s1_frames = {k: v for k, v in s1.groupby(["sequence_id", "frame_id"])}
    rows = []
    for seq in ds.sequence_ids()[:n]:
        gt_f = gt[(gt["sequence_id"] == seq) & (gt["frame_id"] == 1)]
        s0_f = s0_frames.get((seq, 1), s0.iloc[0:0])
        s1_f = s1_frames.get((seq, 1), s1.iloc[0:0])
        tp0, fp0, fn0, _ = match_frame(gt_f, s0_f)
        tp1, fp1, fn1, _ = match_frame(gt_f, s1_f)
        rows.append(
            {
                "image_id": seq,
                "num_gt": len(gt_f),
                "num_pred_s0": len(accepted(s0_f)),
                "num_pred_s1": len(accepted(s1_f)),
                "num_tp_s0": tp0,
                "num_fp_s0": fp0,
                "num_fn_s0": fn0,
                "num_tp_s1": tp1,
                "num_fp_s1": fp1,
                "num_fn_s1": fn1,
                "F1_s0": f1(tp0, fp0, fn0),
                "F1_s1": f1(tp1, fp1, fn1),
            }
        )
    pd.DataFrame(rows).to_csv(results / "metric_debug_sample.csv", index=False)


def write_article_tables(results: Path) -> None:
    df = pd.read_csv(results / "research_matrix.csv")
    all_group = df[df["class_group"] == "all"]
    rows = []
    for (model, eps), g in all_group[all_group["scenario"] != "S0"].groupby(["model_name", "eps"]):
        s0 = all_group[(all_group["model_name"] == model) & (all_group["scenario"] == "S0")].iloc[0]
        s1 = g[g["scenario"] == "S1"].iloc[0]
        sn = g[g["scenario"] == "S_naive"].sort_values("F1", ascending=False).iloc[0]
        s2 = g[g["scenario"] == "S2_conf"].sort_values("F1", ascending=False).iloc[0]
        rows.append(
            {
                "model_name": model,
                "eps": eps,
                "S0_F1": s0["F1"],
                "S1_F1": s1["F1"],
                "S_naive_F1": sn["F1"],
                "S2_conf_F1": s2["F1"],
                "ASR_det_S1": s1.get("ASR_det"),
                "ASR_det_S2_conf": s2.get("ASR_det"),
            }
        )
    pd.DataFrame(rows).to_csv(results / "table_det1_model_robustness.csv", index=False)

    rows = []
    for (model, eps, group), g in df[df["scenario"] != "S0"].groupby(["model_name", "eps", "class_group"]):
        s0 = df[(df["model_name"] == model) & (df["class_group"] == group) & (df["scenario"] == "S0")].iloc[0]
        s1 = g[g["scenario"] == "S1"].iloc[0]
        s2 = g[g["scenario"] == "S2_conf"].sort_values("F1", ascending=False).iloc[0]
        rows.append(
            {
                "model_name": model,
                "eps": eps,
                "class_group": group,
                "S0_F1": s0["F1"],
                "S1_F1": s1["F1"],
                "S2_conf_F1": s2["F1"],
                "delta_F1": s1["F1"] - s0["F1"],
                "ASR_det": s1.get("ASR_det"),
            }
        )
    pd.DataFrame(rows).to_csv(results / "table_det2_class_robustness.csv", index=False)

    keep = df[df["scenario"].isin(["S_naive", "S2_conf"])].copy()
    keep["selected_tau"] = keep["tau"]
    keep[
        [
            "model_name",
            "eps",
            "scenario",
            "t_norm",
            "selected_tau",
            "rejection_rate",
            "FP",
            "FN",
            "F1",
        ]
    ].to_csv(results / "table_det3_threshold_filtering.csv", index=False)

    rows = []
    for (model, eps), g in all_group[all_group["scenario"] != "S0"].groupby(["model_name", "eps"]):
        s1 = g[g["scenario"] == "S1"].iloc[0]
        s2 = g[g["scenario"] == "S2_conf"].sort_values("F1", ascending=False).iloc[0]
        rows.append(
            {
                "model_name": model,
                "eps": eps,
                "S1_F1": s1["F1"],
                "S2_conf_F1": s2["F1"],
                "ASR_det_S1": s1.get("ASR_det"),
                "ASR_det_S2_conf": s2.get("ASR_det"),
            }
        )
    pd.DataFrame(rows).to_csv(results / "table_det4_eps_sensitivity.csv", index=False)


def write_fgsm_examples(cfg: dict, results: Path, figures: Path, model: str, limit: int) -> None:
    ds = VisDroneDetDataset(cfg["dataset"]["root"], "val")
    out = figures / "fgsm_examples"
    out.mkdir(parents=True, exist_ok=True)
    ids = ds.sequence_ids()[:limit]
    eps_frames = []
    for eps_label, eps_value in [("eps004", 0.004), ("eps008", 0.008)]:
        path = results / f"det_{model}_s1_fgsm_eps_{eps_value}.csv"
        if path.exists():
            eps_frames.append((eps_label, pd.read_csv(path)))
    for seq in ids:
        clean = next(ds.frames([seq])).image_path
        shutil.copyfile(clean, out / f"clean_{seq}.jpg")
        clean_img = cv2.imread(str(clean), cv2.IMREAD_COLOR)
        for eps_label, frame in eps_frames:
            rows = frame[frame["sequence_id"] == seq]
            if rows.empty:
                continue
            attacked = Path(str(rows.iloc[0]["image_path"]))
            if not attacked.exists():
                continue
            dst = out / f"attacked_{eps_label}_{seq}.jpg"
            shutil.copyfile(attacked, dst)
            atk_img = cv2.imread(str(attacked), cv2.IMREAD_COLOR)
            if clean_img is None or atk_img is None:
                continue
            diff = cv2.absdiff(clean_img, atk_img)
            if diff.max() > 0:
                diff = np.clip(diff.astype(np.float32) * (255.0 / diff.max()), 0, 255).astype(np.uint8)
            cv2.imwrite(str(out / f"perturbation_{eps_label}_{seq}.png"), diff)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--results", default="outputs/results/det_val")
    p.add_argument("--figures", default="outputs/figures/det_val")
    p.add_argument("--debug-model", default="yolov8n")
    p.add_argument("--debug-eps", type=float, default=0.008)
    p.add_argument("--debug-images", type=int, default=10)
    args = p.parse_args()

    cfg = load_config(args.config)
    results = Path(args.results)
    figures = Path(args.figures)
    write_threshold_summary(results)
    write_metric_debug_sample(cfg, results, args.debug_model, args.debug_eps, args.debug_images)
    write_article_tables(results)
    write_fgsm_examples(cfg, results, figures, args.debug_model, args.debug_images)


if __name__ == "__main__":
    main()
