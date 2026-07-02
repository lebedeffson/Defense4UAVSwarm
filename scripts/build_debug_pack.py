#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO

from defense4uavswarm.class_groups import VISDRONE_ID_TO_NAME, load_class_groups, normalized_names
from defense4uavswarm.config import load_config
from defense4uavswarm.datasets.visdrone import VisDroneDataset
from defense4uavswarm.filtering.tnorms import apply_tnorm
from defense4uavswarm.matrix import load_sequence_list, task_dataset
from defense4uavswarm.metrics.detection import match_frame, precision_recall_f1


def copy_if_exists(src: Path, dst: Path) -> None:
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def sample_frames(ds: VisDroneDataset, seq: str, n: int) -> list[int]:
    return [r.frame_id for r in list(ds.frames([seq]))[:n]]


def subset(df: pd.DataFrame, seq: str, frames: list[int]) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    return df[(df["sequence_id"] == seq) & (df["frame_id"].isin(frames))].copy()


def greedy_match_rows(gt: pd.DataFrame, pred: pd.DataFrame, aliases: dict[str, str], class_agnostic: bool) -> pd.DataFrame:
    rows = []
    empty_pred = pred.iloc[0:0]
    pred_frames = {k: v for k, v in pred.groupby(["sequence_id", "frame_id"])} if len(pred) else {}
    for (seq, frame), gt_f in gt.groupby(["sequence_id", "frame_id"]):
        pred_f = pred_frames.get((seq, frame), empty_pred)
        _, _, _, pairs = match_frame(gt_f, pred_f, class_agnostic=class_agnostic, aliases=aliases)
        matched_gt = {gi for gi, _ in pairs}
        matched_pred = {pi for _, pi in pairs}
        for gi, pi in pairs:
            gr, pr = gt.loc[gi], pred.loc[pi]
            rows.append(match_row(frame, gr, pr, True, class_agnostic, aliases))
        for gi in set(gt_f.index) - matched_gt:
            gr = gt.loc[gi]
            rows.append(match_row(frame, gr, None, False, class_agnostic, aliases, "no_iou_match"))
        for pi in set(pred_f.index) - matched_pred:
            pr = pred.loc[pi]
            rows.append(match_row(frame, None, pr, False, class_agnostic, aliases, "unmatched_prediction"))
    return pd.DataFrame(rows)


def match_row(frame_id: int, gt, pred, matched: bool, class_agnostic: bool, aliases: dict[str, str], reason: str | None = None) -> dict:
    iou = None
    if gt is not None and pred is not None:
        g = np.array([gt.x1, gt.y1, gt.x2, gt.y2], dtype=float)
        p = np.array([pred.x1, pred.y1, pred.x2, pred.y2], dtype=float)
        ix1, iy1 = max(g[0], p[0]), max(g[1], p[1])
        ix2, iy2 = min(g[2], p[2]), min(g[3], p[3])
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        ga = max(0.0, g[2] - g[0]) * max(0.0, g[3] - g[1])
        pa = max(0.0, p[2] - p[0]) * max(0.0, p[3] - p[1])
        iou = inter / max(1e-6, ga + pa - inter)
    return {
        "frame_id": frame_id,
        "gt_id": None if gt is None else gt.gt_track_id,
        "gt_class_id": None if gt is None else gt.class_id,
        "gt_class_name": None if gt is None else gt.class_name,
        "pred_id": None if pred is None else pred.pred_track_id,
        "pred_class_id": None if pred is None else pred.class_id,
        "pred_class_name": None if pred is None else pred.class_name,
        "iou": iou,
        "matched": matched,
        "match_reason": reason or ("iou_match" if class_agnostic else "iou_and_class_match"),
        "class_agnostic": class_agnostic,
    }


def baseline_report(gt: pd.DataFrame, s0: pd.DataFrame, sequences: list[str], models: list[str], aliases: dict[str, str]) -> pd.DataFrame:
    rows = []
    for model in models:
        pred_m = s0[s0["model_name"] == model] if "model_name" in s0 else s0
        for seq in sequences:
            gt_s = gt[gt["sequence_id"] == seq]
            pred_s = pred_m[pred_m["sequence_id"] == seq]
            ag = precision_recall_f1(gt_s, pred_s, include_map=False, class_agnostic=True, aliases=aliases)
            aw = precision_recall_f1(gt_s, pred_s, include_map=False, class_agnostic=False, aliases=aliases)
            rows.append(
                {
                    "model_name": model,
                    "sequence_id": seq,
                    "num_gt": len(gt_s),
                    "num_pred": len(pred_s),
                    "mean_conf": pred_s["confidence"].mean() if len(pred_s) else 0.0,
                    "precision_class_aware": aw["precision"],
                    "recall_class_aware": aw["recall"],
                    "F1_class_aware": aw["F1"],
                    "precision_class_agnostic": ag["precision"],
                    "recall_class_agnostic": ag["recall"],
                    "F1_class_agnostic": ag["F1"],
                }
            )
    return pd.DataFrame(rows)


def ki_distribution(frames: list[pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for df in frames:
        if df is None or df.empty or "k_i" not in df:
            continue
        for key, g in df.groupby(["scenario", "model_name", "eps", "t_norm"], dropna=False):
            k = g["k_i"].dropna().astype(float)
            if k.empty:
                continue
            rows.append(
                {
                    "scenario": key[0],
                    "model_name": key[1],
                    "eps": key[2],
                    "t_norm": key[3],
                    "k_i_mean": k.mean(),
                    "k_i_median": k.median(),
                    "k_i_p10": k.quantile(0.10),
                    "k_i_p90": k.quantile(0.90),
                    "num_low_k": int((k < 0.5).sum()),
                    "num_total": int(len(k)),
                }
            )
    return pd.DataFrame(rows)


def q_tau_distribution(gt: pd.DataFrame, s1: pd.DataFrame, s2_frames: list[pd.DataFrame], aliases: dict[str, str]) -> pd.DataFrame:
    rows = []
    before = precision_recall_f1(gt, s1, include_map=False, class_agnostic=True, aliases=aliases)
    for df in s2_frames:
        if df is None or df.empty or "Q_i" not in df:
            continue
        after = precision_recall_f1(gt, df, include_map=False, class_agnostic=True, aliases=aliases)
        q = df["Q_i"].dropna().astype(float)
        rows.append(
            {
                "scenario": df["scenario"].dropna().iloc[0] if len(df["scenario"].dropna()) else "S2",
                "model_name": df["model_name"].dropna().iloc[0],
                "eps": df["eps"].dropna().iloc[0],
                "t_norm": df["t_norm"].dropna().iloc[0],
                "tau": df["tau"].dropna().iloc[0],
                "Q_mean": q.mean(),
                "Q_median": q.median(),
                "Q_p10": q.quantile(0.10),
                "Q_p90": q.quantile(0.90),
                "accepted_rate": float(df["accepted"].mean()),
                "precision_before_filter": before["precision"],
                "recall_before_filter": before["recall"],
                "precision_after_filter": after["precision"],
                "recall_after_filter": after["recall"],
            }
        )
    return pd.DataFrame(rows)


def write_reports(out: Path, cfg: dict, ds: VisDroneDataset, model_names: list[str], sample_rec) -> None:
    names = YOLO(cfg["model"]["weights"]).names
    img = cv2.imread(str(sample_rec.image_path))
    h, w = img.shape[:2]
    groups = load_class_groups(cfg["class_groups_file"])
    (out / "class_mapping_report.txt").write_text(
        "\n".join(
            [
                "YOLO weights: COCO pretrained inferred from yolov8*.pt names, not VisDrone fine-tuned.",
                f"YOLO model.names: {names}",
                f"VisDrone class names: {VISDRONE_ID_TO_NAME}",
                "Pred class_id is raw COCO class id from YOLO; pred class_name is COCO class name.",
                f"Aliases used for grouping/class-aware audit: {groups.get('aliases', {})}",
                "No full COCO->VisDrone class-id remap is currently applied.",
                "Default experiment evaluation is class_agnostic=true unless CLI overrides it.",
                "Known partial semantic mapping: person->pedestrian, motorcycle->motor; car is not remapped to van.",
                f"Class groups: {groups.get('class_groups', {})}",
            ]
        ),
        encoding="utf-8",
    )
    (out / "preprocessing_report.txt").write_text(
        "\n".join(
            [
                f"Sample frame: {sample_rec.image_path}",
                f"Original frame size: {w}x{h}",
                f"YOLO imgsz: {cfg['model']['imgsz']}",
                "Ultralytics inference uses its internal preprocessing/letterbox.",
                "FGSM implementation resizes RGB image directly to imgsz x imgsz tensor.",
                "Adversarial perturbation is resized back to original image size and clipped to [0,1].",
                "Detection bbox outputs are Ultralytics xyxy in original image coordinates.",
                "Evaluation IoU is computed in original image coordinates.",
                "Potential audit issue: FGSM tensor preprocessing is resize-based, not exact Ultralytics letterbox preprocessing.",
            ]
        ),
        encoding="utf-8",
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--results-dir", default="outputs/results/class_only")
    p.add_argument("--out", default="outputs/debug_pack")
    p.add_argument("--models", nargs="+", default=["yolov8n", "yolov8s"])
    p.add_argument("--task", default="vid")
    p.add_argument("--sequence-list", default="configs/selected_sequences.yaml")
    p.add_argument("--sample-sequence", default=None)
    p.add_argument("--frames", type=int, default=50)
    p.add_argument("--eps", type=float, default=0.032)
    p.add_argument("--t-norm", default="T_min")
    args = p.parse_args()

    cfg = load_config(args.config)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    results = Path(args.results_dir)
    _, ds = task_dataset(cfg, args.task)
    sequences = load_sequence_list(args.sequence_list, args.task) or ds.sequence_ids()
    seq = args.sample_sequence or sequences[0]
    frames = sample_frames(ds, seq, args.frames)
    aliases = load_class_groups(cfg["class_groups_file"]).get("aliases", {})

    copy_if_exists(results / "metadata.json", out / "metadata.json")
    for name in ["loss_comparison.csv", "fgsm_diagnostics.csv", "asr_breakdown.csv", "class_distribution.csv", "sequence_class_distribution.csv"]:
        copy_if_exists(Path("outputs/results") / name, out / name)
        copy_if_exists(results / name, out / name)
    fgsm_diag = out / "fgsm_diagnostics.csv"
    if fgsm_diag.exists():
        diag = pd.read_csv(fgsm_diag)
        if "actual_loss_used" not in diag and "loss_mode_actual" in diag:
            diag["actual_loss_used"] = diag["loss_mode_actual"]
        diag.to_csv(fgsm_diag, index=False)
    copy_if_exists(results / "threshold_selection.csv", out / "threshold_selection.csv")
    research = pd.read_csv(results / "research_matrix.csv") if (results / "research_matrix.csv").exists() else pd.DataFrame()
    research.head(100).to_csv(out / "research_matrix_head.csv", index=False)

    gt_all = ds.all_annotations(sequences)
    gt_sample = subset(gt_all, seq, frames)
    gt_sample.to_csv(out / "sample_gt.csv", index=False)
    all_s0 = []
    for model in args.models:
        s0 = pd.read_csv(results / f"{args.task}_{model}_s0_baseline.csv")
        s1 = pd.read_csv(results / f"{args.task}_{model}_s1_fgsm_eps_{args.eps}.csv")
        s2 = pd.read_csv(results / f"{args.task}_{model}_s2_{args.t_norm}_eps_{args.eps}.csv")
        all_s0.append(s0)
        if model == args.models[0]:
            subset(s0, seq, frames).to_csv(out / "sample_predictions_s0.csv", index=False)
            subset(s1, seq, frames).to_csv(out / "sample_predictions_s1_eps0032.csv", index=False)
            subset(s2, seq, frames).to_csv(out / "sample_predictions_s2_eps0032.csv", index=False)
            greedy_match_rows(gt_sample, subset(s0, seq, frames), aliases, True).to_csv(out / "sample_matches_s0.csv", index=False)
            greedy_match_rows(gt_sample, subset(s1, seq, frames), aliases, True).to_csv(out / "sample_matches_s1_eps0032.csv", index=False)
            greedy_match_rows(gt_sample, subset(s2, seq, frames), aliases, True).to_csv(out / "sample_matches_s2_eps0032.csv", index=False)
            pd.concat([greedy_match_rows(gt_sample, subset(s0, seq, frames), aliases, False)]).to_csv(out / "sample_matches_s0_class_aware.csv", index=False)
    s0_all = pd.concat(all_s0, ignore_index=True)
    baseline_report(gt_all, s0_all, sequences, args.models, aliases).to_csv(out / "baseline_detection_report.csv", index=False)

    s1_first = pd.read_csv(results / f"{args.task}_{args.models[0]}_s1_fgsm_eps_{args.eps}.csv")
    s2_frames = [pd.read_csv(p) for p in sorted(results.glob(f"{args.task}_{args.models[0]}_s2_*_eps_{args.eps}.csv"))]
    ki_distribution([s0_all, s1_first, *s2_frames]).to_csv(out / "ki_distribution.csv", index=False)
    q_tau_distribution(gt_all, s1_first, s2_frames, aliases).to_csv(out / "q_tau_distribution.csv", index=False)
    write_reports(out, cfg, ds, args.models, next(ds.frames([seq])))
    print(out)


if __name__ == "__main__":
    main()
