#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

import pandas as pd

from defense4uavswarm.q1_v5.map_contamination import contamination_metrics
from defense4uavswarm.q1_visdrone import Q1Params, compute_metrics, load_gt_protocol, method_acceptance, train_rf


def read_det(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path)


def gt_for(det: pd.DataFrame, dataset_root: str) -> pd.DataFrame:
    gt, _ = load_gt_protocol(dataset_root, sorted(det["sequence_id"].unique()))
    keys = det[["sequence_id", "frame_id"]].drop_duplicates()
    return gt.merge(keys, on=["sequence_id", "frame_id"], how="inner")


def score(method: str, det: pd.DataFrame, accepted: pd.Series, gt: pd.DataFrame, context: dict[str, Any]) -> dict[str, Any]:
    row = compute_metrics(det, accepted, len(gt), max(1, gt[["sequence_id", "frame_id"]].drop_duplicates().shape[0]), method)
    row.update(contamination_metrics(det, accepted))
    row.update(context)
    return row


def eval_rf_transfer(train: pd.DataFrame, test: pd.DataFrame, gt: pd.DataFrame, context: dict[str, Any]) -> dict[str, Any]:
    if train.empty or test.empty:
        row = {"method": "rf_filter", "status": "skipped", "reason": "empty train or test split"}
        row.update(context)
        return row
    model, tau = train_rf(train)
    accepted = method_acceptance(test, "rf_learned_gate", Q1Params(rf_threshold=tau), model)
    row = score("rf_filter", test, accepted, gt, context)
    row.update({"status": "ok", "rf_threshold": tau, "train_rows": len(train), "test_rows": len(test)})
    return row


def eval_untrained_methods(test: pd.DataFrame, gt: pd.DataFrame, context: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    params = Q1Params()
    for method in ["bytetrack", "geometry_dynamic_no_multiagent", "persistence_gate", "bayesian_existence_filter"]:
        rows.append(score(method, test, method_acceptance(test, method, params), gt, {**context, "status": "ok"}))
    rows.append(score("confidence_threshold_0.2", test, test["confidence"].astype(float) >= 0.2, gt, {**context, "status": "ok"}))
    return rows


def loso(det: pd.DataFrame, dataset_root: str, detector: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for seq in sorted(det["sequence_id"].unique()):
        train = det[~det["sequence_id"].eq(seq)].copy()
        test = det[det["sequence_id"].eq(seq)].copy()
        gt = gt_for(test, dataset_root)
        context = {"protocol": "loso", "train_domain": detector, "test_domain": detector, "test_sequence": seq}
        rows.append(eval_rf_transfer(train, test, gt, context))
        rows.extend(eval_untrained_methods(test, gt, context))
    return pd.DataFrame(rows)


def cross_detector(train_det: pd.DataFrame, test_det: pd.DataFrame, dataset_root: str, train_name: str, test_name: str) -> pd.DataFrame:
    gt = gt_for(test_det, dataset_root)
    context = {"protocol": "cross_detector", "train_domain": train_name, "test_domain": test_name, "test_sequence": "all"}
    rows = [eval_rf_transfer(train_det, test_det, gt, context)]
    rows.extend(eval_untrained_methods(test_det, gt, context))
    return pd.DataFrame(rows)


def summarize(raw: pd.DataFrame) -> pd.DataFrame:
    metric_cols = ["F1", "false_new_tracks", "false_track_occupancy_frames", "recall", "precision"]
    rows = []
    ok = raw[raw["status"].eq("ok")].copy()
    for keys, group in ok.groupby(["protocol", "train_domain", "test_domain", "method"], dropna=False):
        row = dict(zip(["protocol", "train_domain", "test_domain", "method"], keys))
        for col in metric_cols:
            if col in group:
                row[f"mean_{col}"] = float(group[col].astype(float).mean())
                row[f"worst_{col}"] = float(group[col].astype(float).min()) if col in {"F1", "recall", "precision"} else float(group[col].astype(float).max())
        rows.append(row)
    return pd.DataFrame(rows)


def write_claim(out: Path, raw: pd.DataFrame, summary: pd.DataFrame) -> None:
    lines = ["# Calibration Staleness and Transfer Report", ""]
    lines.append("RF is trained only on the stated source domain/fold and evaluated unchanged on the target.")
    skipped = raw[~raw["status"].eq("ok")] if "status" in raw else pd.DataFrame()
    if not skipped.empty:
        lines += ["", "## Skipped", skipped[["protocol", "method", "reason"]].drop_duplicates().to_string(index=False)]
    if not summary.empty:
        lines += ["", "## Summary", summary.to_string(index=False)]
    lines += [
        "",
        "Claim rule: the no-label trust layer supports the stale-calibration niche only where it is close to or more stable than RF under LOSO/cross-detector transfer. Otherwise report RF as stronger when labels transfer cleanly.",
    ]
    (out / "calibration_staleness_claim_safe.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_metadata(out: Path, args: argparse.Namespace) -> None:
    meta = {
        "status": "success",
        "git_commit": git(["rev-parse", "HEAD"]),
        "inputs": {"yolov8s": args.yolov8s_feature_audit, "yolov8n": args.yolov8n_feature_audit},
        "note": "Cross-tracker RF is not inferred from summary tables; it requires candidate-level external-track feature audit.",
    }
    (out / "run_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def git(args: list[str]) -> str:
    try:
        return subprocess.check_output(["git", *args], text=True).strip()
    except Exception:
        return "unavailable"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-root", default="data/visdrone/VisDrone2019-VID-val")
    p.add_argument("--yolov8s-feature-audit", default="outputs/results/q1_final_corrected/yolov8s_main/feature_audit.csv")
    p.add_argument("--yolov8n-feature-audit", default="outputs/results/q1_final_corrected/yolov8n_main/feature_audit.csv")
    p.add_argument("--output-dir", default="outputs/results/q1_v5/calibration_staleness")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if args.dry_run:
        print(f"dry_run=ok output={out}")
        return
    y8s = read_det(args.yolov8s_feature_audit)
    y8n = read_det(args.yolov8n_feature_audit)
    frames = [
        loso(y8s, args.dataset_root, "yolov8s"),
        loso(y8n, args.dataset_root, "yolov8n"),
        cross_detector(y8n, y8s, args.dataset_root, "yolov8n", "yolov8s"),
        cross_detector(y8s, y8n, args.dataset_root, "yolov8s", "yolov8n"),
    ]
    raw = pd.concat(frames, ignore_index=True)
    summary = summarize(raw)
    raw.to_csv(out / "calibration_staleness_raw.csv", index=False)
    summary.to_csv(out / "calibration_staleness_summary.csv", index=False)
    write_claim(out, raw, summary)
    write_metadata(out, args)
    print(f"status=ok output={out / 'calibration_staleness_summary.csv'} rows={len(raw)}")


if __name__ == "__main__":
    main()
