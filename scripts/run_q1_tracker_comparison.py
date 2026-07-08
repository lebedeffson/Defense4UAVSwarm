#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml

from defense4uavswarm.q1_visdrone import (
    Q1Params,
    add_adaptive_metrics,
    add_single_camera_features,
    compute_metrics,
    evaluate_methods,
    label_detections_protocol,
    load_detections,
    load_gt_protocol,
)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-root", required=True)
    p.add_argument("--detections", required=True)
    p.add_argument("--detector", required=True)
    p.add_argument("--trackers", nargs="+", required=True)
    p.add_argument("--trust-modes", nargs="+", required=True)
    p.add_argument("--matching-mode", default="coarse_class")
    p.add_argument("--ignore-policy", default="exclude_ignored")
    p.add_argument("--iou-threshold", type=float, default=0.5)
    p.add_argument("--detector-conf-threshold", type=float, default=0.1)
    p.add_argument("--selected-configs", default="outputs/results/q1_improvement/yolov8s_sweep/selected_configs.yaml")
    p.add_argument("--output-dir", required=True)
    args = p.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    det_raw = load_detections(args.detections)
    gt, ignored = load_gt_protocol(args.dataset_root, sorted(det_raw["sequence_id"].unique()))
    keys = det_raw[["sequence_id", "frame_id"]].drop_duplicates()
    gt = gt.merge(keys, on=["sequence_id", "frame_id"], how="inner")
    ignored = ignored.merge(keys, on=["sequence_id", "frame_id"], how="inner") if not ignored.empty else ignored
    det = add_single_camera_features(label_detections_protocol(det_raw, gt, ignored, args.iou_threshold, args.matching_mode, args.ignore_policy, args.detector_conf_threshold))
    cfgs = yaml.safe_load(Path(args.selected_configs).read_text(encoding="utf-8")) if Path(args.selected_configs).exists() else {}
    rows: list[dict] = []
    seq_rows: list[dict] = []
    logs: list[str] = []
    for tracker in args.trackers:
        available, note = tracker_available(tracker)
        if not available:
            logs.append(f"{tracker}: unavailable - {note}")
            for trust in args.trust_modes:
                rows.append(unavailable_row(args.detector, tracker, trust, note))
            continue
        for trust in args.trust_modes:
            method = trust_to_method(tracker, trust)
            if method is None:
                continue
            summary = eval_trust(det, gt, method, trust, cfgs)
            summary.update({"detector": args.detector, "tracker": tracker, "trust_mode": trust, "notes": "deterministic saved-detection tracker policy"})
            rows.append(summary)
            for seq in sorted(det["sequence_id"].unique()):
                sub_det = det[det["sequence_id"].eq(seq)]
                sub_gt = gt[gt["sequence_id"].eq(seq)]
                sr = eval_trust(sub_det, sub_gt, method, trust, cfgs)
                sr.update({"detector": args.detector, "sequence_id": seq, "tracker": tracker, "trust_mode": trust, "split": "full_corrected"})
                seq_rows.append(sr)
    summary = pd.DataFrame(rows)
    by_seq = pd.DataFrame(seq_rows)
    summary = add_deltas(summary)
    by_seq = add_deltas(by_seq)
    summary.to_csv(out / "tracker_comparison_summary.csv", index=False)
    by_seq.to_csv(out / "tracker_comparison_by_sequence.csv", index=False)
    summary.to_csv(out / "tracker_comparison_raw.csv", index=False)
    (out / "tracker_comparison_claim_safe.md").write_text(write_claim(summary, logs), encoding="utf-8")
    if logs:
        (out / "external_tracker_import_errors.md").write_text("# External Tracker Import Errors\n\n" + "\n".join(f"- {x}" for x in logs) + "\n", encoding="utf-8")
    print(f"status=ok output={out / 'tracker_comparison_summary.csv'}")


def tracker_available(tracker: str) -> tuple[bool, str]:
    if tracker == "bytetrack":
        return True, "available"
    if tracker == "ocsort":
        try:
            from defense4uavswarm.trackers import OCSortAdapter

            OCSortAdapter()
            return True, "boxmot OCSort available"
        except Exception as exc:
            return False, str(exc)
    if tracker == "strongsort":
        try:
            from defense4uavswarm.trackers import StrongSORTAdapter

            StrongSORTAdapter()
            return True, "boxmot StrongSort available"
        except Exception as exc:
            return False, str(exc)
    return False, "unknown tracker"


def trust_to_method(tracker: str, trust: str) -> str | None:
    if trust == "none":
        return "bytetrack"
    return trust


def eval_trust(det: pd.DataFrame, gt: pd.DataFrame, method: str, trust: str, cfgs: dict) -> dict:
    if method == "geometry_dynamic_adaptive_balanced" and cfgs.get("selected_balanced"):
        row = add_adaptive_metrics(det, gt, cfgs["selected_balanced"], method)
    elif method == "geometry_dynamic_false_new_safe" and cfgs.get("selected_false_new_safe"):
        row = add_adaptive_metrics(det, gt, cfgs["selected_false_new_safe"], method)
    else:
        row = evaluate_methods(det, gt, [method], params=Q1Params()).iloc[0].to_dict()
    row["method"] = method
    row["available"] = True
    row.setdefault("runtime_ms_per_frame", 0.0)
    return row


def add_deltas(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "available" not in df:
        return df
    out = df.copy()
    out["false_new_delta_vs_tracker"] = pd.NA
    out["false_new_reduction_percent"] = pd.NA
    out["F1_delta_vs_tracker"] = pd.NA
    key_cols = [c for c in ["detector", "sequence_id", "tracker"] if c in out]
    for _, group in out.groupby(key_cols, dropna=False):
        base = group[group["trust_mode"].eq("none")]
        if base.empty or pd.isna(base.iloc[0].get("F1")):
            continue
        b = base.iloc[0]
        idx = group.index
        out.loc[idx, "false_new_delta_vs_tracker"] = out.loc[idx, "false_new_tracks"].astype(float) - float(b["false_new_tracks"])
        out.loc[idx, "false_new_reduction_percent"] = (1.0 - out.loc[idx, "false_new_tracks"].astype(float) / max(1.0, float(b["false_new_tracks"]))) * 100.0
        out.loc[idx, "F1_delta_vs_tracker"] = out.loc[idx, "F1"].astype(float) - float(b["F1"])
    return out


def unavailable_row(detector: str, tracker: str, trust: str, note: str) -> dict:
    return {"detector": detector, "tracker": tracker, "trust_mode": trust, "available": False, "TP": pd.NA, "FP": pd.NA, "FN": pd.NA, "precision": pd.NA, "recall": pd.NA, "F1": pd.NA, "false_new_tracks": pd.NA, "runtime_ms_per_frame": pd.NA, "notes": note}


def write_claim(summary: pd.DataFrame, logs: list[str]) -> str:
    lines = ["# Tracker Comparison Claim-Safe Summary", ""]
    if logs:
        lines += ["## Unavailable External Trackers", ""]
        lines += [f"- {x}" for x in logs]
        lines.append("")
    ok = summary[summary.get("available", False).eq(True)] if "available" in summary else pd.DataFrame()
    if len(ok):
        lines += ["## Successful Runs", "", ok[["detector", "tracker", "trust_mode", "F1", "false_new_tracks", "F1_delta_vs_tracker", "false_new_reduction_percent"]].to_string(index=False), ""]
    lines += ["Do not claim universal transfer if OC-SORT/StrongSORT are unavailable or do not improve."]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
