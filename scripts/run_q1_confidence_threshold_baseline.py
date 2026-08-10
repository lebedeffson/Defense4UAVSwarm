#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from defense4uavswarm.q1_visdrone import (
    add_single_camera_features,
    compute_metrics,
    label_detections_protocol,
    load_detections,
    load_gt_protocol,
)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-root", required=True)
    p.add_argument("--detections", required=True)
    p.add_argument("--detector", required=True)
    p.add_argument("--thresholds", nargs="+", type=float, required=True)
    p.add_argument("--matching-mode", default="coarse_class")
    p.add_argument("--ignore-policy", default="exclude_ignored")
    p.add_argument("--iou-threshold", type=float, default=0.5)
    p.add_argument("--detector-conf-threshold", type=float, default=0.1)
    p.add_argument("--reference-summary", default="")
    p.add_argument("--output-dir", required=True)
    args = p.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    det_raw = load_detections(args.detections)
    sequence_ids = sorted(det_raw["sequence_id"].unique()) if not det_raw.empty else None
    gt, ignored = load_gt_protocol(args.dataset_root, sequence_ids)
    if not det_raw.empty and not gt.empty:
        keys = det_raw[["sequence_id", "frame_id"]].drop_duplicates()
        gt = gt.merge(keys, on=["sequence_id", "frame_id"], how="inner")
        ignored = ignored.merge(keys, on=["sequence_id", "frame_id"], how="inner") if not ignored.empty else ignored

    labeled = label_detections_protocol(
        det_raw,
        gt,
        ignored,
        iou_threshold=args.iou_threshold,
        matching_mode=args.matching_mode,
        ignore_policy=args.ignore_policy,
        detector_conf_threshold=args.detector_conf_threshold,
    )
    det = add_single_camera_features(labeled)
    reference = load_reference(args.reference_summary, args.detector)

    summary_rows: list[dict[str, Any]] = []
    by_sequence_rows: list[dict[str, Any]] = []
    raw_rows: list[dict[str, Any]] = []
    num_frames = frame_count(gt)
    for threshold in args.thresholds:
        accepted = det["confidence"].astype(float) >= float(threshold)
        row = compute_metrics(det, accepted, len(gt), num_frames, f"confidence_threshold_{threshold:g}")
        row = add_common_fields(row, args.detector, threshold, reference)
        summary_rows.append(row)

        for seq in sorted(det["sequence_id"].unique()) if not det.empty else []:
            sub_det = det[det["sequence_id"].eq(seq)]
            sub_gt = gt[gt["sequence_id"].eq(seq)]
            sub_acc = sub_det["confidence"].astype(float) >= float(threshold)
            sr = compute_metrics(sub_det, sub_acc, len(sub_gt), frame_count(sub_gt), f"confidence_threshold_{threshold:g}")
            sr = add_common_fields(sr, args.detector, threshold, reference)
            sr["sequence_id"] = seq
            by_sequence_rows.append(sr)
            raw_rows.append(sr.copy())

    summary = pd.DataFrame(summary_rows)
    by_sequence = pd.DataFrame(by_sequence_rows)
    raw = pd.DataFrame(raw_rows)

    summary.to_csv(out / "confidence_threshold_summary.csv", index=False)
    by_sequence.to_csv(out / "confidence_threshold_by_sequence.csv", index=False)
    raw.to_csv(out / "confidence_threshold_raw.csv", index=False)
    write_claim_safe(out / "confidence_threshold_claim_safe.md", summary, reference, args)
    print(f"status=ok detector={args.detector} rows={len(summary)} output={out}")


def frame_count(gt: pd.DataFrame) -> int:
    if gt.empty:
        return 0
    return int(gt[["sequence_id", "frame_id"]].drop_duplicates().shape[0])


def add_common_fields(row: dict[str, Any], detector: str, threshold: float, reference: dict[str, float]) -> dict[str, Any]:
    out = dict(row)
    out["detector"] = detector
    out["threshold"] = float(threshold)
    out["false_new_tracks"] = int(out.get("false_new_tracks", 0))
    out["false_new_per_100_frames"] = float(out.get("false_new_tracks_per_100_frames", 0.0))
    if reference:
        out["F1_delta_vs_bytetrack"] = float(out["F1"]) - reference.get("bytetrack_F1", float("nan"))
        out["false_new_delta_vs_bytetrack"] = float(out["false_new_tracks"]) - reference.get("bytetrack_false_new", float("nan"))
    else:
        out["F1_delta_vs_bytetrack"] = pd.NA
        out["false_new_delta_vs_bytetrack"] = pd.NA
    out["notes"] = "fixed detector confidence threshold; no temporal/geometric trust rule"
    return out


def load_reference(path: str, detector: str) -> dict[str, float]:
    candidates = []
    if path:
        candidates.append(Path(path))
    candidates.append(Path("outputs/results/q1_final_corrected") / f"{detector}_main" / "main_comparison_table.csv")
    for candidate in candidates:
        if not candidate.exists():
            continue
        df = pd.read_csv(candidate)
        if "method" not in df:
            continue
        bt = df[df["method"].astype(str).eq("bytetrack")]
        trust = df[df["method"].astype(str).eq("geometry_dynamic_no_multiagent")]
        out: dict[str, float] = {}
        if not bt.empty:
            out["bytetrack_F1"] = float(bt.iloc[0]["F1"])
            out["bytetrack_false_new"] = float(bt.iloc[0]["false_new_tracks"])
        if not trust.empty:
            out["trust_F1"] = float(trust.iloc[0]["F1"])
            out["trust_false_new"] = float(trust.iloc[0]["false_new_tracks"])
        return out
    return {}


def write_claim_safe(path: Path, summary: pd.DataFrame, reference: dict[str, float], args: argparse.Namespace) -> None:
    lines = [
        "# Confidence Threshold Baseline Claim-Safe Notes",
        "",
        "Protocol:",
        f"- detector: `{args.detector}`",
        f"- matching_mode: `{args.matching_mode}`",
        f"- ignore_policy: `{args.ignore_policy}`",
        f"- iou_threshold: `{args.iou_threshold}`",
        f"- detector_conf_threshold before evaluation: `{args.detector_conf_threshold}`",
        "",
    ]
    if summary.empty:
        lines += ["No threshold rows were produced."]
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return

    best_f1 = summary.sort_values(["F1", "false_new_tracks"], ascending=[False, True]).iloc[0]
    lines += [
        "Threshold sweep:",
        "",
        "```text",
        summary[["threshold", "F1", "false_new_tracks", "precision", "recall"]].to_string(index=False),
        "```",
        "",
        f"Best confidence threshold by F1: `{best_f1['threshold']}` with F1={float(best_f1['F1']):.6f}, false_new={int(best_f1['false_new_tracks'])}.",
    ]

    if "trust_false_new" in reference and "trust_F1" in reference:
        trust_false = reference["trust_false_new"]
        trust_f1 = reference["trust_F1"]
        reaches = summary[summary["false_new_tracks"].astype(float) <= trust_false].copy()
        close = summary.iloc[(summary["F1"].astype(float) - trust_f1).abs().argsort().iloc[0]]
        better = summary[(summary["F1"].astype(float) >= trust_f1) & (summary["false_new_tracks"].astype(float) <= trust_false)]
        lines += [
            "",
            "Questions:",
            "",
            f"1. Can a simple confidence threshold reach the trust-layer false_new level ({trust_false:.0f})?",
            answer_reaches(reaches, trust_f1),
            "2. What happens to F1 at that threshold?",
            answer_f1_at_reaches(reaches),
            "3. Is there a point where confidence threshold is better than the trust layer?",
            "Yes." if len(better) else "No under the reported thresholds.",
            "4. Is there a point where trust layer is better at close F1?",
            (
                f"Closest-F1 threshold is {close['threshold']} with F1={float(close['F1']):.6f}, "
                f"false_new={int(close['false_new_tracks'])}; trust has F1={trust_f1:.6f}, false_new={trust_false:.0f}."
            ),
            "5. Safe article claim:",
            safe_claim(summary, trust_f1, trust_false),
        ]
    else:
        lines += [
            "",
            "Trust-layer reference was not found; do not claim superiority over trust from this file alone.",
        ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def answer_reaches(reaches: pd.DataFrame, trust_f1: float) -> str:
    if reaches.empty:
        return "No. None of the tested thresholds reaches the trust-layer false_new count."
    best = reaches.sort_values(["F1", "false_new_tracks"], ascending=[False, True]).iloc[0]
    return f"Yes, at threshold {best['threshold']} or above; best such F1={float(best['F1']):.6f} (delta vs trust {float(best['F1']) - trust_f1:+.6f})."


def answer_f1_at_reaches(reaches: pd.DataFrame) -> str:
    if reaches.empty:
        return "Not applicable."
    best = reaches.sort_values(["F1", "false_new_tracks"], ascending=[False, True]).iloc[0]
    return f"F1 drops to {float(best['F1']):.6f} at the best threshold that reaches the trust false_new level."


def safe_claim(summary: pd.DataFrame, trust_f1: float, trust_false: float) -> str:
    better = summary[(summary["F1"].astype(float) >= trust_f1) & (summary["false_new_tracks"].astype(float) <= trust_false)]
    if len(better):
        return "The confidence threshold baseline can match or exceed the trust point in this sweep; present the trust layer as an interpretable multi-feature rule, not as numerically dominant."
    reaches = summary[summary["false_new_tracks"].astype(float) <= trust_false]
    if len(reaches):
        return "A simple confidence threshold can reduce false-new tracks, but reaching the trust false-new range costs additional F1 in this sweep."
    return "The trust layer shifts the operating point toward fewer false-new tracks without being reducible to the tested fixed confidence thresholds."


if __name__ == "__main__":
    main()
