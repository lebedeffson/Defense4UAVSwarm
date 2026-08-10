#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from defense4uavswarm.q1_visdrone import (
    Q1Params,
    add_adaptive_metrics,
    compute_metrics,
    label_detections_protocol,
    load_detections,
    load_gt_protocol,
    method_acceptance,
)
from defense4uavswarm.v8_sim import vector_iou


COCO_ID_TO_NAME = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}


def read_mot_tracks(path: Path, tracker: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    seq = path.stem
    if not path.exists():
        return pd.DataFrame()
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            parts = [x.strip() for x in line.split(",")]
            if len(parts) < 8:
                continue
            frame_id = int(float(parts[0]))
            track_id = int(float(parts[1]))
            x, y, w, h = [float(v) for v in parts[2:6]]
            conf = float(parts[6])
            class_id = int(float(parts[7]))
            rows.append(
                {
                    "det_id": f"{tracker}_{seq}_{frame_id}_{track_id}_{line_no}",
                    "sequence_id": seq,
                    "frame_id": frame_id,
                    "external_track_id": track_id,
                    "tracklet_id": f"{tracker}_{seq}_{track_id}",
                    "x1": x,
                    "y1": y,
                    "x2": x + w,
                    "y2": y + h,
                    "bbox": [x, y, x + w, y + h],
                    "confidence": conf,
                    "class_id": class_id,
                    "class_name": COCO_ID_TO_NAME.get(class_id, str(class_id)),
                    "detector": tracker,
                }
            )
    return pd.DataFrame(rows)


def load_tracker_tracks(root: Path, tracker: str) -> pd.DataFrame:
    tracker_root = root / tracker
    frames = [read_mot_tracks(path, tracker) for path in sorted(tracker_root.glob("*.txt"))]
    frames = [f for f in frames if not f.empty]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def add_track_features(det: pd.DataFrame) -> pd.DataFrame:
    if det.empty:
        return det.copy()
    d = det.copy().sort_values(["sequence_id", "tracklet_id", "frame_id"]).reset_index(drop=True)
    d["bbox_area"] = (d["x2"] - d["x1"]).clip(lower=1) * (d["y2"] - d["y1"]).clip(lower=1)
    d["bbox_aspect_ratio"] = (d["x2"] - d["x1"]).clip(lower=1) / (d["y2"] - d["y1"]).clip(lower=1)
    d["num_detections_in_frame"] = d.groupby(["sequence_id", "frame_id"])["det_id"].transform("count")
    d["c_i"] = d["confidence"].astype(float).clip(0, 1)
    d["temporal_age"] = 0
    d["k_i"] = 0.35
    for _, group in d.groupby("tracklet_id", sort=False):
        prev_box = None
        age = 0
        for idx, row in group.sort_values("frame_id").iterrows():
            if prev_box is None:
                d.loc[idx, "temporal_age"] = 0
                d.loc[idx, "k_i"] = 0.35
            else:
                box = np.asarray([row.x1, row.y1, row.x2, row.y2], dtype=float)
                k = float(vector_iou(box, np.asarray([prev_box], dtype=float))[0])
                age += 1
                d.loc[idx, "temporal_age"] = age
                d.loc[idx, "k_i"] = max(0.05, min(1.0, k))
            prev_box = np.asarray([row.x1, row.y1, row.x2, row.y2], dtype=float)
    d["s_i"] = 1.0
    d["Q_i"] = np.minimum(d["c_i"], d["k_i"])
    d["confidence_new"] = d["confidence"] * (0.6 + 0.4 * d["Q_i"])
    d["track_status"] = np.where(d["temporal_age"].astype(int) <= 0, "new_candidate", "existing_track")
    return d


def eval_mode(det: pd.DataFrame, gt: pd.DataFrame, trust_mode: str, cfgs: dict[str, Any]) -> dict[str, Any]:
    if trust_mode == "none":
        row = compute_metrics(det, pd.Series(True, index=det.index), len(gt), max(1, gt[["sequence_id", "frame_id"]].drop_duplicates().shape[0]), "none")
    elif trust_mode == "geometry_dynamic_adaptive_balanced" and cfgs.get("selected_balanced"):
        row = add_adaptive_metrics(det, gt, cfgs["selected_balanced"], trust_mode)
    elif trust_mode == "geometry_dynamic_false_new_safe" and cfgs.get("selected_false_new_safe"):
        row = add_adaptive_metrics(det, gt, cfgs["selected_false_new_safe"], trust_mode)
    else:
        accepted = method_acceptance(det, trust_mode, Q1Params())
        row = compute_metrics(det, accepted, len(gt), max(1, gt[["sequence_id", "frame_id"]].drop_duplicates().shape[0]), trust_mode)
    row["trust_mode"] = trust_mode
    row["available"] = True
    return row


def add_deltas(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    out["F1_delta_vs_tracker"] = pd.NA
    out["false_new_delta_vs_tracker"] = pd.NA
    out["false_new_reduction_percent"] = pd.NA
    key_cols = [c for c in ["detector", "sequence_id", "tracker"] if c in out]
    for _, group in out.groupby(key_cols, dropna=False):
        base = group[group["trust_mode"].eq("none")]
        if base.empty or pd.isna(base.iloc[0].get("F1")):
            continue
        b = base.iloc[0]
        idx = group.index
        out.loc[idx, "F1_delta_vs_tracker"] = out.loc[idx, "F1"].astype(float) - float(b["F1"])
        out.loc[idx, "false_new_delta_vs_tracker"] = out.loc[idx, "false_new_tracks"].astype(float) - float(b["false_new_tracks"])
        out.loc[idx, "false_new_reduction_percent"] = (1.0 - out.loc[idx, "false_new_tracks"].astype(float) / max(1.0, float(b["false_new_tracks"]))) * 100.0
    return out


def unavailable_row(detector: str, tracker: str, trust_mode: str, reason: str) -> dict[str, Any]:
    return {
        "detector": detector,
        "tracker": tracker,
        "trust_mode": trust_mode,
        "available": False,
        "TP": pd.NA,
        "FP": pd.NA,
        "FN": pd.NA,
        "precision": pd.NA,
        "recall": pd.NA,
        "F1": pd.NA,
        "IDF1": pd.NA,
        "false_new_tracks": pd.NA,
        "false_new_tracks_per_100_frames": pd.NA,
        "track_breaks": pd.NA,
        "notes": reason,
    }


def write_claim(summary: pd.DataFrame) -> str:
    lines = ["# External Tracker Trust Evaluation", ""]
    ok = summary[summary["available"].astype(bool)] if "available" in summary else pd.DataFrame()
    if not ok.empty:
        cols = ["detector", "tracker", "trust_mode", "F1", "false_new_tracks", "F1_delta_vs_tracker", "false_new_reduction_percent"]
        lines += ["## Successful External Runs", "", ok[cols].to_string(index=False), ""]
    bad = summary[~summary["available"].astype(bool)] if "available" in summary else pd.DataFrame()
    if not bad.empty:
        lines += ["## Unavailable Runs", ""]
        for row in bad.drop_duplicates(["tracker", "notes"]).itertuples():
            lines.append(f"- {row.tracker}: {row.notes}")
        lines.append("")
    lines.append("StrongSORT must not be claimed if only the unavailable log is present.")
    return "\n".join(lines) + "\n"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-root", required=True)
    p.add_argument("--detections", required=True)
    p.add_argument("--tracks-root", required=True)
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
    det_ref = load_detections(args.detections)
    detector_name = str(det_ref["detector"].dropna().iloc[0]) if not det_ref.empty and "detector" in det_ref else Path(args.detections).stem.split("_")[0]
    gt, ignored = load_gt_protocol(args.dataset_root, sorted(det_ref["sequence_id"].unique()))
    keys = det_ref[["sequence_id", "frame_id"]].drop_duplicates()
    gt = gt.merge(keys, on=["sequence_id", "frame_id"], how="inner")
    ignored = ignored.merge(keys, on=["sequence_id", "frame_id"], how="inner") if not ignored.empty else ignored
    cfgs = yaml.safe_load(Path(args.selected_configs).read_text(encoding="utf-8")) if Path(args.selected_configs).exists() else {}

    rows: list[dict[str, Any]] = []
    seq_rows: list[dict[str, Any]] = []
    for tracker in args.trackers:
        track_df = load_tracker_tracks(Path(args.tracks_root), tracker)
        if track_df.empty:
            reason = "external track files missing or empty"
            if tracker == "strongsort" and (Path(args.tracks_root) / tracker / "strongsort_unavailable.md").exists():
                reason = "requires ReID model or embeddings; no ReID weights downloaded"
            for trust in args.trust_modes:
                rows.append(unavailable_row(detector_name, tracker, trust, reason))
            continue
        labeled = label_detections_protocol(track_df, gt, ignored, args.iou_threshold, args.matching_mode, args.ignore_policy, args.detector_conf_threshold)
        featured = add_track_features(labeled)
        for trust in args.trust_modes:
            row = eval_mode(featured, gt, trust, cfgs)
            row.update({"detector": detector_name, "tracker": tracker, "notes": "real external MOT track output"})
            rows.append(row)
            for seq in sorted(featured["sequence_id"].unique()):
                sub_det = featured[featured["sequence_id"].eq(seq)]
                sub_gt = gt[gt["sequence_id"].eq(seq)]
                sr = eval_mode(sub_det, sub_gt, trust, cfgs)
                sr.update({"detector": detector_name, "sequence_id": seq, "tracker": tracker, "split": "full_corrected", "notes": "real external MOT track output"})
                seq_rows.append(sr)

    summary = add_deltas(pd.DataFrame(rows))
    by_seq = add_deltas(pd.DataFrame(seq_rows)) if seq_rows else pd.DataFrame()
    summary.to_csv(out / "tracker_comparison_summary.csv", index=False)
    summary.to_csv(out / "tracker_comparison_raw.csv", index=False)
    by_seq.to_csv(out / "tracker_comparison_by_sequence.csv", index=False)
    (out / "tracker_comparison_claim_safe.md").write_text(write_claim(summary), encoding="utf-8")
    print(f"status=ok output={out / 'tracker_comparison_summary.csv'}")


if __name__ == "__main__":
    main()
