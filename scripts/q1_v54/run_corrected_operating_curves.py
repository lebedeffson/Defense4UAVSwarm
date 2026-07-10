#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from defense4uavswarm.q1_v5.evaluation_matching import MatchingConfig, evaluate_acceptance_mask
from defense4uavswarm.q1_v5.frame_manifest import add_gt_presence, build_visdrone_frame_manifest, manifest_sha256, merge_manifest_dimensions
from defense4uavswarm.q1_v5.initiation_gate import GateConfig, assign_episode_ids, bayesian_terminal_gate, confidence_initiation_gate, m_of_n_confirmation, terminalize_initiation
from defense4uavswarm.q1_visdrone import Q1Params, load_gt_protocol, method_acceptance
from defense4uavswarm.v8_sim import vector_iou


COCO_ID_TO_NAME = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}


def read_config(path: str | Path) -> dict[str, Any]:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}


def matching_config(cfg: dict[str, Any]) -> MatchingConfig:
    m = cfg.get("matching", {})
    return MatchingConfig(
        iou_threshold=float(m.get("iou_threshold", 0.5)),
        class_matching=str(m.get("class_matching", "coarse_class")),
        ignored_policy=str(m.get("ignored_policy", "exclude_ignored")),
        matcher_id=str(m.get("matcher_id", "q1_v542_max_cardinality_iou_v1")),
        confirmation_window_frames=int(m.get("confirmation_window_frames", 3)),
    )


def gate_config(cfg: dict[str, Any]) -> GateConfig:
    g = cfg.get("gate", {})
    return GateConfig(maximum_pending_age_frames=int(g.get("maximum_pending_age_frames", 6)), max_track_gap=int(g.get("max_track_gap", 1)))


def load_candidates(cfg: dict[str, Any]) -> pd.DataFrame:
    source = cfg.get("candidate_source", "feature_audit")
    if source == "feature_audit":
        return pd.read_csv(cfg["feature_audit"])
    if source == "mot_tracks":
        root = Path(cfg["tracks_root"])
        tracker = str(cfg["tracker"])
        tracker_root = root / tracker
        if not tracker_root.exists():
            raise FileNotFoundError(f"No tracker root for {tracker}: {tracker_root}")
        frames = [_read_mot_tracks(path, tracker) for path in sorted(tracker_root.glob("*.txt"))]
        frames = [f for f in frames if not f.empty]
        if not frames:
            raise RuntimeError(f"No MOT track rows found for {tracker}: {tracker_root}")
        return _add_track_features(pd.concat(frames, ignore_index=True))
    raise ValueError(f"Unknown candidate_source: {source}")


def _read_mot_tracks(path: Path, tracker: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    seq = path.stem
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
                    "class_id": class_id,
                    "class_name": COCO_ID_TO_NAME.get(class_id, str(class_id)),
                    "confidence": conf,
                    "detector": tracker,
                    "x1": x,
                    "y1": y,
                    "x2": x + w,
                    "y2": y + h,
                }
            )
    return pd.DataFrame(rows)


def _add_track_features(det: pd.DataFrame) -> pd.DataFrame:
    d = det.copy().sort_values(["sequence_id", "tracklet_id", "frame_id"]).reset_index(drop=True)
    d["bbox_area"] = (d["x2"] - d["x1"]).clip(lower=1) * (d["y2"] - d["y1"]).clip(lower=1)
    d["bbox_aspect_ratio"] = (d["x2"] - d["x1"]).clip(lower=1) / (d["y2"] - d["y1"]).clip(lower=1)
    d["num_detections_in_frame"] = d.groupby(["sequence_id", "frame_id"])["det_id"].transform("count")
    d["c_i"] = d["confidence"].astype(float).clip(0, 1)
    d["temporal_age"] = 0
    d["k_i"] = 0.35
    for _, group in d.groupby(["sequence_id", "tracklet_id"], sort=False):
        prev_box = None
        prev_frame = None
        age = 0
        for idx, row in group.sort_values("frame_id").iterrows():
            box = row[["x1", "y1", "x2", "y2"]].to_numpy(dtype=float)
            if prev_box is not None and prev_frame is not None:
                age = age + 1 if int(row.frame_id) - int(prev_frame) <= 2 else 0
                d.loc[idx, "temporal_age"] = age
                d.loc[idx, "k_i"] = float(vector_iou(box, prev_box.reshape(1, 4))[0])
            prev_box = box
            prev_frame = int(row.frame_id)
    d["s_i"] = 1.0
    d["Q_i"] = d[["c_i", "k_i"]].min(axis=1)
    d["confidence_new"] = d["confidence"] * (0.6 + 0.4 * d["Q_i"])
    d["track_status"] = "external_track"
    return d


def run_method(name: str, det: pd.DataFrame, accepted: pd.Series, gt: pd.DataFrame, ignored: pd.DataFrame, frame_index: pd.DataFrame, mc: MatchingConfig, parameter: str, value: Any) -> dict[str, Any]:
    _, events, metrics = evaluate_acceptance_mask(det, accepted, gt, frame_index, mc, ignored)
    return {"method": name, "parameter": parameter, "parameter_value": value, **metrics, "num_track_events": len(events)}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/q1_v54/visdrone_corrected.yaml")
    p.add_argument("--output-dir", default="outputs/results/q1_v542/curves/bytetrack")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    cfg = read_config(args.config)
    out = Path(args.output_dir)
    if out.exists() and any(out.iterdir()) and not args.overwrite and not args.dry_run:
        raise SystemExit(f"Output exists; use --overwrite: {out}")
    out.mkdir(parents=True, exist_ok=True)
    (out / "resolved_config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    if args.dry_run:
        print(f"dry_run=ok output={out}")
        return
    det = load_candidates(cfg)
    gt, ignored = load_gt_protocol(cfg["dataset_root"], sorted(det["sequence_id"].unique()))
    frame_manifest = add_gt_presence(build_visdrone_frame_manifest(cfg["dataset_root"], sorted(det["sequence_id"].unique())), gt)
    det = merge_manifest_dimensions(det, frame_manifest)
    det = assign_episode_ids(det, gate_config(cfg).max_track_gap)
    frame_index = frame_manifest[["sequence_id", "frame_id", "image_path", "image_width", "image_height", "has_gt"]].copy()
    mc = matching_config(cfg)
    gc = gate_config(cfg)
    rows: list[dict[str, Any]] = []
    rows.append(run_method("tracker_baseline", det, pd.Series(True, index=det.index), gt, ignored, frame_index, mc, "none", 0))
    legacy_trigger = terminalize_initiation(det, method_acceptance(det, "geometry_dynamic_no_multiagent", Q1Params()), gc)
    rows.append(run_method("legacy_trust_terminalized", det, legacy_trigger, gt, ignored, frame_index, mc, "legacy_trigger", 0))
    for th in cfg.get("grids", {}).get("confidence_threshold", [0.2]):
        rows.append(run_method("confidence_initiation_gate", det, confidence_initiation_gate(det, float(th), gc), gt, ignored, frame_index, mc, "threshold", th))
    for spec in cfg.get("grids", {}).get("m_of_n", []):
        accepted = m_of_n_confirmation(det, int(spec["M"]), int(spec["N"]), float(spec["confidence_threshold"]), gc)
        rows.append(run_method("m_of_n_confirmation", det, accepted, gt, ignored, frame_index, mc, f"M={spec['M']};N={spec['N']};conf", spec["confidence_threshold"]))
    for th in cfg.get("grids", {}).get("bayesian_threshold", [1.1]):
        accepted = bayesian_terminal_gate(det, threshold=float(th), cfg=gc)
        rows.append(run_method("bayesian_fixed_terminal", det, accepted, gt, ignored, frame_index, mc, "threshold", th))
    result = pd.DataFrame(rows)
    result.to_csv(out / "corrected_operating_points.csv", index=False)
    frame_manifest.to_csv(out / "frame_manifest.csv", index=False)
    (out / "corrected_operating_claim_safe.md").write_text(write_claim(result), encoding="utf-8")
    (out / "run_metadata.json").write_text(
        json.dumps(
            {
                "status": "success",
                "git_commit": git(["rev-parse", "HEAD"]),
                "branch": git(["branch", "--show-current"]),
                "protocol_id": "q1_v542_article_grade_partial",
                "matcher_id": mc.matcher_id,
                "metric_source": "recomputed_after_acceptance",
                "manifest_sha256": manifest_sha256(frame_manifest),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"status=ok output={out / 'corrected_operating_points.csv'} rows={len(result)}")


def write_claim(result: pd.DataFrame) -> str:
    lines = ["# Corrected Operating Curves", "", "Metrics are recomputed after each acceptance mask with one-to-one GT matching. Old `eval_is_tp` columns are not used as metric source.", ""]
    display_cols = ["method", "parameter", "parameter_value", "F1", "false_new_tracks", "observed_false_track_occupancy_rows", "metric_source"]
    lines.append(result[display_cols].to_string(index=False))
    lines += ["", "Use these corrected tables for new article claims. Frozen legacy tables remain regression audits only."]
    return "\n".join(lines) + "\n"


def git(args: list[str]) -> str:
    try:
        return subprocess.check_output(["git", *args], text=True).strip()
    except Exception:
        return "unavailable"


if __name__ == "__main__":
    main()
