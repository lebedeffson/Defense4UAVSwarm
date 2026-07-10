#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from defense4uavswarm.q1_v5.evaluation_matching import MatchingConfig, build_frame_index, evaluate_acceptance_mask
from defense4uavswarm.q1_v5.initiation_gate import GateConfig, bayesian_terminal_gate, confidence_initiation_gate, m_of_n_confirmation, terminalize_initiation
from defense4uavswarm.q1_visdrone import Q1Params, load_gt_protocol, method_acceptance


def read_config(path: str | Path) -> dict[str, Any]:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}


def matching_config(cfg: dict[str, Any]) -> MatchingConfig:
    m = cfg.get("matching", {})
    return MatchingConfig(
        iou_threshold=float(m.get("iou_threshold", 0.5)),
        class_matching=str(m.get("class_matching", "coarse_class")),
        ignored_policy=str(m.get("ignored_policy", "exclude_ignored")),
        confirmation_window_frames=int(m.get("confirmation_window_frames", 3)),
    )


def gate_config(cfg: dict[str, Any]) -> GateConfig:
    g = cfg.get("gate", {})
    return GateConfig(maximum_pending_age_frames=int(g.get("maximum_pending_age_frames", 6)), max_track_gap=int(g.get("max_track_gap", 1)))


def run_method(name: str, det: pd.DataFrame, accepted: pd.Series, gt: pd.DataFrame, ignored: pd.DataFrame, frame_index: pd.DataFrame, mc: MatchingConfig, parameter: str, value: Any) -> dict[str, Any]:
    _, events, metrics = evaluate_acceptance_mask(det, accepted, gt, frame_index, mc, ignored)
    return {"method": name, "parameter": parameter, "parameter_value": value, **metrics, "num_track_events": len(events)}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/q1_v54/visdrone_corrected.yaml")
    p.add_argument("--output-dir", default="outputs/results/q1_v54/corrected_operating_curves/bytetrack")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    cfg = read_config(args.config)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "resolved_config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    if args.dry_run:
        print(f"dry_run=ok output={out}")
        return
    det = pd.read_csv(cfg["feature_audit"])
    gt, ignored = load_gt_protocol(cfg["dataset_root"], sorted(det["sequence_id"].unique()))
    frame_index = build_frame_index(gt, det)
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
    (out / "corrected_operating_claim_safe.md").write_text(write_claim(result), encoding="utf-8")
    (out / "run_metadata.json").write_text(json.dumps({"status": "success", "git_commit": git(["rev-parse", "HEAD"]), "metric_source": "recomputed_after_acceptance"}, indent=2), encoding="utf-8")
    print(f"status=ok output={out / 'corrected_operating_points.csv'} rows={len(result)}")


def write_claim(result: pd.DataFrame) -> str:
    lines = ["# Corrected Operating Curves", "", "Metrics are recomputed after each acceptance mask with one-to-one GT matching. Old `eval_is_tp` columns are not used as metric source.", ""]
    lines.append(result[["method", "parameter", "parameter_value", "F1", "false_new_tracks", "observed_false_track_occupancy_frames", "metric_source"]].to_string(index=False))
    lines += ["", "Use these corrected tables for new article claims. Frozen legacy tables remain regression audits only."]
    return "\n".join(lines) + "\n"


def git(args: list[str]) -> str:
    try:
        return subprocess.check_output(["git", *args], text=True).strip()
    except Exception:
        return "unavailable"


if __name__ == "__main__":
    main()
