from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import pandas as pd
import yaml

from defense4uavswarm.q1_v5.evaluation_matching import MatchingConfig
from defense4uavswarm.q1_v5.frame_manifest import add_gt_presence, build_visdrone_frame_manifest, merge_manifest_dimensions
from defense4uavswarm.q1_v5.initiation_gate import GateConfig, assign_episode_ids, split_duplicate_observation_tracklets
from defense4uavswarm.q1_visdrone import load_gt_protocol
from scripts.q1_v54.run_corrected_operating_curves import load_candidates


PROTOCOL_ID = "q1_selective_quarantine_v22_risk_prioritized_two_stage"


def read_config(path: str | Path) -> dict[str, Any]:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}


def results_root(cfg: dict[str, Any], output_dir: str | None = None) -> Path:
    return Path(output_dir or cfg.get("results_root", "outputs/results/q1_selective_quarantine_v22_risk_prioritized"))


def prepare_output(path: Path, *, overwrite: bool, dry_run: bool = False) -> None:
    if path.exists() and any(path.iterdir()) and not overwrite and not dry_run:
        raise SystemExit(f"Output exists; use --overwrite: {path}")
    path.mkdir(parents=True, exist_ok=True)


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


def load_frozen_inputs(cfg: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    det = load_candidates(cfg)
    gt, ignored = load_gt_protocol(cfg["dataset_root"], sorted(det["sequence_id"].unique()))
    manifest = add_gt_presence(build_visdrone_frame_manifest(cfg["dataset_root"], sorted(det["sequence_id"].unique())), gt)
    det = merge_manifest_dimensions(det, manifest)
    det = split_duplicate_observation_tracklets(det)
    det = assign_episode_ids(det, gate_config(cfg).max_track_gap)
    return det, gt, ignored, manifest


def run_metadata(extra: dict[str, Any] | None = None) -> dict[str, Any]:
    meta = {
        "status": "success",
        "protocol_id": PROTOCOL_ID,
        "git_commit": git(["rev-parse", "HEAD"]),
        "branch": git(["branch", "--show-current"]),
        "working_tree_dirty": bool(git(["status", "--short"])),
    }
    if extra:
        meta.update(extra)
    return meta


def write_metadata(path: Path, extra: dict[str, Any] | None = None) -> None:
    path.write_text(json.dumps(run_metadata(extra), indent=2, sort_keys=True), encoding="utf-8")


def git(args: list[str]) -> str:
    try:
        return subprocess.check_output(["git", *args], text=True).strip()
    except Exception:
        return "unavailable"
