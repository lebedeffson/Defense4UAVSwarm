#!/usr/bin/env python
from __future__ import annotations

import argparse
import gc
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from defense4uavswarm.q1_v5.frame_manifest import add_gt_presence, build_visdrone_frame_manifest, merge_manifest_dimensions
from defense4uavswarm.q1_v5.initiation_gate import (
    assign_episode_ids,
    bayesian_terminal_gate_result,
    confidence_initiation_gate_result,
    m_of_n_confirmation_result,
    split_duplicate_observation_tracklets,
)
from defense4uavswarm.q1_v5.rf_v21 import (
    RFBudgetSpec,
    RFSpec,
    add_episode_labels,
    build_episode_feature_table,
    false_probability,
    rf_budgeted_gate_result,
    rf_unbounded_gate_result,
    train_random_forest,
)
from defense4uavswarm.q1_v5.selective_quarantine import SelectiveTrustQuarantineConfig, selective_quarantine_gate_result
from defense4uavswarm.q1_visdrone import load_gt_protocol
from scripts.q1_v54.run_corrected_operating_curves import gate_config, load_candidates, matching_config


REPO_ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/q1_v54/bytetrack.yaml")
    p.add_argument("--output-dir", default="outputs/q1_practical_closure_v7/environment")
    p.add_argument("--repetitions", type=int, default=30)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--sequence", default=None)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()
    out = Path(args.output_dir)
    if out.exists() and args.overwrite:
        for name in ["runtime.csv", "runtime_manifest.json"]:
            path = out / name
            if path.exists():
                path.unlink()
    out.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")) or {}
    det, gt, ignored, manifest = prepare_inputs(cfg)
    sequence = args.sequence or sorted(det["sequence_id"].unique())[0]
    bench_det = det[det["sequence_id"].eq(sequence)].copy()
    bench_manifest = manifest[manifest["sequence_id"].eq(sequence)].copy()
    methods = build_methods(cfg, det, gt, ignored, manifest, bench_det, bench_manifest)
    rows = []
    for method, fn, boundary in methods:
        rows.append(measure(method, fn, boundary, det, manifest, args.repetitions, args.warmup))
    runtime = pd.DataFrame(rows)
    runtime.to_csv(out / "runtime.csv", index=False)
    manifest_payload = {
        "status": "success",
        "git_commit": git(["rev-parse", "HEAD"]),
        "git_branch": git(["branch", "--show-current"]),
        "tracker": cfg.get("tracker", "bytetrack"),
        "repetitions": int(args.repetitions),
        "warmup": int(args.warmup),
        "sequence_scope": sequence,
        "detector_and_base_tracker_excluded": True,
        "training_excluded": True,
    }
    (out / "runtime_manifest.json").write_text(json.dumps(manifest_payload, indent=2, sort_keys=True), encoding="utf-8")
    print(f"status=ok output={out / 'runtime.csv'} rows={len(runtime)}")


def prepare_inputs(cfg: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    det = load_candidates(cfg)
    gt, ignored = load_gt_protocol(cfg["dataset_root"], sorted(det["sequence_id"].unique()))
    manifest = add_gt_presence(build_visdrone_frame_manifest(cfg["dataset_root"], sorted(det["sequence_id"].unique())), gt)
    det = merge_manifest_dimensions(det, manifest)
    det = split_duplicate_observation_tracklets(det)
    det = assign_episode_ids(det, gate_config(cfg).max_track_gap)
    return det, gt, ignored, manifest


def build_methods(
    cfg: dict[str, Any],
    train_det: pd.DataFrame,
    gt: pd.DataFrame,
    ignored: pd.DataFrame,
    train_manifest: pd.DataFrame,
    det: pd.DataFrame,
    manifest: pd.DataFrame,
) -> list[tuple[str, Callable[[], Any], str]]:
    selected = load_selected_configs(Path("outputs/results/q1_selective_quarantine_v21/loso/bytetrack/selected_configs_by_fold.json"))
    gc = gate_config(cfg)
    mc = matching_config(cfg)
    rf_hyper = pd.read_csv("outputs/q1_practical_closure_v7/rf_v21/selected_hyperparameters.csv").iloc[0].to_dict()
    rf_budget = pd.read_csv("outputs/q1_practical_closure_v7/rf_v21/selected_budget_parameters.csv").iloc[0].to_dict()
    episode_train = add_episode_labels(train_det, gt, ignored, train_manifest, mc, gc)
    rf_spec = RFSpec(
        n_estimators=int(rf_hyper["n_estimators"]),
        max_depth=int(rf_hyper["max_depth"]),
        min_samples_leaf=int(rf_hyper["min_samples_leaf"]),
        max_features=str(rf_hyper["max_features"]),
        class_weight=str(rf_hyper["class_weight"]),
        decision_threshold=float(rf_hyper["decision_threshold"]),
    )
    rf_budget_spec = RFBudgetSpec(
        decision_threshold=float(rf_budget["decision_threshold"]),
        intervention_fraction=float(rf_budget["intervention_fraction"]),
        budget_capacity=float(rf_budget["budget_capacity"]),
        hard_budget_enabled=bool(rf_budget["hard_budget_enabled"]),
    )
    model = train_random_forest(episode_train, rf_spec)

    confidence_params = json.loads(selected["confidence_initiation_gate"])
    mofn_params = json.loads(selected["m_of_n_confirmation"])
    bayes_params = json.loads(selected["bayesian_fixed_terminal"])
    selective_params = json.loads(selected["selective_trust_quarantine"])

    def rf_features_and_prob() -> tuple[pd.DataFrame, np.ndarray]:
        features = build_episode_feature_table(det)
        return features, false_probability(model, features)

    return [
        (
            "confidence_threshold",
            lambda: confidence_initiation_gate_result(det, float(confidence_params["threshold"]), gc),
            "Gate layer only; detector and base tracker excluded.",
        ),
        (
            "m_of_n",
            lambda: m_of_n_confirmation_result(det, int(mofn_params["M"]), int(mofn_params["N"]), float(mofn_params["confidence_threshold"]), gc),
            "Gate layer only; detector and base tracker excluded.",
        ),
        (
            "bayesian_confirmation",
            lambda: bayesian_terminal_gate_result(det, threshold=float(bayes_params["threshold"]), cfg=gc),
            "Gate layer only; detector and base tracker excluded.",
        ),
        (
            "rf_unbounded",
            lambda: (lambda fp: rf_unbounded_gate_result(det, fp[0], fp[1], rf_spec, gc))(rf_features_and_prob()),
            "Episode feature construction, RF inference, and gate application; RF training, detector, and base tracker excluded.",
        ),
        (
            "rf_budgeted",
            lambda: (lambda fp: rf_budgeted_gate_result(det, fp[0], fp[1], rf_budget_spec, gc))(rf_features_and_prob()),
            "Episode feature construction, RF inference, budget simulation, and gate application; RF training, detector, and base tracker excluded.",
        ),
        (
            "selective_quarantine_v21",
            lambda: selective_quarantine_gate_result(det, manifest, SelectiveTrustQuarantineConfig.from_mapping(selective_params)),
            "Selective quarantine layer only; detector and base tracker excluded.",
        ),
    ]


def load_selected_configs(path: Path) -> dict[str, str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    selected: dict[str, str] = {}
    for row in payload:
        if int(row.get("outer_fold", -1)) != 0:
            continue
        method = str(row["method"])
        if method in {"confidence_initiation_gate", "m_of_n_confirmation", "bayesian_fixed_terminal", "selective_trust_quarantine"}:
            selected[method] = str(row["selected_parameter_json"])
    missing = {"confidence_initiation_gate", "m_of_n_confirmation", "bayesian_fixed_terminal", "selective_trust_quarantine"} - set(selected)
    if missing:
        raise RuntimeError(f"Missing selected runtime configs: {sorted(missing)}")
    return selected


def measure(method: str, fn: Callable[[], Any], boundary: str, det: pd.DataFrame, manifest: pd.DataFrame, repetitions: int, warmup: int) -> dict[str, Any]:
    for _ in range(int(warmup)):
        fn()
    gc_enabled = gc.isenabled()
    gc.disable()
    times = []
    try:
        for _ in range(int(repetitions)):
            start = time.perf_counter_ns()
            fn()
            times.append(time.perf_counter_ns() - start)
    finally:
        if gc_enabled:
            gc.enable()
    arr_ms = np.asarray(times, dtype=float) / 1e6
    frames = int(len(manifest.drop_duplicates(["sequence_id", "frame_id"])))
    candidates = int(len(det))
    return {
        "tracker": "ByteTrack",
        "method": method,
        "repetitions": int(repetitions),
        "warmup_repetitions": int(warmup),
        "measurement_boundary": boundary,
        "frames": frames,
        "candidates": candidates,
        "mean_ms": float(arr_ms.mean()),
        "median_ms": float(np.median(arr_ms)),
        "p95_ms": float(np.quantile(arr_ms, 0.95)),
        "mean_ms_per_frame": float(arr_ms.mean() / max(1, frames)),
        "mean_us_per_candidate": float(arr_ms.mean() * 1000.0 / max(1, candidates)),
    }


def git(args: list[str]) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True).strip()
    except Exception:
        return "unavailable"


if __name__ == "__main__":
    main()
