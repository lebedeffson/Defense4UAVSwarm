#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from defense4uavswarm.q1_v5.evaluation_matching import MatchingConfig, evaluate_gate_result
from defense4uavswarm.q1_v5.frame_manifest import add_gt_presence, build_visdrone_frame_manifest, merge_manifest_dimensions
from defense4uavswarm.q1_v5.initiation_gate import (
    GateConfig,
    GateResult,
    assign_episode_ids,
    bayesian_terminal_gate_result,
    confidence_initiation_gate_result,
    legacy_trust_gate_result,
    m_of_n_confirmation_result,
    split_duplicate_observation_tracklets,
    tracker_baseline_gate_result,
)
from defense4uavswarm.q1_visdrone import Q1Params, load_gt_protocol, method_acceptance
from scripts.q1_v54.run_corrected_operating_curves import load_candidates, matching_config, gate_config


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/q1_v54/bytetrack.yaml")
    p.add_argument("--output-dir", default="outputs/results/q1_v542/loso")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    out = Path(args.output_dir)
    if out.exists() and any(out.iterdir()) and not args.overwrite and not args.dry_run:
        raise SystemExit(f"Output exists; use --overwrite: {out}")
    out.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")) or {}
    if args.dry_run:
        print(f"dry_run=ok config={args.config} output={out}")
        return

    det = load_candidates(cfg)
    gt, ignored = load_gt_protocol(cfg["dataset_root"], sorted(det["sequence_id"].unique()))
    manifest = add_gt_presence(build_visdrone_frame_manifest(cfg["dataset_root"], sorted(det["sequence_id"].unique())), gt)
    det = merge_manifest_dimensions(det, manifest)
    det = split_duplicate_observation_tracklets(det)
    det = assign_episode_ids(det, gate_config(cfg).max_track_gap)
    mc = matching_config(cfg)
    gc = gate_config(cfg)
    sequences = sorted(det["sequence_id"].unique())
    configs = candidate_gates(cfg)
    all_seq_metrics = []
    for seq in sequences:
        all_seq_metrics.extend(evaluate_sequence(seq, det, gt, ignored, manifest, mc, gc, configs))
    seq_metrics = pd.DataFrame(all_seq_metrics)
    folds = []
    selected_rows = []
    outer_rows = []
    for fold_idx, test_seq in enumerate(sequences):
        train_seqs = [s for s in sequences if s != test_seq]
        folds.append({"outer_fold": fold_idx, "outer_test_sequence": test_seq, "outer_train_sequences": train_seqs})
        base_train = seq_metrics[seq_metrics["sequence_id"].isin(train_seqs) & seq_metrics["method"].eq("tracker_baseline")]
        base_f1 = float(base_train["F1"].mean())
        for method in sorted(seq_metrics["method"].unique()):
            method_train = seq_metrics[seq_metrics["sequence_id"].isin(train_seqs) & seq_metrics["method"].eq(method)].copy()
            if method == "tracker_baseline":
                selected = method_train.iloc[0].copy()
                status = "fixed_baseline"
            else:
                grouped = (
                    method_train.groupby(["method", "parameter_json"], as_index=False)
                    .agg(
                        inner_mean_F1=("F1", "mean"),
                        inner_mean_occupancy=("observed_false_track_occupancy_rows", "mean"),
                        inner_mean_true_track_confirmation_rate=("true_track_confirmation_rate", "mean"),
                        inner_mean_delay=("median_confirmation_delay_from_gt", "mean"),
                    )
                    .sort_values(["inner_mean_occupancy", "inner_mean_delay", "parameter_json"], ascending=[True, True, True])
                )
                feasible = grouped[grouped["inner_mean_F1"] >= base_f1 - 0.01].copy()
                if feasible.empty:
                    selected = grouped.iloc[0].copy()
                    status = "infeasible_noninferiority"
                else:
                    selected = feasible.iloc[0].copy()
                    status = "selected_feasible"
            parameter_json = str(selected["parameter_json"])
            selected_rows.append(
                {
                    "outer_fold": fold_idx,
                    "outer_test_sequence": test_seq,
                    "method": method,
                    "selection_status": status,
                    "selected_parameter_json": parameter_json,
                    "inner_mean_F1": float(selected.get("inner_mean_F1", base_f1)),
                    "inner_mean_occupancy": float(selected.get("inner_mean_occupancy", 0.0)),
                }
            )
            test_hit = seq_metrics[
                seq_metrics["sequence_id"].eq(test_seq)
                & seq_metrics["method"].eq(method)
                & seq_metrics["parameter_json"].astype(str).eq(parameter_json)
            ]
            if test_hit.empty:
                continue
            row = test_hit.iloc[0].to_dict()
            row.update({"outer_fold": fold_idx, "outer_test_sequence": test_seq, "selection_status": status, "selected_parameter_json": parameter_json})
            outer_rows.append(row)
    outer = pd.DataFrame(outer_rows)
    selected_df = pd.DataFrame(selected_rows)
    summary = (
        outer.groupby(["tracker", "method", "selection_status"], as_index=False)
        .agg(
            F1=("F1", "mean"),
            false_new_tracks_per_100_frames=("false_new_tracks_per_100_frames", "mean"),
            observed_false_track_occupancy_rows=("observed_false_track_occupancy_rows", "mean"),
            true_track_confirmation_rate=("true_track_confirmation_rate", "mean"),
            median_gate_delay_from_candidate=("median_gate_delay_from_candidate", "median"),
            median_confirmation_delay_from_gt=("median_confirmation_delay_from_gt", "median"),
        )
    )
    (out / "outer_predictions").mkdir(exist_ok=True)
    Path(out / "outer_folds.json").write_text(json.dumps(folds, indent=2), encoding="utf-8")
    selected_df.to_csv(out / "inner_selection_results.csv", index=False)
    Path(out / "selected_configs_by_fold.json").write_text(json.dumps(selected_rows, indent=2), encoding="utf-8")
    outer.to_csv(out / "outer_test_by_sequence.csv", index=False)
    summary.to_csv(out / "outer_test_summary.csv", index=False)
    (out / "run_metadata.json").write_text(json.dumps({"status": "success", "git_commit": git(["rev-parse", "HEAD"]), "branch": git(["branch", "--show-current"]), "tracker": cfg.get("tracker", "unknown")}, indent=2), encoding="utf-8")
    print(f"status=ok output={out} folds={len(folds)} rows={len(outer)}")


def candidate_gates(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    rows = [{"method": "tracker_baseline", "parameter_json": "{}", "kind": "tracker"}]
    rows.append({"method": "legacy_trust_terminalized", "parameter_json": '{"source":"geometry_dynamic_no_multiagent"}', "kind": "legacy"})
    grids = cfg.get("grids", {})
    for th in grids.get("confidence_threshold", [0.2]):
        rows.append({"method": "confidence_initiation_gate", "parameter_json": json.dumps({"threshold": float(th)}, sort_keys=True, separators=(",", ":")), "kind": "confidence", "threshold": float(th)})
    for spec in grids.get("m_of_n", []):
        rows.append({"method": "m_of_n_confirmation", "parameter_json": json.dumps({"M": int(spec["M"]), "N": int(spec["N"]), "confidence_threshold": float(spec["confidence_threshold"])}, sort_keys=True, separators=(",", ":")), "kind": "mofn", **spec})
    for th in grids.get("bayesian_threshold", [1.1]):
        rows.append({"method": "bayesian_fixed_terminal", "parameter_json": json.dumps({"high_update": 0.9, "mid_update": 0.35, "negative_update": -0.25, "threshold": float(th)}, sort_keys=True, separators=(",", ":")), "kind": "bayesian", "threshold": float(th)})
    rows.append({"method": "rf_terminal_gate", "parameter_json": '{"status":"not_implemented_in_v542a"}', "kind": "unavailable"})
    return rows


def evaluate_sequence(seq: str, det: pd.DataFrame, gt: pd.DataFrame, ignored: pd.DataFrame, manifest: pd.DataFrame, mc: MatchingConfig, gc: GateConfig, configs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    d = det[det["sequence_id"].eq(seq)].copy()
    g = gt[gt["sequence_id"].eq(seq)].copy()
    ig = ignored[ignored["sequence_id"].eq(seq)].copy() if not ignored.empty else ignored
    fm = manifest[manifest["sequence_id"].eq(seq)].copy()
    rows = []
    for spec in configs:
        kind = spec["kind"]
        if kind == "tracker":
            gate = tracker_baseline_gate_result(d, gc)
        elif kind == "legacy":
            gate = legacy_trust_gate_result(d, method_acceptance(d, "geometry_dynamic_no_multiagent", Q1Params()), gc)
        elif kind == "confidence":
            gate = confidence_initiation_gate_result(d, float(spec["threshold"]), gc)
        elif kind == "mofn":
            gate = m_of_n_confirmation_result(d, int(spec["M"]), int(spec["N"]), float(spec["confidence_threshold"]), gc)
        elif kind == "bayesian":
            gate = bayesian_terminal_gate_result(d, threshold=float(spec["threshold"]), cfg=gc)
        else:
            rows.append(
                {
                    "sequence_id": seq,
                    "tracker": cfg_tracker(d),
                    "method": spec["method"],
                    "parameter_json": spec["parameter_json"],
                    "selection_status": "unavailable",
                    "label_access": label_access(spec["method"]),
                    "F1": pd.NA,
                }
            )
            continue
        result = evaluate_gate_result(d, gate, g, ig, fm, mc)
        row = dict(result.summary)
        row.update({"sequence_id": seq, "tracker": cfg_tracker(d), "method": spec["method"], "parameter_json": gate.parameter_json, "label_access": label_access(spec["method"])})
        rows.append(row)
    return rows


def cfg_tracker(d: pd.DataFrame) -> str:
    return d["detector"].iloc[0] if "detector" in d and not d.empty else "unknown"


def label_access(method: str) -> str:
    if method == "rf_terminal_gate":
        return "supervised_source_label"
    if method in {"confidence_initiation_gate", "m_of_n_confirmation", "bayesian_fixed_terminal"}:
        return "zero_target_label"
    return "fixed_zero_label"


def git(args: list[str]) -> str:
    try:
        return subprocess.check_output(["git", *args], text=True).strip()
    except Exception:
        return "unavailable"


if __name__ == "__main__":
    main()
