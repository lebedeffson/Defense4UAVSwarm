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
from defense4uavswarm.q1_v5.selective_quarantine import SelectiveTrustQuarantineConfig, selective_quarantine_gate_result
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
    tracker_name = str(cfg.get("tracker", "unknown"))
    for seq in sequences:
        all_seq_metrics.extend(evaluate_sequence(seq, det, gt, ignored, manifest, mc, gc, configs, tracker_name))
    seq_metrics = pd.DataFrame(all_seq_metrics)
    folds = []
    selected_rows = []
    outer_rows = []
    selection_cfg = cfg.get("selection", {})
    max_f1_loss = float(selection_cfg.get("robust_inner_max_f1_loss", 0.01))
    mean_f1_loss = float(selection_cfg.get("robust_inner_mean_f1_loss", 0.005))
    max_confirm_loss = float(selection_cfg.get("robust_inner_max_confirm_loss", 0.01))
    for fold_idx, test_seq in enumerate(sequences):
        train_seqs = [s for s in sequences if s != test_seq]
        folds.append({"outer_fold": fold_idx, "tracker": tracker_name, "outer_test_sequence": test_seq, "outer_train_sequences": train_seqs})
        base_train = seq_metrics[seq_metrics["sequence_id"].isin(train_seqs) & seq_metrics["method"].eq("tracker_baseline")]
        base_f1 = float(base_train["F1"].mean())
        for method in sorted(seq_metrics["method"].unique()):
            method_train = seq_metrics[seq_metrics["sequence_id"].isin(train_seqs) & seq_metrics["method"].eq(method)].copy()
            if method == "tracker_baseline":
                selected = method_train.iloc[0].copy()
                status = "fixed_baseline"
            else:
                grouped = _group_inner_candidates(method_train, base_train)
                if grouped.empty:
                    selected_rows.append(
                        {
                            "outer_fold": fold_idx,
                            "outer_test_sequence": test_seq,
                            "method": method,
                            "selection_status": "unavailable",
                            "selected_parameter_json": "",
                            "inner_mean_F1": float("nan"),
                            "inner_mean_occupancy": float("nan"),
                            "inner_mean_occupancy_per_100_frames": float("nan"),
                            "inner_min_F1_delta": float("nan"),
                            "inner_mean_F1_delta": float("nan"),
                            "inner_min_confirm_delta": float("nan"),
                        }
                    )
                    continue
                feasible = grouped[
                    (grouped["inner_min_F1_delta"] >= -max_f1_loss)
                    & (grouped["inner_mean_F1_delta"] >= -mean_f1_loss)
                    & (grouped["inner_min_confirm_delta"] >= -max_confirm_loss)
                ].copy()
                if feasible.empty:
                    passthrough = grouped[grouped["parameter_json"].astype(str).str.contains('"quarantine_fraction_max":0.0', regex=False)].copy()
                    if not passthrough.empty:
                        selected = passthrough.iloc[0].copy()
                        status = "baseline_passthrough_no_safe_active_profile"
                    else:
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
                    "inner_mean_occupancy_per_100_frames": float(selected.get("inner_mean_occupancy_per_100_frames", 0.0)),
                    "inner_min_F1_delta": float(selected.get("inner_min_F1_delta", 0.0)),
                    "inner_mean_F1_delta": float(selected.get("inner_mean_F1_delta", 0.0)),
                    "inner_min_confirm_delta": float(selected.get("inner_min_confirm_delta", 0.0)),
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
    summary_group_cols = ["tracker", "method", "label_access", "selection_status"]
    summary = (
        outer.groupby(summary_group_cols, as_index=False)
        .agg(
            F1=("F1", "mean"),
            false_new_tracks_per_100_frames=("false_new_tracks_per_100_frames", "mean"),
            observed_false_track_occupancy_rows=("observed_false_track_occupancy_rows", "mean"),
            observed_false_track_occupancy_per_100_frames=("observed_false_track_occupancy_per_100_frames", "mean"),
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
    write_quarantine_budget_audit(outer, out / "quarantine_budget_by_sequence.csv")
    write_terminal_audits(outer, out)
    (out / "run_metadata.json").write_text(json.dumps({"status": "success", "git_commit": git(["rev-parse", "HEAD"]), "branch": git(["branch", "--show-current"]), "tracker": tracker_name}, indent=2), encoding="utf-8")
    print(f"status=ok output={out} folds={len(folds)} rows={len(outer)}")


def write_terminal_audits(outer: pd.DataFrame, out: Path) -> None:
    data = outer[outer["method"].eq("selective_trust_quarantine")].copy() if not outer.empty and "method" in outer else pd.DataFrame()
    terminal_cols = [
        "tracker",
        "outer_fold",
        "sequence_id",
        "episode_starts",
        "quarantined_starts",
        "evidence_confirmations",
        "baseline_releases",
        "hard_veto_rejections",
        "temporal_absence_rejections",
        "right_censored_episode_count",
        "pending_end_count",
        "selected_maximum_quarantine_frames",
    ]
    censor_cols = [
        "tracker",
        "outer_fold",
        "sequence_id",
        "full_sequence_frame_count",
        "primary_frame_count",
        "censoring_guard_frames",
        "right_censored_episode_count",
        "pending_end_count",
    ]
    if data.empty:
        pd.DataFrame(columns=terminal_cols).to_csv(out / "episode_terminal_audit.csv", index=False)
        pd.DataFrame(columns=censor_cols).to_csv(out / "censoring_audit.csv", index=False)
        return
    for col in terminal_cols + censor_cols:
        if col not in data:
            data[col] = 0
    data[terminal_cols].to_csv(out / "episode_terminal_audit.csv", index=False)
    data[censor_cols].to_csv(out / "censoring_audit.csv", index=False)


def write_quarantine_budget_audit(outer: pd.DataFrame, path: Path) -> None:
    cols = [
        "tracker",
        "outer_fold",
        "sequence_id",
        "episode_starts",
        "quarantined_starts",
        "configured_fraction",
        "realized_fraction",
        "timeout_releases",
        "hard_veto_rejections",
        "temporal_absence_rejections",
        "right_censored_episode_count",
        "pending_end_count",
        "budget_invariant_pass",
    ]
    if outer.empty or "method" not in outer:
        pd.DataFrame(columns=cols).to_csv(path, index=False)
        return
    rows = []
    data = outer[outer["method"].eq("selective_trust_quarantine")].copy()
    for row in data.itertuples(index=False):
        starts = int(getattr(row, "episode_starts", 0) or 0)
        quarantined = int(getattr(row, "quarantined_starts", 0) or 0)
        configured = float(getattr(row, "configured_quarantine_fraction", 0.0) or 0.0)
        realized = float(getattr(row, "realized_quarantine_fraction", quarantined / max(1, starts)) or 0.0)
        tolerance = 1.0 / max(1, starts)
        rows.append(
            {
                "tracker": getattr(row, "tracker", ""),
                "outer_fold": int(getattr(row, "outer_fold", -1)),
                "sequence_id": getattr(row, "sequence_id", ""),
                "episode_starts": starts,
                "quarantined_starts": quarantined,
                "configured_fraction": configured,
                "realized_fraction": realized,
                "timeout_releases": int(getattr(row, "timeout_releases", 0) or 0),
                "hard_veto_rejections": int(getattr(row, "hard_veto_rejections", 0) or 0),
                "temporal_absence_rejections": int(getattr(row, "temporal_absence_rejections", 0) or 0),
                "right_censored_episode_count": int(getattr(row, "right_censored_episode_count", 0) or 0),
                "pending_end_count": int(getattr(row, "pending_end_count", 0) or 0),
                "budget_invariant_pass": bool(realized <= configured + tolerance),
            }
        )
    pd.DataFrame(rows, columns=cols).to_csv(path, index=False)


def _group_inner_candidates(method_train: pd.DataFrame, base_train: pd.DataFrame) -> pd.DataFrame:
    if method_train.empty:
        return pd.DataFrame()
    method_train = method_train.dropna(subset=["F1", "observed_false_track_occupancy_per_100_frames"]).copy()
    if method_train.empty:
        return pd.DataFrame(
            columns=[
                "method",
                "parameter_json",
                "inner_mean_F1",
                "inner_mean_occupancy",
                "inner_mean_occupancy_per_100_frames",
                "inner_mean_true_track_confirmation_rate",
                "inner_mean_delay",
                "inner_min_F1_delta",
                "inner_mean_F1_delta",
                "inner_min_confirm_delta",
            ]
        )
    base = base_train[
        [
            "sequence_id",
            "F1",
            "true_track_confirmation_rate",
            "num_eval_frames",
        ]
    ].rename(
        columns={
            "F1": "baseline_F1",
            "true_track_confirmation_rate": "baseline_true_track_confirmation_rate",
            "num_eval_frames": "baseline_num_eval_frames",
        }
    )
    joined = method_train.merge(base, on="sequence_id", how="left")
    joined["F1_delta"] = joined["F1"].astype(float) - joined["baseline_F1"].astype(float)
    joined["confirm_delta"] = joined["true_track_confirmation_rate"].astype(float) - joined["baseline_true_track_confirmation_rate"].astype(float)
    frames = joined["num_eval_frames"].astype(float).where(joined["num_eval_frames"].astype(float).gt(0), joined["baseline_num_eval_frames"].astype(float))
    joined["occupancy_per_100_frames"] = joined["observed_false_track_occupancy_per_100_frames"].astype(float)
    return (
        joined.groupby(["method", "parameter_json"], as_index=False)
        .agg(
            inner_mean_F1=("F1", "mean"),
            inner_mean_occupancy=("observed_false_track_occupancy_rows", "mean"),
            inner_mean_occupancy_per_100_frames=("occupancy_per_100_frames", "mean"),
            inner_mean_true_track_confirmation_rate=("true_track_confirmation_rate", "mean"),
            inner_mean_delay=("median_confirmation_delay_from_gt", "mean"),
            inner_min_F1_delta=("F1_delta", "min"),
            inner_mean_F1_delta=("F1_delta", "mean"),
            inner_min_confirm_delta=("confirm_delta", "min"),
        )
        .sort_values(["inner_mean_occupancy_per_100_frames", "inner_mean_delay", "parameter_json"], ascending=[True, True, True])
    )


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
    for spec in grids.get("selective_quarantine", []):
        qcfg = SelectiveTrustQuarantineConfig.from_mapping(spec)
        rows.append({"method": "selective_trust_quarantine", "parameter_json": json.dumps(qcfg.to_parameters(), sort_keys=True, separators=(",", ":")), "kind": "selective_quarantine", "config": qcfg.to_parameters()})
    rows.append({"method": "rf_terminal_gate", "parameter_json": '{"status":"not_implemented_in_v542a"}', "kind": "unavailable"})
    return rows


def evaluate_sequence(seq: str, det: pd.DataFrame, gt: pd.DataFrame, ignored: pd.DataFrame, manifest: pd.DataFrame, mc: MatchingConfig, gc: GateConfig, configs: list[dict[str, Any]], tracker_name: str) -> list[dict[str, Any]]:
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
        elif kind == "selective_quarantine":
            gate = selective_quarantine_gate_result(d, fm, SelectiveTrustQuarantineConfig.from_mapping(spec.get("config")))
        else:
            rows.append(
                {
                    "sequence_id": seq,
                    "tracker": tracker_name,
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
        row.update({"sequence_id": seq, "tracker": tracker_name, "method": spec["method"], "parameter_json": gate.parameter_json, "label_access": label_access(spec["method"])})
        if kind == "selective_quarantine":
            row["active_profile"] = "selective_quarantine_v21"
            row["selected_maximum_quarantine_frames"] = SelectiveTrustQuarantineConfig.from_mapping(spec.get("config")).maximum_quarantine_frames
        rows.append(row)
    return rows


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
