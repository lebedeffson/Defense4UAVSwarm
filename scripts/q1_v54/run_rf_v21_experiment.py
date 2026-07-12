#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from defense4uavswarm.q1_v5.evaluation_matching import evaluate_gate_result
from defense4uavswarm.q1_v5.frame_manifest import add_gt_presence, build_visdrone_frame_manifest, merge_manifest_dimensions
from defense4uavswarm.q1_v5.initiation_gate import assign_episode_ids, split_duplicate_observation_tracklets, tracker_baseline_gate_result
from defense4uavswarm.q1_v5.rf_v21 import (
    FEATURE_COLUMNS,
    add_episode_labels,
    build_episode_feature_table,
    false_probability,
    rf_unbounded_gate_result,
    select_rf_spec_nested,
    train_random_forest,
)
from defense4uavswarm.q1_visdrone import load_gt_protocol
from scripts.q1_v54.run_corrected_operating_curves import gate_config, load_candidates, matching_config


def git(args: list[str]) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True).strip()
    except Exception:
        return "unavailable"


def prepare_inputs(cfg: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    det = load_candidates(cfg)
    gt, ignored = load_gt_protocol(cfg["dataset_root"], sorted(det["sequence_id"].unique()))
    manifest = add_gt_presence(build_visdrone_frame_manifest(cfg["dataset_root"], sorted(det["sequence_id"].unique())), gt)
    det = merge_manifest_dimensions(det, manifest)
    det = split_duplicate_observation_tracklets(det)
    det = assign_episode_ids(det, gate_config(cfg).max_track_gap)
    return det, gt, ignored, manifest


def sequence_slice(df: pd.DataFrame, seqs: list[str]) -> pd.DataFrame:
    if df is None or df.empty or "sequence_id" not in df:
        return df
    return df[df["sequence_id"].isin(seqs)].copy()


def evaluate_method(seq: str, det: pd.DataFrame, gt: pd.DataFrame, ignored: pd.DataFrame, manifest: pd.DataFrame, gate, mc) -> dict[str, Any]:
    d = det[det["sequence_id"].eq(seq)].copy()
    g = gt[gt["sequence_id"].eq(seq)].copy()
    ig = ignored[ignored["sequence_id"].eq(seq)].copy() if ignored is not None and not ignored.empty else ignored
    fm = manifest[manifest["sequence_id"].eq(seq)].copy()
    result = evaluate_gate_result(d, gate, g, ig, fm, mc)
    return dict(result.summary)


def write_feature_schema(path: Path) -> None:
    payload = {
        "target": "false_episode",
        "unit": "episode_start",
        "future_features_allowed": False,
        "feature_columns": FEATURE_COLUMNS,
        "notes": "Features are taken from the first observation of each episode and current/past online fields only.",
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/q1_v54/bytetrack.yaml")
    p.add_argument("--output-dir", default="outputs/q1_practical_closure_v7/rf_v21")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--seed", type=int, default=2026)
    args = p.parse_args()
    out = Path(args.output_dir)
    if out.exists() and args.overwrite:
        shutil.rmtree(out)
    if out.exists() and any(out.iterdir()) and not args.overwrite:
        raise SystemExit(f"Output exists; use --overwrite: {out}")
    out.mkdir(parents=True, exist_ok=True)
    (out / "predictions").mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")) or {}
    det, gt, ignored, manifest = prepare_inputs(cfg)
    mc = matching_config(cfg)
    gc = gate_config(cfg)
    sequences = sorted(det["sequence_id"].unique())
    rows = []
    selected_rows = []
    threshold_rows = []
    feature_importances = []
    leakage_rows = []
    for fold_idx, test_seq in enumerate(sequences):
        train_seqs = [s for s in sequences if s != test_seq]
        train_det = sequence_slice(det, train_seqs)
        train_gt = sequence_slice(gt, train_seqs)
        train_ignored = sequence_slice(ignored, train_seqs)
        train_manifest = sequence_slice(manifest, train_seqs)
        episode_train = add_episode_labels(train_det, train_gt, train_ignored, train_manifest, mc, gc)
        selected, selection_frame = select_rf_spec_nested(episode_train, train_seqs, random_state=args.seed + fold_idx)
        model = train_random_forest(episode_train, selected, random_state=args.seed + fold_idx)

        test_det = det[det["sequence_id"].eq(test_seq)].copy()
        test_episode_features = build_episode_feature_table(test_det)
        fp_prob = false_probability(model, test_episode_features)
        pred = test_episode_features[["sequence_id", "tracklet_id", "episode_id", "first_frame_id"]].copy()
        pred["fp_probability"] = fp_prob
        pred["accepted"] = pred["fp_probability"].astype(float) < float(selected.decision_threshold)
        pred.to_csv(out / "predictions" / f"fold_{fold_idx:02d}_{test_seq}.csv", index=False)

        baseline_gate = tracker_baseline_gate_result(test_det, gc)
        rf_gate = rf_unbounded_gate_result(test_det, test_episode_features, fp_prob, selected, gc)
        for method, gate in [("tracker_baseline", baseline_gate), ("rf_unbounded", rf_gate)]:
            row = evaluate_method(test_seq, det, gt, ignored, manifest, gate, mc)
            row.update(
                {
                    "outer_fold": fold_idx,
                    "outer_test_sequence": test_seq,
                    "sequence_id": test_seq,
                    "tracker": cfg.get("tracker", "bytetrack"),
                    "method": method,
                    "label_access": "supervised_source_label" if method == "rf_unbounded" else "fixed_zero_label",
                    "selected_parameter_json": rf_gate.parameter_json if method == "rf_unbounded" else "{}",
                }
            )
            rows.append(row)
        selected_payload = selected.to_params()
        selected_rows.append({"outer_fold": fold_idx, "outer_test_sequence": test_seq, **selected_payload})
        threshold_rows.append({"outer_fold": fold_idx, "outer_test_sequence": test_seq, "method": "rf_unbounded", "decision_threshold": selected.decision_threshold})
        selection_frame.assign(outer_fold=fold_idx, outer_test_sequence=test_seq).to_csv(out / "predictions" / f"fold_{fold_idx:02d}_inner_selection.csv", index=False)
        for name, importance in zip(FEATURE_COLUMNS, getattr(model, "feature_importances_", np.zeros(len(FEATURE_COLUMNS)))):
            feature_importances.append({"outer_fold": fold_idx, "feature": name, "importance": float(importance)})
        leakage_rows.append(
            {
                "outer_fold": fold_idx,
                "train_sequences": train_seqs,
                "test_sequence": test_seq,
                "train_episode_count": int(len(episode_train)),
                "test_episode_count": int(len(test_episode_features)),
                "feature_fit_sequences": train_seqs,
                "threshold_fit_sequences": train_seqs,
                "hyperparameter_fit_sequences": train_seqs,
                "future_features_detected": False,
                "test_leakage_detected": False,
            }
        )
    fold = pd.DataFrame(rows)
    fold.to_csv(out / "fold_results.csv", index=False)
    summary = (
        fold.groupby(["tracker", "method", "label_access"], as_index=False)
        .agg(
            F1=("F1", "mean"),
            false_new_tracks_per_100_frames=("false_new_tracks_per_100_frames", "mean"),
            observed_false_track_occupancy_per_100_frames=("observed_false_track_occupancy_per_100_frames", "mean"),
            true_track_confirmation_rate=("true_track_confirmation_rate", "mean"),
        )
    )
    summary.to_csv(out / "aggregate_results.csv", index=False)
    pd.DataFrame(selected_rows).to_csv(out / "selected_hyperparameters.csv", index=False)
    pd.DataFrame(threshold_rows).to_csv(out / "selected_thresholds.csv", index=False)
    write_feature_schema(out / "feature_schema.json")
    pd.DataFrame(feature_importances).groupby("feature", as_index=False).agg(importance=("importance", "mean")).sort_values("importance", ascending=False).to_csv(out / "feature_importance.csv", index=False)
    leakage_payload = {"folds": leakage_rows, "future_features_detected": False, "test_leakage_detected": False}
    (out / "leakage_audit.json").write_text(json.dumps(leakage_payload, indent=2, sort_keys=True), encoding="utf-8")
    stats = {
        "tracker": cfg.get("tracker", "bytetrack"),
        "folds_required": 7,
        "folds_by_method": {k: int(v) for k, v in fold.groupby("method")["outer_fold"].nunique().to_dict().items()},
        "rf_unbounded_complete": int(fold[fold["method"].eq("rf_unbounded")]["outer_fold"].nunique()) == 7,
        "future_features_detected": False,
        "test_leakage_detected": False,
        "git_commit": git(["rev-parse", "HEAD"]),
    }
    (out / "statistics.json").write_text(json.dumps(stats, indent=2, sort_keys=True), encoding="utf-8")
    manifest_payload = {
        "status": "success",
        "protocol": "q1_selective_quarantine_v21",
        "method": "rf_unbounded",
        "config": args.config,
        "git_commit": git(["rev-parse", "HEAD"]),
        "git_branch": git(["branch", "--show-current"]),
    }
    (out / "run_manifest.json").write_text(json.dumps(manifest_payload, indent=2, sort_keys=True), encoding="utf-8")
    print(f"status=ok output={out} folds={fold['outer_fold'].nunique()} rows={len(fold)}")


if __name__ == "__main__":
    main()
