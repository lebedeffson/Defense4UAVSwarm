#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import pandas as pd

from defense4uavswarm.q1_v5.evaluation_matching import evaluate_gate_result
from defense4uavswarm.q1_v5.initiation_gate import tracker_baseline_gate_result
from defense4uavswarm.q1_v5.risk_ranker import train_stage_ranker, train_thresholds
from defense4uavswarm.q1_v5.two_stage_quarantine import RiskPrioritizedTwoStageConfig, risk_prioritized_two_stage_gate_result
from scripts.q1_v55.common import gate_config, load_frozen_inputs, matching_config, prepare_output, read_config, results_root, write_metadata


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/q1_v55/bytetrack_risk_prioritized.yaml")
    p.add_argument("--output-dir")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    cfg = read_config(args.config)
    root = results_root(cfg, args.output_dir)
    out = root / "loso"
    prepare_output(out, overwrite=args.overwrite, dry_run=args.dry_run)
    if args.dry_run:
        print(f"dry_run=ok output={out}")
        return
    ranker_gate = root / "ranker" / "go_no_go.json"
    if ranker_gate.exists():
        gate_b = json.loads(ranker_gate.read_text(encoding="utf-8")).get("gate_b", {})
        if gate_b.get("decision") == "STOP_RANKER_NO_GAIN":
            stop = {
                "protocol_id": cfg.get("protocol_id"),
                "method": "risk_prioritized_two_stage_quarantine",
                "status": "STOP_RANKER_NO_GAIN",
                "reason": "Gate B failed; online integration is blocked by the v2.2-RP protocol.",
            }
            pd.DataFrame([stop]).to_csv(out / "outer_test_by_sequence.csv", index=False)
            pd.DataFrame([stop]).to_csv(out / "outer_test_summary.csv", index=False)
            pd.DataFrame([stop]).to_csv(out / "inner_selection_results.csv", index=False)
            pd.DataFrame([stop]).to_csv(out / "budget_audit.csv", index=False)
            pd.DataFrame([stop]).to_csv(out / "terminal_state_audit.csv", index=False)
            pd.DataFrame([stop]).to_csv(out / "censoring_audit.csv", index=False)
            pd.DataFrame([stop]).to_csv(out / "admission_log.csv", index=False)
            Path(out / "outer_folds.json").write_text("[]\n", encoding="utf-8")
            Path(out / "selected_configs_by_fold.json").write_text("[]\n", encoding="utf-8")
            (out / "outer_predictions").mkdir(exist_ok=True)
            write_metadata(out / "run_metadata.json", stop)
            print(f"status=STOP_RANKER_NO_GAIN output={out}")
            return
    det, gt, ignored, manifest = load_frozen_inputs(cfg)
    episodes = pd.read_parquet(root / "episode_harm" / "episode_harm_table.parquet")
    sequences = sorted(episodes["sequence_id"].astype(str).unique())
    mc = matching_config(cfg)
    gc = gate_config(cfg)
    configs = frozen_configs(cfg)
    all_metrics = []
    ledgers = []
    admissions = []
    folds = []
    for eval_seq in sequences:
        train_eps = episodes[~episodes["sequence_id"].astype(str).eq(eval_seq)].copy()
        eval_eps = episodes[episodes["sequence_id"].astype(str).eq(eval_seq)].copy()
        scores, thresholds = fit_predict_scores(train_eps, eval_eps, cfg)
        seq_metrics, seq_ledgers, seq_admissions = evaluate_sequence(eval_seq, det, gt, ignored, manifest, mc, gc, configs, scores, thresholds, str(cfg.get("tracker", "bytetrack")))
        all_metrics.extend(seq_metrics)
        ledgers.extend(seq_ledgers)
        admissions.extend(seq_admissions)
    seq_metrics_df = pd.DataFrame(all_metrics)
    selected_rows = []
    outer_rows = []
    selection = cfg.get("selection", {})
    max_f1_loss = float(selection.get("robust_inner_max_f1_loss", 0.01))
    mean_f1_loss = float(selection.get("robust_inner_mean_f1_loss", 0.005))
    max_confirm_loss = float(selection.get("robust_inner_max_confirm_loss", 0.01))
    for fold_idx, test_seq in enumerate(sequences):
        train_seqs = [s for s in sequences if s != test_seq]
        folds.append({"outer_fold": fold_idx, "tracker": cfg.get("tracker", "bytetrack"), "outer_test_sequence": test_seq, "outer_train_sequences": train_seqs})
        base_train = seq_metrics_df[seq_metrics_df["sequence_id"].isin(train_seqs) & seq_metrics_df["method"].eq("tracker_baseline")]
        method_train = seq_metrics_df[seq_metrics_df["sequence_id"].isin(train_seqs) & seq_metrics_df["method"].eq("risk_prioritized_two_stage_quarantine")]
        grouped = group_configs(method_train, base_train)
        feasible = grouped[
            (grouped["inner_min_F1_delta"] >= -max_f1_loss)
            & (grouped["inner_mean_F1_delta"] >= -mean_f1_loss)
            & (grouped["inner_min_confirm_delta"] >= -max_confirm_loss)
            & (grouped["inner_budget_invariants_pass"].astype(bool))
        ].copy()
        if feasible.empty:
            selected = grouped[grouped["parameter_json"].str.contains('"stage1_fraction_max":0.0', regex=False)].iloc[0]
            status = "baseline_passthrough_no_safe_active_profile"
        else:
            selected = feasible.iloc[0]
            status = "selected_feasible"
        selected_rows.append({"outer_fold": fold_idx, "outer_test_sequence": test_seq, "selection_status": status, **selected.to_dict()})
        base_test = seq_metrics_df[seq_metrics_df["sequence_id"].eq(test_seq) & seq_metrics_df["method"].eq("tracker_baseline")]
        outer_rows.extend(base_test.assign(outer_fold=fold_idx, outer_test_sequence=test_seq, selection_status="fixed_baseline", selected_parameter_json="{}").to_dict(orient="records"))
        hit = seq_metrics_df[
            seq_metrics_df["sequence_id"].eq(test_seq)
            & seq_metrics_df["method"].eq("risk_prioritized_two_stage_quarantine")
            & seq_metrics_df["parameter_json"].eq(selected["parameter_json"])
        ]
        if not hit.empty:
            outer_rows.extend(hit.assign(outer_fold=fold_idx, outer_test_sequence=test_seq, selection_status=status, selected_parameter_json=selected["parameter_json"]).to_dict(orient="records"))
    outer = pd.DataFrame(outer_rows)
    selected_df = pd.DataFrame(selected_rows)
    summary = outer.groupby(["tracker", "method", "label_access", "selection_status"], as_index=False).mean(numeric_only=True)
    (out / "outer_predictions").mkdir(exist_ok=True)
    Path(out / "outer_folds.json").write_text(json.dumps(folds, indent=2), encoding="utf-8")
    selected_df.to_csv(out / "inner_selection_results.csv", index=False)
    Path(out / "selected_configs_by_fold.json").write_text(json.dumps(selected_rows, indent=2, default=str), encoding="utf-8")
    outer.to_csv(out / "outer_test_by_sequence.csv", index=False)
    summary.to_csv(out / "outer_test_summary.csv", index=False)
    pd.DataFrame(ledgers).to_csv(out / "budget_audit.csv", index=False)
    pd.DataFrame(admissions).to_csv(out / "admission_log.csv", index=False)
    terminal = outer[outer["method"].eq("risk_prioritized_two_stage_quarantine")].copy()
    terminal.to_csv(out / "terminal_state_audit.csv", index=False)
    terminal.to_csv(out / "censoring_audit.csv", index=False)
    write_metadata(out / "run_metadata.json", {"outer_folds": len(folds), "method": "risk_prioritized_two_stage_quarantine"})
    print(f"status=ok output={out} outer_rows={len(outer)}")


def frozen_configs(cfg: dict[str, Any]) -> list[RiskPrioritizedTwoStageConfig]:
    v = cfg.get("v22", {})
    configs = [RiskPrioritizedTwoStageConfig(stage1_fraction_max=0.0, stage2_fraction_max=0.0, stage2_horizon_frames=3, tracker_profile=str(cfg.get("tracker", "bytetrack")))]
    for q1 in v.get("stage1_budgets", [0.05, 0.10, 0.15]):
        for q2 in v.get("stage2_budgets", [0.01, 0.03, 0.05]):
            if float(q2) > float(q1):
                continue
            for h in v.get("stage2_horizons", [3, 5, 10]):
                configs.append(RiskPrioritizedTwoStageConfig(stage1_fraction_max=float(q1), stage2_fraction_max=float(q2), stage2_horizon_frames=int(h), tracker_profile=str(cfg.get("tracker", "bytetrack"))))
    return configs


def fit_predict_scores(train_eps: pd.DataFrame, eval_eps: pd.DataFrame, cfg: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, float]]:
    r1 = train_stage_ranker(train_eps, stage=1, seed=int(cfg.get("v22", {}).get("seed", 2026)))
    r2 = train_stage_ranker(train_eps, stage=2, seed=int(cfg.get("v22", {}).get("seed", 2026)))
    p1_train = r1.predict(train_eps, horizon=1).rename(columns={"risk_score": "stage1_risk_score"})
    p2_train = r2.predict(train_eps[train_eps["second_frame_id"].notna()].copy(), horizon=10).rename(columns={"risk_score": "stage2_risk_score"})
    thresholds = train_thresholds(train_eps, p1_train.rename(columns={"stage1_risk_score": "risk_score"}), p2_train.rename(columns={"stage2_risk_score": "risk_score"}), alpha1=float(cfg.get("v22", {}).get("alpha_stage1_true", 0.10)), alpha2=float(cfg.get("v22", {}).get("alpha_stage2_true", 0.05)))
    p1 = r1.predict(eval_eps, horizon=1).rename(columns={"risk_score": "stage1_risk_score", "p_false": "stage1_p_false", "predicted_bounded_false_rows": "stage1_predicted_rows"})
    p2 = r2.predict(eval_eps[eval_eps["second_frame_id"].notna()].copy(), horizon=10).rename(columns={"risk_score": "stage2_risk_score", "p_false": "stage2_p_false", "predicted_bounded_false_rows": "stage2_predicted_rows"})
    scores = p1.merge(p2, on=["sequence_id", "tracklet_id", "episode_id"], how="left")
    scores["stage2_risk_score"] = scores["stage2_risk_score"].fillna(0.0)
    return scores, thresholds


def evaluate_sequence(seq: str, det: pd.DataFrame, gt: pd.DataFrame, ignored: pd.DataFrame, manifest: pd.DataFrame, mc: Any, gc: Any, configs: list[RiskPrioritizedTwoStageConfig], scores: pd.DataFrame, thresholds: dict[str, float], tracker: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    d = det[det["sequence_id"].astype(str).eq(seq)].copy()
    g = gt[gt["sequence_id"].astype(str).eq(seq)].copy()
    ig = ignored[ignored["sequence_id"].astype(str).eq(seq)].copy() if not ignored.empty else ignored
    fm = manifest[manifest["sequence_id"].astype(str).eq(seq)].copy()
    rows = []
    ledgers = []
    admissions = []
    base = evaluate_gate_result(d, tracker_baseline_gate_result(d, gc), g, ig, fm, mc)
    base_row = dict(base.summary)
    base_row.update({"sequence_id": seq, "tracker": tracker, "method": "tracker_baseline", "parameter_json": "{}", "label_access": "fixed_zero_label"})
    rows.append(base_row)
    for qcfg in configs:
        gate, ledger, admission = risk_prioritized_two_stage_gate_result(d, fm, scores[scores["sequence_id"].astype(str).eq(seq)].copy(), tau_stage1=thresholds["tau_stage1"], tau_stage2=thresholds["tau_stage2"], cfg=qcfg)
        result = evaluate_gate_result(d, gate, g, ig, fm, mc)
        row = dict(result.summary)
        row.update({"sequence_id": seq, "tracker": tracker, "method": gate.method_id, "parameter_json": gate.parameter_json, "label_access": "source_sequence_labels_only"})
        rows.append(row)
        if not ledger.empty:
            ledgers.extend(ledger.assign(sequence_id=seq, parameter_json=gate.parameter_json).to_dict(orient="records"))
        if not admission.empty:
            admissions.extend(admission.assign(sequence_id=seq, parameter_json=gate.parameter_json).to_dict(orient="records"))
    return rows, ledgers, admissions


def group_configs(method_train: pd.DataFrame, base_train: pd.DataFrame) -> pd.DataFrame:
    base = base_train[["sequence_id", "F1", "true_track_confirmation_rate"]].rename(columns={"F1": "baseline_F1", "true_track_confirmation_rate": "baseline_true_track_confirmation_rate"})
    joined = method_train.merge(base, on="sequence_id", how="left")
    joined["F1_delta"] = joined["F1"].astype(float) - joined["baseline_F1"].astype(float)
    joined["confirm_delta"] = joined["true_track_confirmation_rate"].astype(float) - joined["baseline_true_track_confirmation_rate"].astype(float)
    grouped = (
        joined.groupby(["method", "parameter_json"], as_index=False)
        .agg(
            inner_mean_F1=("F1", "mean"),
            inner_mean_occupancy_per_100_frames=("observed_false_track_occupancy_per_100_frames", "mean"),
            inner_min_F1_delta=("F1_delta", "min"),
            inner_mean_F1_delta=("F1_delta", "mean"),
            inner_min_confirm_delta=("confirm_delta", "min"),
            inner_budget_invariants_pass=("budget_invariant_pass", "min"),
        )
        .sort_values(["inner_mean_occupancy_per_100_frames", "inner_min_F1_delta", "parameter_json"], ascending=[True, False, True])
    )
    return grouped


if __name__ == "__main__":
    main()
