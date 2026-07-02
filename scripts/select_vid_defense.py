#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml

from defense4uavswarm.config import load_config
from defense4uavswarm.datasets.visdrone import VisDroneDataset
from defense4uavswarm.filtering.tnorms import apply_tnorm
from defense4uavswarm.metrics.asr import attack_success_breakdown
from defense4uavswarm.metrics.detection import match_frame
from defense4uavswarm.pipeline import evaluate, save


def load_split(path: str | Path, name: str) -> list[str]:
    with open(path, "r", encoding="utf-8") as f:
        payload = yaml.safe_load(f) or {}
    return [str(x) for x in payload[name]]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--split-config", default="configs/vid_split.yaml")
    p.add_argument("--split", default="calibration")
    p.add_argument("--results", default="outputs/results/vid_calibration")
    p.add_argument("--model", default="yolov8n")
    p.add_argument("--eps", type=float, default=0.008)
    p.add_argument("--t-norms", nargs="+", default=["T_min"])
    p.add_argument("--k-variants", nargs="+", default=["center", "gate", "robust_min"])
    p.add_argument("--filter-modes", nargs="+", default=["new_track_suppression", "track_aware"])
    p.add_argument("--alpha-scales", nargs="+", type=float, default=[0.10, 0.30, 0.50, 1.00])
    p.add_argument("--gamma-assoc", nargs="+", type=float, default=None)
    p.add_argument("--soft-conf-floors", nargs="+", type=float, default=None)
    p.add_argument("--reject-patience", nargs="+", type=int, default=None)
    p.add_argument("--betas", nargs="+", default=["none", "0.7"])
    p.add_argument("--taus", nargs="+", type=float, default=[0.05, 0.20, 0.40])
    p.add_argument("--tau-existing-grid", nargs="+", type=float, default=None)
    p.add_argument("--tau-new-grid", nargs="+", type=float, default=None)
    p.add_argument("--risk-tau-grid", nargs="+", type=float, default=None)
    p.add_argument("--new-conf-tau-grid", nargs="+", type=float, default=None)
    p.add_argument("--confirm-age", nargs="+", type=int, default=None)
    p.add_argument("--confirmed-conf-floors", nargs="+", type=float, default=None)
    p.add_argument("--tracking-top-k", type=int, default=12)
    args = p.parse_args()

    cfg = load_config(args.config)
    results = Path(args.results)
    sequences = load_split(args.split_config, args.split)
    gt = VisDroneDataset(cfg["dataset"]["root"], "val", subset_hint="VID").all_annotations(sequences)
    s0 = pd.read_csv(results / f"vid_{args.model}_s0_baseline.csv")
    s1 = pd.read_csv(results / f"vid_{args.model}_s1_fgsm_eps_{args.eps}.csv")
    betas = [None if str(x).lower() in {"none", "null", "off"} else float(x) for x in args.betas]
    clean_base = evaluate(gt, s0, "S0", 0.0, cfg=cfg, include_map=False, include_tracking=False)
    clean_limit = float(cfg["filtering"].get("robust_clean_f1_drop_limit", 0.02))
    clean_fallback = float(cfg["filtering"].get("robust_clean_f1_drop_fallback", 0.03))
    recall_limit = float(cfg["filtering"].get("robust_recall_drop_limit", 0.03))
    min_rejection = float(cfg["filtering"].get("min_attack_rejection_rate", 0.01))
    gamma_values = args.gamma_assoc or cfg["filtering"].get("gamma_assoc_grid", [float(cfg["filtering"].get("gamma_assoc", 0.2))])
    soft_floors = args.soft_conf_floors or cfg["filtering"].get("soft_conf_floor_grid", [float(cfg["filtering"].get("soft_conf_floor", 0.0))])
    patience_values = args.reject_patience or cfg["filtering"].get("reject_patience_grid", [int(cfg["filtering"].get("reject_patience", 1))])
    tau_existing_values = args.tau_existing_grid or args.taus
    tau_new_values = args.tau_new_grid or args.taus
    risk_tau_values = args.risk_tau_grid or cfg["filtering"].get("risk_tau_grid", [0.7])
    new_conf_tau_values = args.new_conf_tau_grid or cfg["filtering"].get("new_conf_tau_grid", [0.2])
    confirm_age_values = args.confirm_age or cfg["filtering"].get("confirm_age_grid", [int(cfg["filtering"].get("confirm_age", 3))])
    confirmed_floors = args.confirmed_conf_floors or cfg["filtering"].get("confirmed_conf_floor_grid", [float(cfg["filtering"].get("confirmed_conf_floor", 0.03))])
    k_new_track = float(cfg["filtering"].get("k_new_track", 1.0))
    tau_soft_update = float(cfg["filtering"].get("tau_soft_update", 0.05))
    max_missed_frames = int(cfg["filtering"].get("max_missed_frames", 5))
    s1_base = evaluate(gt, s1, "S1", args.eps, cfg=cfg, include_map=False, include_tracking=False)

    rows, frames = [], {}
    for t_norm in args.t_norms:
        for k_variant in args.k_variants:
            for alpha_scale in args.alpha_scales:
                for gamma_assoc in gamma_values:
                    for filter_mode in args.filter_modes:
                        floors = soft_floors if filter_mode == "soft_reweight" else [None]
                        patience_grid = patience_values if filter_mode in {"delayed_hard_filter", "track_aware", "new_track_suppression"} else [1]
                        for soft_conf_floor in floors:
                            for reject_patience in patience_grid:
                                for beta in betas:
                                    for tau_existing in tau_existing_values:
                                        for tau_new in tau_new_values:
                                            if float(tau_new) < float(tau_existing):
                                                continue
                                            for risk_tau in risk_tau_values:
                                                for new_conf_tau in new_conf_tau_values:
                                                    for confirm_age in confirm_age_values:
                                                        for confirmed_conf_floor in confirmed_floors:
                                                            tau = tau_existing
                                                            spec_key = (
                                                                t_norm,
                                                                tau,
                                                                k_variant,
                                                                alpha_scale,
                                                                gamma_assoc,
                                                                filter_mode,
                                                                soft_conf_floor,
                                                                reject_patience,
                                                                beta,
                                                                tau_existing,
                                                                tau_new,
                                                                confirm_age,
                                                                confirmed_conf_floor,
                                                                risk_tau,
                                                                new_conf_tau,
                                                            )
                                                            common = dict(
                                                                mode=filter_mode,
                                                                k_variant=k_variant,
                                                                alpha_scale=alpha_scale,
                                                                beta=beta,
                                                                preassociation=True,
                                                                gamma_assoc=gamma_assoc,
                                                                k_new_track=k_new_track,
                                                                tau_soft_update=tau_soft_update,
                                                                max_missed_frames=max_missed_frames,
                                                                soft_conf_floor=float(soft_conf_floor or 0.0),
                                                                reject_patience=reject_patience,
                                                                tau_existing=tau_existing,
                                                                tau_new=tau_new,
                                                                confirm_age=confirm_age,
                                                                max_confirm_missed=int(cfg["filtering"].get("max_confirm_missed", 1)),
                                                                confirmed_conf_floor=float(confirmed_conf_floor),
                                                                risk_tau=float(risk_tau),
                                                                new_conf_tau=float(new_conf_tau),
                                                                risk_weights=tuple(cfg["filtering"].get("risk_weights", [0.4, 0.3, 0.3])),
                                                            )
                                                            clean = apply_tnorm(s0, t_norm, tau, **common)
                                                            attack = apply_tnorm(s1, t_norm, tau, **common)
                                                            frames[spec_key] = (clean, attack)
                                                            cm = evaluate(gt, clean, "S2_clean", 0.0, t_norm, tau, cfg=cfg, include_map=False, include_tracking=False)
                                                            am = evaluate(gt, attack, "S2", args.eps, t_norm, tau, cfg=cfg, include_map=False, include_tracking=False)
                                                            br = attack_success_breakdown(gt, s0, attack)
                                                            asr = br["asr_total"]
                                                            clean_drop = clean_base["F1"] - cm["F1"]
                                                            recall_drop = s1_base["recall"] - am["recall"]
                                                            fn_rate = am["FN"] / max(1, am["TP"] + am["FN"])
                                                            robust_score = 0.50 * am["F1"] - 0.25 * asr - 0.15 * fn_rate - 0.10 * clean_drop
                                                            rows.append(
                                                                {
                                                                    "model_name": args.model,
                                                                    "eps": args.eps,
                                                                    "t_norm": t_norm,
                                                                    "tau_Q": tau,
                                                                    "tau_existing": tau_existing,
                                                                    "tau_new": tau_new,
                                                                    "risk_tau": risk_tau,
                                                                    "new_conf_tau": new_conf_tau,
                                                                    "confirm_age": confirm_age,
                                                                    "confirmed_conf_floor": confirmed_conf_floor,
                                                                    "k_variant": k_variant,
                                                                    "alpha_scale": alpha_scale,
                                                                    "gamma_assoc": gamma_assoc,
                                                                    "filter_mode": filter_mode,
                                                                    "soft_conf_floor": soft_conf_floor,
                                                                    "reject_patience": reject_patience,
                                                                    "beta": beta,
                                                                    "clean_F1": cm["F1"],
                                                                    "attack_F1": am["F1"],
                                                                    "attack_IDF1": None,
                                                                    "attack_ASR": asr,
                                                                    "FN_rate_attack": fn_rate,
                                                                    "recall_drop_vs_S1": recall_drop,
                                                                    "clean_drop": clean_drop,
                                                                    "rejection_rate": am["rejection_rate"],
                                                                    "robust_score": robust_score,
                                                                    "passes_clean_constraint": clean_drop <= clean_limit,
                                                                    "passes_clean_fallback": clean_drop <= clean_fallback,
                                                                    "passes_recall_constraint": recall_drop <= recall_limit,
                                                                    "passes_rejection_constraint": am["rejection_rate"] >= min_rejection,
                                                                    "reason": "",
                                                                    "selected": False,
                                                                }
                                                            )
    eligible = [r for r in rows if r["passes_clean_constraint"] and r["passes_rejection_constraint"] and r["passes_recall_constraint"]]
    fallback = [r for r in rows if r["passes_clean_fallback"] and r["passes_rejection_constraint"] and r["passes_recall_constraint"]]
    if eligible:
        selection_status = "selected"
        best = sorted(eligible, key=lambda r: (-r["robust_score"], r["attack_ASR"], r["tau_Q"]))[0]
        for row in rows:
            row["selected"] = all(row[k] == best[k] for k in ["t_norm", "tau_Q", "k_variant", "alpha_scale", "gamma_assoc", "filter_mode", "soft_conf_floor", "reject_patience", "beta"])
    elif fallback:
        selection_status = "fallback_clean_003"
        best = sorted(fallback, key=lambda r: (-r["robust_score"], r["attack_ASR"], r["tau_Q"]))[0]
        for row in rows:
            row["selected"] = all(row[k] == best[k] for k in ["t_norm", "tau_Q", "k_variant", "alpha_scale", "gamma_assoc", "filter_mode", "soft_conf_floor", "reject_patience", "beta"])
    else:
        selection_status = "not_selected"
        best = sorted(rows, key=lambda r: (-r["rejection_rate"], r["clean_drop"], -r["robust_score"]))[0]
        for row in rows:
            if not row["passes_rejection_constraint"]:
                row["reason"] = "no_candidate_with_nonzero_rejection"
            elif not row["passes_recall_constraint"]:
                row["reason"] = "recall_constraint_failed"
            else:
                row["reason"] = "clean_constraint_failed"
    table = pd.DataFrame(rows)
    candidate_truth = candidate_rejection_truth(gt, table, frames)
    if not candidate_truth.empty:
        merge_keys = [c for c in _candidate_keys() if c in table.columns and c in candidate_truth.columns]
        table = table.merge(candidate_truth, on=merge_keys, how="left")
    save(k_distribution(args.model, args.eps, table, frames), results / "k_distribution.csv")
    save(rejection_debug(gt, args.model, args.eps, table, frames, cfg), results / "rejection_debug.csv")
    err = error_type_summary(gt, s0, s1, args.model, args.eps, table, frames, cfg, tracking_top_k=args.tracking_top_k)
    table, best, selection_status = finalize_tracking_selection(table, err, cfg, gt)
    save(table, results / "robust_threshold_selection.csv")
    save(table, results / "kinematics_param_selection.csv")
    save(err, results / "error_type_summary.csv")
    save(error_intensity_summary(err, gt), results / "error_intensity_summary.csv")
    payload = {
        "selection_scope": "model_eps_level",
        "selection_status": selection_status,
        "reason": None if selection_status != "not_selected" else "no_safe_non_noop_candidate",
        "selected": {
            args.model: {
                f"eps_{args.eps}": {
                    "k_variant": best["k_variant"],
                    "alpha_scale": float(best["alpha_scale"]),
                    "gamma_assoc": float(best["gamma_assoc"]),
                    "t_norm": best["t_norm"],
                    "tau_Q": float(best["tau_Q"]),
                    "tau_existing": float(best.get("tau_existing", best["tau_Q"])),
                    "tau_new": float(best.get("tau_new", best["tau_Q"])),
                    "risk_tau": float(best.get("risk_tau", cfg["filtering"].get("risk_tau_grid", [0.7])[0])),
                    "new_conf_tau": float(best.get("new_conf_tau", cfg["filtering"].get("new_conf_tau_grid", [0.2])[0])),
                    "filter_mode": best["filter_mode"],
                    "confirm_age": int(best.get("confirm_age", cfg["filtering"].get("confirm_age", 3))),
                    "confirmed_conf_floor": float(best.get("confirmed_conf_floor", cfg["filtering"].get("confirmed_conf_floor", 0.03))),
                    "soft_conf_floor": None if pd.isna(best["soft_conf_floor"]) else best["soft_conf_floor"],
                    "reject_patience": int(best["reject_patience"]),
                    "beta": best["beta"],
                }
            }
        },
    }
    with open(results / "selected_defense_params.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False)
    selected_clean, selected_attack = frames[_frame_key(best)]
    sample = selected_attack.head(200)
    save(selected_attack, results / f"vid_{args.model}_s2_selected_eps_{args.eps}.csv")
    naive_path = results / f"vid_{args.model}_s_naive_eps_{args.eps}.csv"
    summary = [
        evaluate(gt, s0, "S0", 0.0, cfg=cfg, include_map=False, include_tracking=False),
        evaluate(gt, s1, "S1", args.eps, cfg=cfg, include_map=False, include_tracking=False),
    ]
    if naive_path.exists():
        summary.append(evaluate(gt, pd.read_csv(naive_path), "S_naive", args.eps, "confidence", None, cfg=cfg, include_map=False, include_tracking=False))
    s2m = evaluate(gt, selected_attack, "S2", args.eps, best["t_norm"], best["tau_Q"], cfg=cfg, include_map=False, include_tracking=False)
    s2m.update(attack_success_breakdown(gt, s0, selected_attack))
    s2m["ASR_track"] = s2m["asr_total"]
    summary.append(s2m)
    save(pd.DataFrame(summary), results / "summary_metrics.csv")
    tracking_summary = [
        evaluate(gt, s0, "S0", 0.0, cfg=cfg, include_map=False, include_tracking=True),
        evaluate(gt, s1, "S1", args.eps, cfg=cfg, include_map=False, include_tracking=True),
        evaluate(gt, selected_attack, "S2", args.eps, best["t_norm"], best["tau_Q"], cfg=cfg, include_map=False, include_tracking=True),
    ]
    save(pd.DataFrame(tracking_summary), results / "summary_metrics_tracking_selected.csv")
    save(track_status_summary(gt, args.model, args.eps, selected_attack, cfg), results / "track_status_summary.csv")
    save(fp_source_summary(gt, args.model, args.eps, selected_attack, cfg), results / "fp_source_summary.csv")
    truth_audit = rejection_truth_audit(gt, args.model, args.eps, selected_attack, cfg)
    save(truth_audit, results / "rejection_truth_audit.csv")
    save(rejection_truth_summary(truth_audit), results / "rejection_truth_summary.csv")
    save(status_transition_summary(gt, args.model, args.eps, selected_attack, cfg), results / "status_transition_summary.csv")
    save(oracle_new_fp_filter_summary(gt, s0, s1, selected_attack, args.model, args.eps, cfg), results / "oracle_new_fp_filter_summary.csv")
    keep = [
        "sequence_id",
        "frame_id",
        "pred_track_id",
        "confidence",
        "k_center",
        "k_iou",
        "k_combined",
        "k_i",
        "Q_raw",
        "Q_smooth",
        "accepted",
        "filter_mode",
    ]
    sample[[c for c in keep if c in sample.columns]].head(50).to_csv(results / "kinematics_debug_sample.csv", index=False)
    lifecycle_cols = [
        "sequence_id",
        "frame_id",
        "pred_track_id",
        "track_status",
        "track_age",
        "missed_frames",
        "assoc_score",
        "confidence",
        "k_i",
        "Q_i",
        "accepted",
        "suspicious",
        "low_Q_streak",
        "filter_action",
    ]
    selected_attack[[c for c in lifecycle_cols if c in selected_attack.columns]].to_csv(results / "track_lifecycle_debug.csv", index=False)


def k_distribution(model: str, eps: float, table: pd.DataFrame, frames: dict) -> pd.DataFrame:
    rows = []
    for row in table.to_dict("records"):
        frame = frames[_frame_key(row)][1]
        k = frame["k_i"].astype(float)
        rows.append(
            {
                "model_name": model,
                "eps": eps,
                "scenario": "S2",
                "k_variant": row["k_variant"],
                "alpha_scale": row["alpha_scale"],
                "gamma_assoc": row["gamma_assoc"],
                "filter_mode": row["filter_mode"],
                "tau_Q": row["tau_Q"],
                "count": len(k),
                "k_min": k.min(),
                "k_p01": k.quantile(0.01),
                "k_p05": k.quantile(0.05),
                "k_p10": k.quantile(0.10),
                "k_p25": k.quantile(0.25),
                "k_median": k.median(),
                "k_mean": k.mean(),
                "k_p75": k.quantile(0.75),
                "k_p90": k.quantile(0.90),
                "k_p95": k.quantile(0.95),
                "k_p99": k.quantile(0.99),
                "k_max": k.max(),
                "share_k_lt_0_9": float((k < 0.9).mean()),
                "share_k_lt_0_7": float((k < 0.7).mean()),
                "share_k_lt_0_5": float((k < 0.5).mean()),
                "share_k_lt_0_3": float((k < 0.3).mean()),
                "share_matched_existing_track": float((frame["assoc_status"] == "matched_existing_track").mean()) if "assoc_status" in frame else None,
                "share_new_track": float((frame["assoc_status"] == "new_track").mean()) if "assoc_status" in frame else None,
                "share_low_assoc": float((frame["assoc_status"] == "low_assoc").mean()) if "assoc_status" in frame else None,
            }
        )
    return pd.DataFrame(rows)


def rejection_debug(gt: pd.DataFrame, model: str, eps: float, table: pd.DataFrame, frames: dict, cfg: dict) -> pd.DataFrame:
    rows = []
    for row in table.to_dict("records"):
        before = frames[_frame_key(row)][1]
        after = before[before["accepted"] == True]
        mb = evaluate(gt, before.assign(accepted=True), "before", eps, cfg=cfg, include_map=False, include_tracking=False)
        ma = evaluate(gt, before, "after", eps, cfg=cfg, include_map=False, include_tracking=False)
        rejected = before[before["accepted"] != True]
        accepted = before[before["accepted"] == True]
        rows.append(
            {
                "model_name": model,
                "eps": eps,
                "scenario": "S2",
                "k_variant": row["k_variant"],
                "alpha_scale": row["alpha_scale"],
                "gamma_assoc": row["gamma_assoc"],
                "t_norm": row["t_norm"],
                "tau_Q": row["tau_Q"],
                "filter_mode": row["filter_mode"],
                "soft_conf_floor": row["soft_conf_floor"],
                "reject_patience": row["reject_patience"],
                "num_before": len(before),
                "num_after": len(after),
                "num_rejected": len(rejected),
                "rejection_rate": len(rejected) / max(1, len(before)),
                "rejected_mean_confidence": rejected["confidence_original"].astype(float).mean() if len(rejected) else None,
                "rejected_mean_k": rejected["k_i"].astype(float).mean() if len(rejected) else None,
                "accepted_mean_confidence": accepted["confidence_original"].astype(float).mean() if len(accepted) else None,
                "accepted_mean_k": accepted["k_i"].astype(float).mean() if len(accepted) else None,
                "share_matched_existing_track": float((before["assoc_status"] == "matched_existing_track").mean()) if "assoc_status" in before else None,
                "share_new_track": float((before["assoc_status"] == "new_track").mean()) if "assoc_status" in before else None,
                "share_low_assoc": float((before["assoc_status"] == "low_assoc").mean()) if "assoc_status" in before else None,
                "FP_before": mb["FP"],
                "FP_after": ma["FP"],
                "FN_before": mb["FN"],
                "FN_after": ma["FN"],
            }
        )
    return pd.DataFrame(rows)


def track_status_summary(gt: pd.DataFrame, model: str, eps: float, frame: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    rows = []
    before_all = evaluate(gt, frame.assign(accepted=True), "before", eps, cfg=cfg, include_map=False, include_tracking=True)
    after_all = evaluate(gt, frame, "after", eps, cfg=cfg, include_map=False, include_tracking=True)
    for status, g in frame.groupby("track_status", dropna=False):
        accepted = g[g["accepted"] == True]
        rows.append(
            {
                "model_name": model,
                "eps": eps,
                "scenario": "S2",
                "filter_mode": frame["filter_mode"].dropna().iloc[0] if "filter_mode" in frame and len(frame["filter_mode"].dropna()) else None,
                "track_status": status,
                "num_before": len(g),
                "num_after": len(accepted),
                "num_rejected": len(g) - len(accepted),
                "rejection_rate": (len(g) - len(accepted)) / max(1, len(g)),
                "mean_confidence": g["confidence_original"].astype(float).mean() if "confidence_original" in g and len(g) else None,
                "mean_k": g["k_i"].astype(float).mean() if "k_i" in g and len(g) else None,
                "mean_Q": g["Q_i"].astype(float).mean() if "Q_i" in g and len(g) else None,
                "FP_before": before_all["FP"],
                "FP_after": after_all["FP"],
                "FN_before": before_all["FN"],
                "FN_after": after_all["FN"],
                "IDSW": after_all["IDSW"],
                "track_breaks": after_all["track_breaks"],
            }
        )
    return pd.DataFrame(rows)


def fp_source_summary(gt: pd.DataFrame, model: str, eps: float, frame: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    before = evaluate(gt, frame.assign(accepted=True), "before", eps, cfg=cfg, include_map=False, include_tracking=True)
    after = evaluate(gt, frame, "after", eps, cfg=cfg, include_map=False, include_tracking=True)
    fp_rows = false_positive_rows(gt, frame)
    rejected = frame[frame["accepted"] != True] if "accepted" in frame else frame.iloc[0:0]
    new_statuses = {"new_candidate", "tentative_existing"}
    confirmed_statuses = {"confirmed_existing"}
    unmatched_statuses = {"unmatched_detection"}
    return pd.DataFrame(
        [
            {
                "model_name": model,
                "eps": eps,
                "scenario": "S2",
                "filter_mode": frame["filter_mode"].dropna().iloc[0] if "filter_mode" in frame and len(frame["filter_mode"].dropna()) else None,
                "FP_total": after["FP"],
                "FP_new_track": int((fp_rows["track_status"].isin(new_statuses)).sum()) if "track_status" in fp_rows else None,
                "FP_confirmed_track": int((fp_rows["track_status"].isin(confirmed_statuses)).sum()) if "track_status" in fp_rows else None,
                "FP_unmatched": int((fp_rows["track_status"].isin(unmatched_statuses)).sum()) if "track_status" in fp_rows else None,
                "FN_total": after["FN"],
                "FN_confirmed_track": int((rejected["track_status"].isin(confirmed_statuses)).sum()) if "track_status" in rejected else None,
                "FP_before": before["FP"],
                "FN_before": before["FN"],
                "IDSW": after["IDSW"],
                "track_breaks": after["track_breaks"],
            }
        ]
    )


def false_positive_rows(gt: pd.DataFrame, pred: pd.DataFrame) -> pd.DataFrame:
    rows = []
    pred_frames = {k: v for k, v in pred.groupby(["sequence_id", "frame_id"])} if len(pred) else {}
    empty = pred.iloc[0:0]
    for (seq, frame), gt_f in gt.groupby(["sequence_id", "frame_id"]):
        pred_f = pred_frames.get((seq, frame), empty)
        _, _, _, pairs = match_frame(gt_f, pred_f)
        matched = {pi for _, pi in pairs}
        accepted = pred_f[pred_f["accepted"] == True] if "accepted" in pred_f else pred_f
        rows.append(accepted.loc[[idx for idx in accepted.index if idx not in matched]])
    return pd.concat(rows, ignore_index=False) if rows else pred.iloc[0:0]


def rejection_truth_audit(gt: pd.DataFrame, model: str, eps: float, frame: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    rows = []
    pred_frames = {k: v for k, v in frame.groupby(["sequence_id", "frame_id"])} if len(frame) else {}
    empty = frame.iloc[0:0]
    for (seq, frame_id), gt_f in gt.groupby(["sequence_id", "frame_id"]):
        pred_f = pred_frames.get((seq, frame_id), empty)
        _, _, _, pairs = match_frame(gt_f, pred_f.assign(accepted=True))
        pred_to_gt = {pi: gi for gi, pi in pairs}
        for pi, det in pred_f.iterrows():
            is_rejected = bool(det.get("accepted") != True)
            is_soft = float(det.get("confidence", det.get("confidence_original", 0.0))) < float(det.get("confidence_original", det.get("confidence", 0.0)))
            is_suspicious = bool(det.get("suspicious") == True)
            if not (is_rejected or is_soft or is_suspicious):
                continue
            gi = pred_to_gt.get(pi)
            gt_row = gt.loc[gi] if gi is not None else None
            iou_val = _row_iou(det, gt_row) if gt_row is not None else None
            is_tp = gi is not None
            rows.append(
                {
                    "sequence_id": seq,
                    "frame_id": frame_id,
                    "model_name": model,
                    "eps": eps,
                    "scenario": "S2",
                    "filter_mode": det.get("filter_mode"),
                    "k_variant": det.get("k_variant"),
                    "alpha_scale": det.get("alpha_scale"),
                    "gamma_assoc": det.get("gamma_assoc"),
                    "tau_existing": det.get("tau_existing"),
                    "tau_new": det.get("tau_new"),
                    "tau_Q": det.get("tau"),
                    "track_status": det.get("track_status"),
                    "pred_track_id": det.get("pred_track_id"),
                    "gt_track_id_matched": None if gt_row is None else gt_row.get("gt_track_id"),
                    "class_name_pred": det.get("class_name"),
                    "class_name_gt": None if gt_row is None else gt_row.get("class_name"),
                    "confidence": det.get("confidence_original", det.get("confidence")),
                    "k_i": det.get("k_i"),
                    "Q_i": det.get("Q_i"),
                    "assoc_score": det.get("assoc_score"),
                    "filter_action": det.get("filter_action"),
                    "accepted_before": True,
                    "accepted_after": bool(det.get("accepted") == True),
                    "is_rejected": is_rejected,
                    "is_soft_downweighted": is_soft,
                    "matched_gt_iou": iou_val,
                    "is_true_positive_before_filter": is_tp,
                    "is_false_positive_before_filter": not is_tp,
                    "becomes_false_negative_after_filter": bool(is_rejected and is_tp),
                }
            )
    return pd.DataFrame(rows)


def rejection_truth_summary(audit: pd.DataFrame) -> pd.DataFrame:
    if audit.empty:
        return pd.DataFrame()
    rows = []
    keys = ["model_name", "eps", "filter_mode", "k_variant", "alpha_scale", "gamma_assoc", "tau_existing", "tau_new", "tau_Q", "track_status"]
    rejected = audit[audit["is_rejected"] == True]
    for key, g in rejected.groupby(keys, dropna=False):
        num = len(g)
        fp = int(g["is_false_positive_before_filter"].sum())
        tp = int(g["is_true_positive_before_filter"].sum())
        rows.append(
            {
                **dict(zip(keys, key)),
                "num_rejected": num,
                "rejected_TP": tp,
                "rejected_FP": fp,
                "rejected_unknown": num - tp - fp,
                "rejected_FP_rate": fp / max(1, num),
                "rejected_TP_rate": tp / max(1, num),
                "FP_removed_per_FN_created": fp / max(1, tp),
            }
        )
    return pd.DataFrame(rows)


def candidate_rejection_truth(gt: pd.DataFrame, table: pd.DataFrame, frames: dict) -> pd.DataFrame:
    rows = []
    for row in table.to_dict("records"):
        frame = frames[_frame_key(row)][1]
        audit = rejection_truth_audit(gt, row["model_name"], row["eps"], frame, {})
        rejected = audit[audit["is_rejected"] == True] if not audit.empty else audit
        num = len(rejected)
        fp = int(rejected["is_false_positive_before_filter"].sum()) if num else 0
        tp = int(rejected["is_true_positive_before_filter"].sum()) if num else 0
        rows.append(
            {
                **{k: row.get(k) for k in _candidate_keys()},
                "num_rejected_truth": num,
                "rejected_FP": fp,
                "rejected_TP": tp,
                "rejected_FP_rate": fp / max(1, num),
                "rejected_TP_rate": tp / max(1, num),
                "FP_removed_per_FN_created": fp / max(1, tp),
            }
        )
    return pd.DataFrame(rows)


def status_transition_summary(gt: pd.DataFrame, model: str, eps: float, frame: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    pred_frames = {k: v for k, v in frame.groupby(["sequence_id", "frame_id"])} if len(frame) else {}
    empty = frame.iloc[0:0]
    labels = {}
    for (seq, frame_id), gt_f in gt.groupby(["sequence_id", "frame_id"]):
        pred_f = pred_frames.get((seq, frame_id), empty)
        _, _, _, pairs = match_frame(gt_f, pred_f.assign(accepted=True))
        for _, pi in pairs:
            labels[pi] = True
    rows = []
    tmp = frame.copy()
    tmp["is_TP"] = [bool(labels.get(idx, False)) for idx in tmp.index]
    for status, g in tmp.groupby("track_status", dropna=False):
        tp = int(g["is_TP"].sum())
        fp = len(g) - tp
        rows.append(
            {
                "model_name": model,
                "eps": eps,
                "track_status": status,
                "num_detections": len(g),
                "num_TP": tp,
                "num_FP": fp,
                "TP_rate": tp / max(1, len(g)),
                "FP_rate": fp / max(1, len(g)),
                "mean_confidence": g["confidence_original"].astype(float).mean() if "confidence_original" in g else g["confidence"].astype(float).mean(),
                "mean_k_i": g["k_i"].astype(float).mean(),
                "mean_Q_i": g["Q_i"].astype(float).mean(),
                "mean_assoc_score": g["assoc_score"].astype(float).mean(),
            }
        )
    return pd.DataFrame(rows)


def oracle_new_fp_filter_summary(gt: pd.DataFrame, s0: pd.DataFrame, s1: pd.DataFrame, frame: pd.DataFrame, model: str, eps: float, cfg: dict) -> pd.DataFrame:
    oracle = frame.copy()
    fp_rows = false_positive_rows(gt, oracle.assign(accepted=True))
    fp_idx = set(fp_rows.index)
    new_statuses = {"new_candidate", "unmatched_detection"}
    oracle["accepted"] = [
        False if idx in fp_idx and status in new_statuses else bool(acc)
        for idx, status, acc in zip(oracle.index, oracle["track_status"], oracle["accepted"])
    ]
    rows = []
    for scenario, df in [("S1", s1), ("S2_real", frame), ("oracle_new_fp_filter", oracle)]:
        m = evaluate(gt, df, scenario, eps, cfg=cfg, include_map=False, include_tracking=True)
        frames = max(1, len(gt[["sequence_id", "frame_id"]].drop_duplicates()))
        m["failure_intensity"] = (m["FP"] + m["FN"] + (m["IDSW"] or 0) + (m["track_breaks"] or 0)) / frames * 100.0
        rows.append(m)
    return pd.DataFrame(rows)


def _row_iou(pred, gt_row) -> float:
    ix1 = max(float(pred.x1), float(gt_row.x1))
    iy1 = max(float(pred.y1), float(gt_row.y1))
    ix2 = min(float(pred.x2), float(gt_row.x2))
    iy2 = min(float(pred.y2), float(gt_row.y2))
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    pa = max(0.0, float(pred.x2) - float(pred.x1)) * max(0.0, float(pred.y2) - float(pred.y1))
    ga = max(0.0, float(gt_row.x2) - float(gt_row.x1)) * max(0.0, float(gt_row.y2) - float(gt_row.y1))
    return inter / max(1e-6, pa + ga - inter)


def error_type_summary(gt: pd.DataFrame, s0: pd.DataFrame, s1: pd.DataFrame, model: str, eps: float, table: pd.DataFrame, frames: dict, cfg: dict, tracking_top_k: int = 12) -> pd.DataFrame:
    base = evaluate(gt, s1, "S1", eps, cfg=cfg, include_map=False, include_tracking=True)
    base_asr = attack_success_breakdown(gt, s0, s1)["asr_total"]
    ranked = table.sort_values(["selected", "passes_rejection_constraint", "robust_score"], ascending=[False, False, False]).head(tracking_top_k)
    tracking_keys = {_frame_key(row) for row in ranked.to_dict("records")}
    rows = []
    for row in table.to_dict("records"):
        frame = frames[_frame_key(row)][1]
        with_tracking = _frame_key(row) in tracking_keys
        m = evaluate(gt, frame, "S2", eps, row["t_norm"], row["tau_Q"], cfg=cfg, include_map=False, include_tracking=with_tracking)
        br = attack_success_breakdown(gt, s0, frame)
        asr = br["asr_total"]
        rows.append(
            {
                "model_name": model,
                "eps": eps,
                "scenario": "S2",
                "t_norm": row["t_norm"],
                "k_variant": row["k_variant"],
                "alpha_scale": row["alpha_scale"],
                "gamma_assoc": row["gamma_assoc"],
                "filter_mode": row["filter_mode"],
                "tau_Q": row["tau_Q"],
                "tau_existing": row.get("tau_existing", row["tau_Q"]),
                "tau_new": row.get("tau_new", row["tau_Q"]),
                "risk_tau": row.get("risk_tau"),
                "new_conf_tau": row.get("new_conf_tau"),
                "confirm_age": row.get("confirm_age"),
                "confirmed_conf_floor": row.get("confirmed_conf_floor"),
                "reject_patience": row.get("reject_patience"),
                "beta": row.get("beta"),
                "FP": m["FP"],
                "FN": m["FN"],
                "IDF1": m["IDF1"],
                "IDSW": m["IDSW"],
                "track_breaks": m["track_breaks"],
                "ASR_track": asr,
                "delta_FP_vs_S1": m["FP"] - base["FP"],
                "delta_FN_vs_S1": m["FN"] - base["FN"],
                "delta_IDF1_vs_S1": None if m["IDF1"] is None else m["IDF1"] - base["IDF1"],
                "delta_IDSW_vs_S1": None if m["IDSW"] is None else m["IDSW"] - base["IDSW"],
                "delta_track_breaks_vs_S1": None if m["track_breaks"] is None else m["track_breaks"] - base["track_breaks"],
                "delta_ASR_vs_S1": asr - base_asr,
                "ASR_miss": br["asr_miss"],
                "ASR_false_positive": br["asr_fp"],
                "ASR_track_break": br["asr_break"],
                "ASR_id_switch": br["asr_idsw"],
                "ASR_any": asr,
                "num_tracks_affected": br["missed_tracks"] + br["false_positive_tracks"] + br["track_break_tracks"] + br["id_switch_tracks"],
                "num_tracks_total": br["num_gt_tracks"],
            }
        )
    return pd.DataFrame(rows)


def finalize_tracking_selection(table: pd.DataFrame, err: pd.DataFrame, cfg: dict, gt: pd.DataFrame) -> tuple[pd.DataFrame, dict, str]:
    keys = ["t_norm", "k_variant", "alpha_scale", "gamma_assoc", "filter_mode", "tau_Q"]
    for optional in ["tau_existing", "tau_new", "risk_tau", "new_conf_tau", "confirm_age", "confirmed_conf_floor", "reject_patience", "beta"]:
        if optional in table.columns and optional in err.columns:
            keys.append(optional)
    merged = table.merge(err, on=keys, how="left", suffixes=("", "_tracking"))
    frames = max(1, len(gt[["sequence_id", "frame_id"]].drop_duplicates()))
    merged["failure_intensity_delta_vs_S1"] = (
        merged["delta_FP_vs_S1"].fillna(0)
        + merged["delta_FN_vs_S1"].fillna(0)
        + merged["delta_IDSW_vs_S1"].fillna(0)
        + merged["delta_track_breaks_vs_S1"].fillna(0)
    ) / frames * 100.0
    merged["robust_score_v12"] = (
        0.30 * merged["attack_F1"]
        + 0.25 * merged["IDF1"].fillna(0)
        - 0.15 * merged["ASR_any"].fillna(1.0)
        - 0.15 * (merged["IDSW"].fillna(0) / frames)
        - 0.10 * (merged["track_breaks"].fillna(0) / frames)
        - 0.05 * merged["clean_drop"]
    )
    strict = merged[
        (merged["clean_drop"] <= 0.02)
        & (merged["recall_drop_vs_S1"] <= 0.02)
        & (merged["delta_IDSW_vs_S1"] <= 0)
        & (merged["delta_track_breaks_vs_S1"] <= 0)
        & (merged["rejection_rate"] >= float(cfg["filtering"].get("min_attack_rejection_rate", 0.005)))
        & (merged["rejected_FP_rate"].fillna(0) >= 0.50)
        & (merged["FP_removed_per_FN_created"].fillna(0) >= 1.0)
        & (
            (merged["delta_FP_vs_S1"] < 0)
            | (merged["failure_intensity_delta_vs_S1"] < 0)
            | (merged["delta_IDF1_vs_S1"] > 0)
        )
    ]
    fallback = merged[
        (merged["clean_drop"] <= 0.03)
        & (merged["recall_drop_vs_S1"] <= 0.03)
        & (merged["delta_IDSW_vs_S1"] <= 2)
        & (merged["delta_track_breaks_vs_S1"] <= 5)
        & (merged["rejection_rate"] >= float(cfg["filtering"].get("min_attack_rejection_rate", 0.005)))
        & (merged["rejected_FP_rate"].fillna(0) >= 0.50)
        & (merged["FP_removed_per_FN_created"].fillna(0) >= 1.0)
        & (merged["failure_intensity_delta_vs_S1"] <= 1.0)
    ]
    if len(strict):
        status = "selected"
        best_row = strict.sort_values("robust_score_v12", ascending=False).iloc[0]
    elif len(fallback):
        status = "fallback_tracking_tolerance"
        best_row = fallback.sort_values("robust_score_v12", ascending=False).iloc[0]
    else:
        status = "not_selected"
        non_noop = merged[merged["rejection_rate"] >= float(cfg["filtering"].get("min_attack_rejection_rate", 0.005))]
        if len(non_noop):
            best_row = non_noop.sort_values(
                ["rejected_FP_rate", "FP_removed_per_FN_created", "rejection_rate", "robust_score_v12"],
                ascending=[False, False, False, False],
            ).iloc[0]
        else:
            pool = merged.dropna(subset=["robust_score_v12"])
            best_row = (pool if len(pool) else merged).sort_values("robust_score", ascending=False).iloc[0]
    out = table.copy()
    out["selected"] = False
    out["selection_status"] = status
    out["failure_intensity_delta_vs_S1"] = merged["failure_intensity_delta_vs_S1"]
    out["robust_score_v12"] = merged["robust_score_v12"]
    for col in ["delta_IDSW_vs_S1", "delta_track_breaks_vs_S1", "delta_IDF1_vs_S1", "delta_FP_vs_S1", "ASR_any"]:
        if col in merged:
            out[col] = merged[col]
    for col in ["rejected_FP_rate", "rejected_TP_rate", "FP_removed_per_FN_created", "rejected_FP", "rejected_TP"]:
        if col in merged:
            out[col] = merged[col]
    best = best_row.to_dict()
    mask = pd.Series(True, index=out.index)
    for key in keys:
        lhs = out[key]
        rhs = best.get(key)
        if pd.isna(rhs):
            mask &= lhs.isna()
        else:
            mask &= lhs == rhs
    out.loc[mask, "selected"] = status != "not_selected"
    if status == "not_selected":
        out.loc[:, "reason"] = "no_safe_non_noop_candidate"
    return out, best, status


def _candidate_keys() -> list[str]:
    return [
        "model_name",
        "eps",
        "t_norm",
        "tau_Q",
        "tau_existing",
        "tau_new",
        "risk_tau",
        "new_conf_tau",
        "confirm_age",
        "confirmed_conf_floor",
        "k_variant",
        "alpha_scale",
        "gamma_assoc",
        "filter_mode",
        "soft_conf_floor",
        "reject_patience",
        "beta",
    ]


def error_intensity_summary(error_df: pd.DataFrame, gt: pd.DataFrame) -> pd.DataFrame:
    frames = max(1, len(gt[["sequence_id", "frame_id"]].drop_duplicates()))
    out = error_df.copy()
    out["FP_per_100_frames"] = out["FP"] / frames * 100.0
    out["FN_per_100_frames"] = out["FN"] / frames * 100.0
    out["IDSW_per_100_frames"] = out["IDSW"] / frames * 100.0
    out["track_breaks_per_100_frames"] = out["track_breaks"] / frames * 100.0
    out["failures_per_100_frames"] = (out["FP"] + out["FN"] + out["IDSW"].fillna(0) + out["track_breaks"].fillna(0)) / frames * 100.0
    keep = [
        "model_name",
        "eps",
        "scenario",
        "t_norm",
        "k_variant",
        "alpha_scale",
        "gamma_assoc",
        "filter_mode",
        "tau_Q",
        "tau_existing",
        "tau_new",
        "risk_tau",
        "new_conf_tau",
        "confirm_age",
        "reject_patience",
        "FP_per_100_frames",
        "FN_per_100_frames",
        "IDSW_per_100_frames",
        "track_breaks_per_100_frames",
        "failures_per_100_frames",
        "ASR_any",
        "ASR_miss",
        "ASR_false_positive",
        "ASR_track_break",
        "ASR_id_switch",
    ]
    return out[[c for c in keep if c in out.columns]]


def _frame_key(row: dict) -> tuple:
    beta = row.get("beta")
    if pd.isna(beta):
        beta = None
    soft_conf_floor = row.get("soft_conf_floor")
    if pd.isna(soft_conf_floor):
        soft_conf_floor = None
    return (
        row["t_norm"],
        row["tau_Q"],
        row["k_variant"],
        row["alpha_scale"],
        row["gamma_assoc"],
        row["filter_mode"],
        soft_conf_floor,
        row["reject_patience"],
        beta,
        row.get("tau_existing", row["tau_Q"]),
        row.get("tau_new", row["tau_Q"]),
        row.get("confirm_age"),
        row.get("confirmed_conf_floor"),
        row.get("risk_tau"),
        row.get("new_conf_tau"),
    )


if __name__ == "__main__":
    main()
