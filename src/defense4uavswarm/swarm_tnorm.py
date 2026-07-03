from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from defense4uavswarm.datasets.visdrone import VISDRONE_CLASSES, VisDroneDataset
from defense4uavswarm.matrix import load_split_sequences
from defense4uavswarm.swarm import compute_inter_agent_consistency, jitter_bbox, load_frame_index, transform_bbox
from defense4uavswarm.xai import compute_cam, compute_xai_score, select_xai_candidates


def run_swarm_tnorm_smoke(
    swarm_config: str | Path,
    split_config: str | Path,
    split: str,
    eps_values: list[float],
    scenarios: list[str],
    t_norms: list[str],
    output_dir: str | Path,
    limit_sequences: int | None = None,
    xai_method: str = "eigencam",
    xai_max_per_frame: int = 5,
    semantic_max_frames: int | None = None,
    pseudo_attack_config: str | Path | None = None,
    swarm_root: str | Path | None = None,
    use_selected_params: str | Path | None = None,
    sequence_batch_size: int | None = None,
    min_free_disk_gb: float | None = None,
    max_temp_gb: float | None = None,
    cleanup_temp: bool = False,
    xai_newtrack_only: bool = False,
    xai_boundary_margins: list[float] | None = None,
    xai_q_triggers: list[float] | None = None,
    xai_mins: list[float] | None = None,
    xai_floors: list[float] | None = None,
    xai_veto_modes: list[str] | None = None,
    reweight_modes: list[str] | None = None,
    q_floors: list[float] | None = None,
    gammas: list[float] | None = None,
    q_hard_mins: list[float] | None = None,
    q_floors_new: list[float] | None = None,
    q_floors_existing: list[float] | None = None,
    q_new_modes: list[str] | None = None,
    q_new_mins: list[float] | None = None,
    new_track_thresholds: list[float] | None = None,
    existing_track_thresholds: list[float] | None = None,
) -> None:
    swarm_cfg = _load_yaml(swarm_config)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    disk_before = disk_report(Path.cwd())
    if min_free_disk_gb is not None:
        free_gb = shutil.disk_usage(Path.cwd()).free / (1024**3)
        if free_gb < min_free_disk_gb:
            raise RuntimeError(f"Not enough free disk: {free_gb:.1f}G < required {min_free_disk_gb:.1f}G")
    root = Path(swarm_root) if swarm_root else Path(swarm_cfg["output_root"]) / split
    frame_index_path = root / "metadata" / "frame_index.csv"
    if not frame_index_path.exists():
        raise FileNotFoundError(f"Build pseudo-swarm first: missing {frame_index_path}")
    selected_cfg = _load_yaml(use_selected_params) if use_selected_params else {}
    if selected_cfg:
        if selected_cfg.get("selected_t_norm"):
            t_norms = [selected_cfg["selected_t_norm"]]
        if selected_cfg.get("selected_reweight_mode"):
            reweight_modes = [selected_cfg["selected_reweight_mode"]]
        if selected_cfg.get("selected_q_floor") is not None:
            q_floors = [float(selected_cfg["selected_q_floor"])]
        if selected_cfg.get("selected_q_floor_new") is not None:
            q_floors_new = [float(selected_cfg["selected_q_floor_new"])]
        if selected_cfg.get("selected_q_floor_existing") is not None:
            q_floors_existing = [float(selected_cfg["selected_q_floor_existing"])]
        if selected_cfg.get("selected_q_new_mode"):
            q_new_modes = [selected_cfg["selected_q_new_mode"]]
        if selected_cfg.get("selected_q_new_min") is not None:
            q_new_mins = [float(selected_cfg["selected_q_new_min"])]
        if selected_cfg.get("selected_q_hard_min") is not None:
            q_hard_mins = [float(selected_cfg["selected_q_hard_min"])]
        if selected_cfg.get("selected_new_track_threshold") is not None:
            new_track_thresholds = [float(selected_cfg["selected_new_track_threshold"])]
        if selected_cfg.get("selected_existing_track_threshold") is not None:
            existing_track_thresholds = [float(selected_cfg["selected_existing_track_threshold"])]
    frame_index = load_frame_index(root)
    sequences = load_split_sequences(split_config, split) or sorted(frame_index["sequence_id"].unique())
    if limit_sequences:
        sequences = sequences[:limit_sequences]
    frame_index = frame_index[frame_index["sequence_id"].isin(sequences)].copy()
    ds = VisDroneDataset(swarm_cfg["source_root"], "val", subset_hint="VID")
    gt = ds.all_annotations(sequences)
    clean_detections = _pseudo_detections(gt, frame_index)
    detections = clean_detections
    attack_events = pd.DataFrame()
    attack_cfg = _load_yaml(pseudo_attack_config) if pseudo_attack_config else {}
    if attack_cfg:
        detections, attack_events = inject_pseudo_attack(detections, attack_cfg)
        attack_events.to_csv(out / "attack_event_audit.csv", index=False)
        attack_event_summary(attack_events).to_csv(out / "attack_event_summary.csv", index=False)
    clean_features, _ = compute_inter_agent_consistency(clean_detections, frame_index, iou_min=0.3, s_missing_policy="neutral")
    clean_features = prepare_features(clean_features, eps_values)
    features, matches = compute_inter_agent_consistency(detections, frame_index, iou_min=0.3, s_missing_policy="neutral")
    features = prepare_features(features, eps_values)
    xai_audit = pd.DataFrame()
    xai_summary = pd.DataFrame()
    if any(s.lower() in {"s3_tnorm_xai", "s3", "s3_tnorm_xai_hard", "s3_tnorm_xai_soft", "s4_soft_recovery"} for s in scenarios):
        features, xai_audit, xai_summary = apply_xai_smoke(features, frame_index, xai_method, xai_max_per_frame, semantic_max_frames)
    tau_q_grid = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]
    tau_conf_grid = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60]
    reweight_modes = reweight_modes or ["multiplicative_floor"]
    q_floors = q_floors or [0.60, 0.70, 0.80]
    q_floors_new = q_floors_new or [0.50, 0.60, 0.70]
    q_floors_existing = q_floors_existing or [0.90]
    q_new_modes = q_new_modes or ["min_ks"]
    q_new_mins = q_new_mins or [0.0, 0.05, 0.10]
    xai_boundary_margins = xai_boundary_margins or [0.05]
    xai_q_triggers = xai_q_triggers or [0.30]
    xai_mins = xai_mins or [0.20]
    xai_floors = xai_floors or [0.80]
    xai_veto_modes = xai_veto_modes or ["soft"]
    gammas = gammas or [0.25, 0.50]
    q_hard_mins = q_hard_mins or [0.0]
    new_track_thresholds = new_track_thresholds or [0.30]
    existing_track_thresholds = existing_track_thresholds or [0.10]
    rows = []
    threshold_rows = []
    feature_frames = []
    total_frames = int(frame_index[["sequence_id", "frame_id"]].drop_duplicates().shape[0])
    expected_gt = int((clean_features["track_id"] != -1).sum())
    for scenario in scenarios:
        if scenario.lower() in {"s0", "s0_clean", "s1", "s1_fgsm"}:
            frame = clean_features.copy() if scenario.lower().startswith("s0") else features.copy()
            frame["scenario"] = "S0_clean" if scenario.lower().startswith("s0") else "S1_fgsm"
            frame["Q_i"] = frame["c_i"]
            frame["t_norm"] = None
            frame["tau_Q"] = None
            frame["accepted"] = True
            rows.append(_summary(frame, str(frame["scenario"].iloc[0]), None, None, total_frames, expected_gt))
            feature_frames.append(frame)
        elif scenario.lower() in {"s_naive", "snaive"}:
            for tau in tau_conf_grid:
                frame = features.copy()
                frame["scenario"] = "S_naive"
                frame["Q_i"] = frame["c_i"]
                frame["t_norm"] = "confidence"
                frame["tau_Q"] = tau
                frame["accepted"] = frame["Q_i"] >= tau
                rows.append(_summary(frame, "S_naive", "confidence", tau, total_frames, expected_gt))
                feature_frames.append(frame)
                threshold_rows.append({"scenario": "S_naive", "t_norm": "confidence", "tau_Q": tau, "selected": False})
        elif scenario.lower() in {"s2_tnorm_no_xai", "s2", "s2_tnorm_hard"}:
            for norm in t_norms:
                for tau in tau_q_grid:
                    frame = features.copy()
                    frame["scenario"] = "S2_tnorm_no_xai" if scenario.lower() in {"s2_tnorm_no_xai", "s2"} else "S2_tnorm_hard"
                    frame["x_i"] = 1.0
                    frame["t_norm"] = norm
                    frame["tau_Q"] = tau
                    frame["Q_i"] = _q(frame, norm)
                    frame["accepted"] = frame["Q_i"] >= tau
                    rows.append(_summary(frame, frame["scenario"].iloc[0], norm, tau, total_frames, expected_gt))
                    feature_frames.append(frame)
                    threshold_rows.append({"scenario": frame["scenario"].iloc[0], "t_norm": norm, "tau_Q": tau, "selected": False})
        elif scenario.lower() in {"s3_tnorm_xai", "s3", "s3_tnorm_xai_hard"}:
            for norm in t_norms:
                for tau in tau_q_grid:
                    frame = features.copy()
                    frame["scenario"] = "S3_tnorm_xai" if scenario.lower() in {"s3_tnorm_xai", "s3"} else "S3_tnorm_xai_hard"
                    frame["t_norm"] = norm
                    frame["tau_Q"] = tau
                    frame["Q_i"] = _q(frame, norm)
                    frame["accepted"] = frame["Q_i"] >= tau
                    rows.append(_summary(frame, frame["scenario"].iloc[0], norm, tau, total_frames, expected_gt))
                    feature_frames.append(frame)
                    threshold_rows.append({"scenario": frame["scenario"].iloc[0], "t_norm": norm, "tau_Q": tau, "selected": False})
        elif scenario.lower() in {"s2_tnorm_soft", "s3_tnorm_xai_soft", "s4_soft_recovery"}:
            use_xai = scenario.lower() in {"s3_tnorm_xai_soft", "s4_soft_recovery"}
            scenario_name = {"s2_tnorm_soft": "S2_tnorm_soft", "s3_tnorm_xai_soft": "S3_tnorm_xai_soft", "s4_soft_recovery": "S4_soft_recovery"}[scenario.lower()]
            for norm in t_norms:
                for mode in reweight_modes:
                    floors = q_floors if mode == "multiplicative_floor" else [None]
                    powers = gammas if mode == "power" else [None]
                    for q_floor in floors:
                        for gamma in powers:
                            for q_hard_min in q_hard_mins:
                                for new_thr in new_track_thresholds:
                                    for existing_thr in existing_track_thresholds:
                                        frame = soft_reweight_frame(features, scenario_name, norm, use_xai, mode, q_floor, gamma, q_hard_min, new_thr, existing_thr)
                                        rows.append(_summary(frame, scenario_name, norm, None, total_frames, expected_gt))
                                        feature_frames.append(frame)
                                        threshold_rows.append({"scenario": scenario_name, "t_norm": norm, "tau_Q": None, "reweight_mode": mode, "q_floor": q_floor, "gamma": gamma, "q_hard_min": q_hard_min, "new_track_threshold": new_thr, "existing_track_threshold": existing_thr, "selected": False})
        elif scenario.lower() in {"s2_tnorm_soft_v2", "s2_tnorm_newgate", "s2_tnorm_adaptive"}:
            scenario_name = {
                "s2_tnorm_soft_v2": "S2_tnorm_soft_v2",
                "s2_tnorm_newgate": "S2_tnorm_newgate",
                "s2_tnorm_adaptive": "S2_tnorm_adaptive",
            }[scenario.lower()]
            for norm in t_norms:
                for q_new_mode in q_new_modes:
                    for q_floor_new in q_floors_new:
                        for q_floor_existing in q_floors_existing:
                            for q_new_min in q_new_mins:
                                for new_thr in new_track_thresholds:
                                    for existing_thr in existing_track_thresholds:
                                        frame = adaptive_newgate_frame(
                                            features,
                                            scenario_name,
                                            norm,
                                            q_new_mode,
                                            q_floor_new,
                                            q_floor_existing,
                                            q_new_min,
                                            new_thr,
                                            existing_thr,
                                        )
                                        rows.append(_summary(frame, scenario_name, norm, None, total_frames, expected_gt))
                                        feature_frames.append(frame)
                                        threshold_rows.append(
                                            {
                                                "scenario": scenario_name,
                                                "t_norm": norm,
                                                "tau_Q": None,
                                                "q_new_mode": q_new_mode,
                                                "q_floor_new": q_floor_new,
                                                "q_floor_existing": q_floor_existing,
                                                "q_new_min": q_new_min,
                                                "new_track_threshold": new_thr,
                                                "existing_track_threshold": existing_thr,
                                                "selected": False,
                                            }
                                        )
        elif scenario.lower() == "s3_xai_newtrack_veto":
            for norm in t_norms:
                for mode in reweight_modes:
                    floors = q_floors if mode == "multiplicative_floor" else [None]
                    powers = gammas if mode == "power" else [None]
                    for q_floor in floors:
                        for gamma in powers:
                            for q_hard_min in q_hard_mins:
                                for new_thr in new_track_thresholds:
                                    for existing_thr in existing_track_thresholds:
                                        base_frame = soft_reweight_frame(features, "S3_xai_newtrack_veto", norm, False, mode, q_floor, gamma, q_hard_min, new_thr, existing_thr)
                                        for boundary_margin in xai_boundary_margins:
                                            for q_trigger in xai_q_triggers:
                                                for xai_veto_mode in xai_veto_modes:
                                                    mins = xai_mins if xai_veto_mode == "hard" else [None]
                                                    floors_x = xai_floors if xai_veto_mode == "soft" else [None]
                                                    for x_min in mins:
                                                        for xai_floor in floors_x:
                                                            frame, audit_part = apply_xai_newtrack_veto(
                                                                base_frame,
                                                                frame_index,
                                                                xai_method,
                                                                xai_max_per_frame,
                                                                boundary_margin,
                                                                q_trigger,
                                                                xai_veto_mode,
                                                                x_min,
                                                                xai_floor,
                                                            )
                                                            rows.append(_summary(frame, "S3_xai_newtrack_veto", norm, None, total_frames, expected_gt))
                                                            feature_frames.append(frame)
                                                            if len(audit_part):
                                                                xai_audit = pd.concat([xai_audit, audit_part], ignore_index=True)
                                                            threshold_rows.append(
                                                                {
                                                                    "scenario": "S3_xai_newtrack_veto",
                                                                    "t_norm": norm,
                                                                    "tau_Q": None,
                                                                    "reweight_mode": mode,
                                                                    "q_floor": q_floor,
                                                                    "gamma": gamma,
                                                                    "q_hard_min": q_hard_min,
                                                                    "new_track_threshold": new_thr,
                                                                    "existing_track_threshold": existing_thr,
                                                                    "xai_veto_mode": xai_veto_mode,
                                                                    "boundary_margin": boundary_margin,
                                                                    "q_xai_trigger": q_trigger,
                                                                    "x_min": x_min,
                                                                    "xai_floor": xai_floor,
                                                                    "selected": False,
                                                                }
                                                            )
        elif scenario.lower() in {"s3_safe_recovery", "s4_hybrid"}:
            rows.append(_not_implemented_summary("S3_safe_recovery" if scenario.lower() == "s3_safe_recovery" else "S4_hybrid", total_frames))
    audit = features.copy()
    audit["scenario"] = "S1_fgsm_base_features"
    audit["Q_i"] = audit["c_i"]
    audit["t_norm"] = None
    audit["tau_Q"] = None
    audit["accepted"] = True
    audit["confidence_original"] = audit["confidence"]
    audit["confidence_new"] = audit["confidence"]
    audit["q_floor"] = None
    audit["reweight_mode"] = None
    audit["reweight_stage"] = None
    audit["q_hard_min"] = 0.0
    audit["was_hard_rejected"] = False
    audit["new_track_allowed"] = True
    audit["existing_track_allowed"] = True
    audit["filter_action"] = "keep"
    audit.to_csv(out / "swarm_feature_audit.csv", index=False)
    matches.to_csv(out / "inter_agent_matching_audit.csv", index=False)
    if len(xai_audit):
        xai_audit.to_csv(out / "xai_feature_audit.csv", index=False)
        xai_summary.to_csv(out / "xai_summary.csv", index=False)
        xai_score_auc(xai_audit).to_csv(out / "xai_score_auc.csv", index=False)
    summary = pd.DataFrame(rows)
    threshold = pd.DataFrame(threshold_rows)
    selected = select_params(summary, threshold)
    stage = _stage_name(scenarios, split, out, selected_cfg, use_selected_params)
    selected.setdefault("source_stage", stage)
    if len(threshold) and selected.get("selection_status") == "selected":
        mask = (threshold["scenario"] == selected["selected_scenario"]) & (threshold["t_norm"] == selected["selected_t_norm"])
        for key, col in [
            ("selected_tau_Q", "tau_Q"),
            ("selected_reweight_mode", "reweight_mode"),
            ("selected_q_floor", "q_floor"),
            ("selected_gamma", "gamma"),
            ("selected_q_hard_min", "q_hard_min"),
            ("selected_new_track_threshold", "new_track_threshold"),
            ("selected_existing_track_threshold", "existing_track_threshold"),
            ("selected_q_new_mode", "q_new_mode"),
            ("selected_q_floor_new", "q_floor_new"),
            ("selected_q_floor_existing", "q_floor_existing"),
            ("selected_q_new_min", "q_new_min"),
            ("selected_xai_veto_mode", "xai_veto_mode"),
            ("selected_boundary_margin", "boundary_margin"),
            ("selected_q_xai_trigger", "q_xai_trigger"),
            ("selected_x_min", "x_min"),
            ("selected_xai_floor", "xai_floor"),
        ]:
            if key in selected and col in threshold:
                val = selected[key]
                mask &= threshold[col].isna() if val is None else threshold[col].fillna("__nan__").eq(val)
        threshold.loc[mask, "selected"] = True
    summary.to_csv(out / "summary_metrics.csv", index=False)
    summary.to_csv(out / "research_matrix.csv", index=False)
    threshold.to_csv(out / "threshold_selection.csv", index=False)
    summary[summary["scenario"].str.contains("tnorm|S4", case=False, na=False)].to_csv(out / "tnorm_comparison.csv", index=False)
    build_ablation(features, total_frames, expected_gt).to_csv(out / "ablation_summary.csv", index=False)
    error_type_breakdown(summary, attack_events).to_csv(out / "error_type_breakdown.csv", index=False)
    new_track_suppression(summary, features).to_csv(out / "new_track_suppression_summary.csv", index=False)
    new_track_gate_summary(summary, feature_frames).to_csv(out / "new_track_gate_summary.csv", index=False)
    candidate_score_distribution(feature_frames).to_csv(out / "candidate_score_distribution.csv", index=False)
    xai_newtrack_veto_summary(summary, feature_frames).to_csv(out / "xai_newtrack_veto_summary.csv", index=False)
    recall_preservation(summary).to_csv(out / "recall_preservation_summary.csv", index=False)
    summary.to_csv(out / "robustness_summary.csv", index=False)
    write_selected_params(out / "selected_params.yaml", selected)
    (out / "metadata.json").write_text(
        json.dumps(
            {
                "stage": stage,
                "dataset_type": "synthetic pseudo-swarm",
                "swarm_dataset_type": "synthetic pseudo-swarm",
                "source_dataset": "VisDrone2019-VID-val",
                "stress_transforms_enabled": "stress" in str(swarm_config),
                "limitation": "pseudo-swarm approximates multi-agent observations using transformed views of the same source frame",
                "split": split,
                "swarm_root": str(root),
                "selected_params_path": str(use_selected_params) if use_selected_params else None,
                "sequence_batch_size": sequence_batch_size,
                "max_temp_gb": max_temp_gb,
                "cleanup_temp": cleanup_temp,
                "num_agents": int(frame_index["agent_id"].nunique()),
                "num_sequences": int(frame_index["sequence_id"].nunique()),
                "num_synchronized_frames": int(frame_index.groupby(["sequence_id", "frame_id"])["agent_id"].nunique().eq(frame_index["agent_id"].nunique()).sum()),
                "xai_enabled": bool(len(xai_audit)),
                "xai_method_requested": xai_method,
                "xai_method_used": xai_method if len(xai_audit) else None,
                "gradcam_status": "not_attempted_or_failed" if xai_method == "eigencam" else "fallback_not_used",
                "xai_candidate_policy": {"c_low": 0.5, "sigma_k": 0.3, "sigma_s": 0.3, "tau_pre": 0.3, "M_xai": xai_max_per_frame},
                "x_i_formula": "top-5-percent CAM pixels inside bbox share",
                "x_i_selected_method": "x_top5_inside",
                "xai_score_warning": "selected after smoke; calibration evaluates whether it generalizes",
                "xai_stage": "v2.5_calibration" if len(xai_audit) else None,
                "xai_not_yet_final_filter": bool(len(xai_audit)),
                "x_i_neutral_when_not_called": 1.0,
                "s_missing_policy": "neutral",
                "attack_model_type": "controlled detection-level pseudo attack" if attack_cfg else None,
                "attack_model_name": attack_cfg.get("attack_model", {}).get("name") if attack_cfg else None,
                "pseudo_attack_config": str(pseudo_attack_config) if pseudo_attack_config else None,
                "reweight_stage": "post_nms_pre_tracker",
                "soft_reweighting_enabled": any("soft" in s.lower() for s in scenarios),
                "s3_safe_recovery_status": "not implemented for swarm_vid pseudo-swarm in v2.7",
                "s4_hybrid_status": "soft confidence proxy; full v1.8 recovery is not implemented for swarm_vid pseudo-swarm",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    disk_after = disk_report(Path.cwd())
    (out / "disk_usage_report.txt").write_text(
        "\n".join(
            [
                "df_before:",
                disk_before,
                "df_after:",
                disk_after,
                f"swarm_root_du: {du(root)}",
                f"output_dir_du: {du(out)}",
                f"materialization_mode: {materialization_mode(frame_index)}",
                f"max_temp_gb: {max_temp_gb}",
                f"cleanup_temp_enabled: {cleanup_temp}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"output: {out}")
    print(f"mean_s_i: {features['s_i'].mean():.4f}")
    s2_rows = summary[summary["scenario"] == "S2_tnorm_no_xai"]
    print(f"s2_rejected_best: {int(s2_rows['num_rejected'].min()) if len(s2_rows) else 0}")
    if len(xai_summary):
        print(f"num_xai_called: {int(xai_summary.iloc[0]['num_xai_called'])}")
        print(f"mean_x_i: {float(xai_summary.iloc[0]['mean_x_i']):.4f}")
    print(f"selection_status: {selected['selection_status']}")
    print(f"holdout_allowed: {selected['holdout_allowed']}")


def apply_xai_smoke(features: pd.DataFrame, frame_index: pd.DataFrame, method: str, max_per_frame: int, max_frames: int | None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    out = features.copy()
    out["Q_pre"] = out[["c_i", "k_i", "s_i"]].min(axis=1)
    out["risk"] = (1.0 - out["c_i"]) + (1.0 - out["k_i"]) + (1.0 - out["s_i"])
    image_paths = frame_index.set_index(["sequence_id", "frame_id", "agent_id"])["image_path"].to_dict()
    audit_rows = []
    frame_count = 0
    cam_cache: dict[str, tuple[np.ndarray | None, float]] = {}
    ordered = out.sort_values(["sequence_id", "frame_id", "agent_id", "risk"], ascending=[True, True, True, False])
    for frame_key, group in ordered.groupby(["sequence_id", "frame_id"], sort=False):
        if max_frames is not None and frame_count >= max_frames:
            break
        candidates = select_xai_candidates(group, max_per_frame=max_per_frame)
        if candidates.empty:
            candidates = group.sort_values("risk", ascending=False).head(1)
        if candidates.empty:
            continue
        for rank, (idx, row) in enumerate(candidates.iterrows(), start=1):
            image_key = (row.sequence_id, row.frame_id, row.agent_id)
            image_path = image_paths[image_key]
            if image_path not in cam_cache:
                cam_cache[image_path] = compute_cam(image_path, method=method)
            cam, latency = cam_cache[image_path]
            scores = compute_xai_score(cam, (row.x1, row.y1, row.x2, row.y2))
            available = scores["x_i_selected"] is not None
            value = float(scores["x_i_selected"]) if available else 1.0
            out.loc[idx, "x_i"] = value
            out.loc[idx, "x_i_available"] = available
            out.loc[idx, "xai_called"] = True
            out.loc[idx, "xai_method"] = method
            audit_row = {
                    "sequence_id": row.sequence_id,
                    "frame_id": row.frame_id,
                    "agent_id": row.agent_id,
                    "det_id": row.det_id,
                    "track_id": row.track_id,
                    "scenario": "S3_tnorm_xai",
                    "model_name": "pseudo_yolo",
                    "eps": row.eps,
                    "xai_method": method,
                    "xai_called": True,
                    "xai_available": available,
                    "xai_rank_in_frame": rank,
                    "confidence": row.confidence,
                    "c_i": row.c_i,
                    "k_i": row.k_i,
                    "s_i": row.s_i,
                    "Q_pre": row.Q_pre,
                    "risk": row.risk,
                    "bbox_x1": row.x1,
                    "bbox_y1": row.y1,
                    "bbox_x2": row.x2,
                    "bbox_y2": row.y2,
                    "xai_latency_ms": latency,
                    "is_TP": row.track_id != -1,
                    "is_FP": row.track_id == -1,
                    "matched_gt_id": row.track_id if row.track_id != -1 else None,
                    "matched_gt_iou": 1.0 if row.track_id != -1 else 0.0,
                }
            audit_row.update(scores)
            audit_rows.append(audit_row)
        frame_count += 1
    audit = pd.DataFrame(audit_rows)
    called = out[out["xai_called"] == True]
    available = called[called["x_i_available"] == True]
    summary = pd.DataFrame(
        [
            {
                "xai_method": method,
                "num_frames": frame_count,
                "num_detections": len(out),
                "num_xai_candidates": len(called),
                "num_xai_called": len(called),
                "xai_available_rate": len(available) / max(1, len(called)),
                "mean_x_i": float(available["x_i"].mean()) if len(available) else 0.0,
                "median_x_i": float(available["x_i"].median()) if len(available) else 0.0,
                "x_i_min": float(available["x_i"].min()) if len(available) else 0.0,
                "x_i_max": float(available["x_i"].max()) if len(available) else 0.0,
                "x_raw_mean": float(audit["x_raw"].dropna().mean()) if len(audit) else 0.0,
                "x_density_norm_cap5_mean": float(audit["x_i_density_norm_cap5"].dropna().mean()) if len(audit) else 0.0,
                "x_top5_inside_mean": float(audit["x_top5_inside"].dropna().mean()) if len(audit) else 0.0,
                "x_peak_inside_rate": float(audit["x_peak_inside"].dropna().mean()) if len(audit) else 0.0,
                "share_x_i_lt_0_3": float((available["x_i"] < 0.3).mean()) if len(available) else 0.0,
                "share_x_i_lt_0_5": float((available["x_i"] < 0.5).mean()) if len(available) else 0.0,
                "mean_xai_latency_ms": float(audit["xai_latency_ms"].mean()) if len(audit) else 0.0,
                "xai_calls_per_frame": len(called) / max(1, frame_count),
            }
        ]
    )
    return out, audit, summary


def apply_xai_newtrack_veto(
    base_frame: pd.DataFrame,
    frame_index: pd.DataFrame,
    method: str,
    max_per_frame: int,
    boundary_margin: float,
    q_trigger: float,
    veto_mode: str,
    x_min: float | None,
    xai_floor: float | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = base_frame.copy()
    frame["xai_called"] = False
    frame["xai_method"] = None
    frame["xai_veto_mode"] = veto_mode
    frame["xai_max_per_frame"] = max_per_frame
    frame["boundary_margin"] = boundary_margin
    frame["q_xai_trigger"] = q_trigger
    frame["x_min"] = x_min
    frame["xai_floor"] = xai_floor
    frame["confidence_xai"] = frame["confidence_new"]
    is_new = frame["track_status"].eq("new_candidate")
    near_boundary = (frame["confidence_new"] - frame["new_track_threshold"]).abs() <= boundary_margin
    low_q = frame["Q_i"] <= q_trigger
    candidates_mask = is_new & (near_boundary | low_q)
    image_paths = frame_index.set_index(["sequence_id", "frame_id", "agent_id"])["image_path"].to_dict()
    audit_rows = []
    cam_cache: dict[str, tuple[np.ndarray | None, float]] = {}
    for _, group in frame[candidates_mask].groupby(["sequence_id", "frame_id"], sort=False):
        candidates = group.sort_values(["confidence_new", "Q_i"], ascending=[False, True]).head(max_per_frame)
        for rank, (idx, row) in enumerate(candidates.iterrows(), start=1):
            image_key = (row.sequence_id, row.frame_id, row.agent_id)
            image_path = image_paths.get(image_key, "")
            if image_path not in cam_cache:
                cam_cache[image_path] = compute_cam(image_path, method=method)
            cam, latency = cam_cache[image_path]
            scores = compute_xai_score(cam, (row.x1, row.y1, row.x2, row.y2))
            x_value = float(scores["x_i_selected"]) if scores["x_i_selected"] is not None else 1.0
            frame.loc[idx, "x_i"] = x_value
            frame.loc[idx, "xai_called"] = True
            frame.loc[idx, "xai_method"] = method
            if veto_mode == "hard":
                if x_min is not None and x_value < x_min:
                    frame.loc[idx, "accepted"] = False
                    frame.loc[idx, "filter_action"] = "xai_hard_veto_new_track"
            elif veto_mode == "soft":
                floor = float(xai_floor if xai_floor is not None else 0.8)
                conf_xai = float(row.confidence_new) * (floor + (1.0 - floor) * x_value)
                frame.loc[idx, "confidence_xai"] = conf_xai
                frame.loc[idx, "confidence"] = conf_xai
                allowed = conf_xai >= float(row.new_track_threshold)
                frame.loc[idx, "accepted"] = bool(allowed)
                if not allowed:
                    frame.loc[idx, "filter_action"] = "xai_soft_veto_new_track"
            else:
                raise ValueError(f"Unknown XAI veto mode: {veto_mode}")
            audit_row = {
                "sequence_id": row.sequence_id,
                "frame_id": row.frame_id,
                "agent_id": row.agent_id,
                "det_id": row.det_id,
                "track_id": row.track_id,
                "scenario": "S3_xai_newtrack_veto",
                "xai_method": method,
                "xai_called": True,
                "xai_available": scores["x_i_selected"] is not None,
                "xai_rank_in_frame": rank,
                "confidence": row.confidence_original,
                "confidence_new": row.confidence_new,
                "Q_i": row.Q_i,
                "boundary_margin": boundary_margin,
                "q_xai_trigger": q_trigger,
                "xai_veto_mode": veto_mode,
                "x_min": x_min,
                "xai_floor": xai_floor,
                "x_i": x_value,
                "xai_latency_ms": latency,
                "is_TP": bool(row.eval_is_tp),
                "is_FP": not bool(row.eval_is_tp),
            }
            audit_row.update(scores)
            audit_rows.append(audit_row)
    return frame, pd.DataFrame(audit_rows)


def prepare_features(features: pd.DataFrame, eps_values: list[float]) -> pd.DataFrame:
    out = features.copy()
    if "eval_is_tp" not in out:
        out["eval_is_tp"] = out["track_id"] != -1
    out["scenario"] = "S1_fgsm"
    out["model_name"] = "pseudo_yolo"
    out["eps"] = eps_values[0] if eps_values else 0.0
    out = add_kinematics(out)
    out["k_i"] = out["k_i_selected"]
    out["x_i"] = 1.0
    out["x_i_available"] = False
    out["xai_called"] = False
    out["xai_method"] = None
    out = add_track_status(out)
    return out


def add_track_status(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.sort_values(["sequence_id", "agent_id", "track_id", "frame_id", "det_id"]).copy()
    out["track_age"] = out.groupby(["sequence_id", "agent_id", "track_id"], sort=False).cumcount() + 1
    out["track_status"] = np.where(out["track_id"].eq(-1) | out["track_age"].eq(1), "new_candidate", "confirmed_existing")
    return out


def xai_score_auc(audit: pd.DataFrame) -> pd.DataFrame:
    scores = {
        "1-x_raw": "x_raw",
        "1-x_i_density_norm_cap2": "x_i_density_norm_cap2",
        "1-x_i_density_norm_cap5": "x_i_density_norm_cap5",
        "1-x_i_density_norm_cap10": "x_i_density_norm_cap10",
        "1-x_top1_inside": "x_top1_inside",
        "1-x_top5_inside": "x_top5_inside",
        "1-x_top10_inside": "x_top10_inside",
        "1-x_peak_inside": "x_peak_inside",
    }
    rows = []
    for name, col in scores.items():
        data = audit[[col, "is_FP"]].dropna()
        y = data["is_FP"].astype(int)
        risk = 1.0 - data[col].astype(float)
        rows.append(
            {
                "score_name": name,
                "num_samples": len(data),
                "num_TP": int((y == 0).sum()),
                "num_FP": int((y == 1).sum()),
                "roc_auc": _roc_auc(y, risk),
                "average_precision": _average_precision(y, risk),
                "precision_at_1pct_rejection": _precision_at(y, risk, 0.01),
                "precision_at_5pct_rejection": _precision_at(y, risk, 0.05),
                "precision_at_10pct_rejection": _precision_at(y, risk, 0.10),
                "best_threshold": float(risk.median()) if len(risk) else 0.0,
                "best_FP_detection_F1": _best_f1(y, risk),
            }
        )
    return pd.DataFrame(rows)


def _roc_auc(y: pd.Series, score: pd.Series) -> float:
    pos = score[y == 1].to_numpy()
    neg = score[y == 0].to_numpy()
    if len(pos) == 0 or len(neg) == 0:
        return 0.5
    return float(((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean()))


def _average_precision(y: pd.Series, score: pd.Series) -> float:
    if int(y.sum()) == 0:
        return 0.0
    order = score.sort_values(ascending=False).index
    yy = y.loc[order].to_numpy()
    precision = np.cumsum(yy) / np.arange(1, len(yy) + 1)
    return float((precision * yy).sum() / max(1, yy.sum()))


def _precision_at(y: pd.Series, score: pd.Series, frac: float) -> float:
    n = max(1, int(round(len(score) * frac)))
    idx = score.sort_values(ascending=False).head(n).index
    return float(y.loc[idx].mean()) if len(idx) else 0.0


def _best_f1(y: pd.Series, score: pd.Series) -> float:
    best = 0.0
    for threshold in score.quantile([0.1, 0.2, 0.3, 0.5, 0.7, 0.9]).unique():
        pred = score >= threshold
        tp = int(((pred == 1) & (y == 1)).sum())
        fp = int(((pred == 1) & (y == 0)).sum())
        fn = int(((pred == 0) & (y == 1)).sum())
        best = max(best, 2 * tp / max(1, 2 * tp + fp + fn))
    return float(best)


def add_kinematics(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.sort_values(["agent_id", "track_id", "sequence_id", "frame_id"]).copy()
    out[["k_center", "k_iou", "k_combined"]] = 1.0
    for _, group in out.groupby(["agent_id", "track_id"], sort=False):
        prev_box = None
        prev_center = None
        prev_velocity = (0.0, 0.0)
        for idx, row in group.iterrows():
            box = (row.x1, row.y1, row.x2, row.y2)
            center = ((row.x1 + row.x2) / 2.0, (row.y1 + row.y2) / 2.0)
            if prev_box is None:
                k_center, k_iou = 1.0, 1.0
            else:
                pred_center = (prev_center[0] + prev_velocity[0], prev_center[1] + prev_velocity[1])
                pred_box = _shift_box(prev_box, prev_velocity)
                dist = ((center[0] - pred_center[0]) ** 2 + (center[1] - pred_center[1]) ** 2) ** 0.5
                diag = max(1.0, ((prev_box[2] - prev_box[0]) ** 2 + (prev_box[3] - prev_box[1]) ** 2) ** 0.5)
                k_center = float(np.exp(-dist / diag))
                k_iou = _iou(box, pred_box)
            k_combined = float((k_center * max(k_iou, 1e-6)) ** 0.5)
            out.loc[idx, ["k_center", "k_iou", "k_combined"]] = [k_center, k_iou, k_combined]
            prev_velocity = (center[0] - prev_center[0], center[1] - prev_center[1]) if prev_center else (0.0, 0.0)
            prev_center, prev_box = center, box
    out["k_i_selected"] = out["k_combined"]
    out["k_i_selected_method"] = "k_combined"
    return out


def _shift_box(box: tuple[float, float, float, float], velocity: tuple[float, float]) -> tuple[float, float, float, float]:
    return (box[0] + velocity[0], box[1] + velocity[1], box[2] + velocity[0], box[3] + velocity[1])


def _iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return inter / max(1e-9, area_a + area_b - inter)


def _pseudo_detections(gt: pd.DataFrame, frame_index: pd.DataFrame) -> pd.DataFrame:
    transforms = frame_index.set_index(["sequence_id", "frame_id", "agent_id"])["transform_from_reference_matrix"].to_dict()
    jitters = frame_index.set_index(["sequence_id", "frame_id", "agent_id"])["bbox_jitter"].to_dict()
    rows = []
    det_id = 0
    for frame_row in frame_index[["sequence_id", "frame_id", "agent_id"]].drop_duplicates().itertuples(index=False):
        frame_gt = gt[(gt.sequence_id == frame_row.sequence_id) & (gt.frame_id == frame_row.frame_id)]
        matrix = transforms[(frame_row.sequence_id, frame_row.frame_id, frame_row.agent_id)]
        jitter = float(jitters.get((frame_row.sequence_id, frame_row.frame_id, frame_row.agent_id), 0.0))
        for g in frame_gt.itertuples(index=False):
            box = transform_bbox((g.x1, g.y1, g.x2, g.y2), matrix)
            box = jitter_bbox(box, jitter, frame_row.sequence_id, frame_row.frame_id, frame_row.agent_id, g.gt_track_id)
            rows.append({"det_id": det_id, "sequence_id": frame_row.sequence_id, "frame_id": frame_row.frame_id, "agent_id": frame_row.agent_id, "track_id": g.gt_track_id, "class_id": g.class_id, "class_name": VISDRONE_CLASSES.get(int(g.class_id), "unknown"), "confidence": 0.9, "x1": box[0], "y1": box[1], "x2": box[2], "y2": box[3]})
            det_id += 1
        if frame_row.agent_id == "agent_2" and int(frame_row.frame_id) % 10 == 0:
            rows.append({"det_id": det_id, "sequence_id": frame_row.sequence_id, "frame_id": frame_row.frame_id, "agent_id": frame_row.agent_id, "track_id": -1, "class_id": 4, "class_name": "car", "confidence": 0.25, "x1": 8.0, "y1": 8.0, "x2": 42.0, "y2": 38.0})
            det_id += 1
    return pd.DataFrame(rows)


def _q(frame: pd.DataFrame, norm: str) -> pd.Series:
    if norm in {"min", "T_min"}:
        return frame[["c_i", "k_i", "s_i", "x_i"]].min(axis=1)
    if norm in {"prod", "T_prod"}:
        return frame["c_i"] * frame["k_i"] * frame["s_i"] * frame["x_i"]
    if norm.lower() in {"lukasiewicz", "t_lukasiewicz"}:
        return (frame["c_i"] + frame["k_i"] + frame["s_i"] + frame["x_i"] - 3.0).clip(lower=0.0)
    raise ValueError(f"Unknown t-norm: {norm}")


def soft_reweight_frame(
    features: pd.DataFrame,
    scenario: str,
    norm: str,
    use_xai: bool,
    mode: str,
    q_floor: float | None,
    gamma: float | None,
    q_hard_min: float,
    new_track_threshold: float,
    existing_track_threshold: float,
) -> pd.DataFrame:
    frame = features.copy()
    frame["scenario"] = scenario
    if not use_xai:
        frame["x_i"] = 1.0
    frame["t_norm"] = norm
    frame["Q_i"] = _q(frame, norm)
    frame["confidence_original"] = frame["confidence"].astype(float)
    if mode == "multiplicative_floor":
        floor = float(q_floor if q_floor is not None else 0.7)
        weight = floor + (1.0 - floor) * frame["Q_i"]
    elif mode == "power":
        g = float(gamma if gamma is not None else 0.5)
        weight = frame["Q_i"].clip(lower=1e-6) ** g
    else:
        raise ValueError(f"Unknown reweight mode: {mode}")
    frame["confidence_new"] = frame["confidence_original"] * weight
    frame["confidence"] = frame["confidence_new"]
    frame["reweight_mode"] = mode
    frame["q_floor"] = q_floor
    frame["gamma"] = gamma
    frame["q_hard_min"] = q_hard_min
    frame["reweight_stage"] = "post_nms_pre_tracker"
    frame["was_hard_rejected"] = frame["Q_i"] < q_hard_min
    frame["new_track_threshold"] = new_track_threshold
    frame["existing_track_threshold"] = existing_track_threshold
    frame["new_track_allowed"] = frame["confidence_new"] >= new_track_threshold
    frame["existing_track_allowed"] = frame["confidence_new"] >= existing_track_threshold
    is_new = frame["track_status"].eq("new_candidate")
    frame["accepted"] = (~frame["was_hard_rejected"]) & np.where(is_new, frame["new_track_allowed"], frame["existing_track_allowed"])
    frame["filter_action"] = np.where(frame["was_hard_rejected"], "hard_reject_min_q", "soft_reweight")
    frame["tau_Q"] = None
    return frame


def q_new_score(frame: pd.DataFrame, mode: str) -> pd.Series:
    k = frame["k_i"].clip(lower=1e-6)
    s = frame["s_i"].clip(lower=1e-6)
    if mode == "min_ks":
        return pd.concat([k, s], axis=1).min(axis=1)
    if mode == "geom_ks":
        return (k * s) ** 0.5
    if mode == "hmean_ks":
        return 2.0 / (1.0 / k + 1.0 / s)
    raise ValueError(f"Unknown q_new_mode: {mode}")


def adaptive_newgate_frame(
    features: pd.DataFrame,
    scenario: str,
    norm: str,
    q_new_mode: str,
    q_floor_new: float,
    q_floor_existing: float,
    q_new_min: float,
    new_track_threshold: float,
    existing_track_threshold: float,
) -> pd.DataFrame:
    frame = features.copy()
    frame["scenario"] = scenario
    frame["x_i"] = 1.0
    frame["t_norm"] = norm
    frame["Q_existing"] = _q(frame, norm)
    frame["Q_new"] = q_new_score(frame, q_new_mode)
    is_new = frame["track_status"].eq("new_candidate")
    frame["Q_i"] = np.where(scenario == "S2_tnorm_adaptive", np.where(is_new, frame["Q_new"], frame["Q_existing"]), frame["Q_existing"])
    frame["confidence_original"] = frame["confidence"].astype(float)
    weight_new = q_floor_new + (1.0 - q_floor_new) * frame["Q_new"]
    weight_existing = q_floor_existing + (1.0 - q_floor_existing) * frame["Q_existing"]
    if scenario == "S2_tnorm_soft_v2":
        weight_new = q_floor_new + (1.0 - q_floor_new) * frame["Q_existing"]
    frame["confidence_new"] = frame["confidence_original"] * np.where(is_new, weight_new, weight_existing)
    frame["confidence"] = frame["confidence_new"]
    frame["q_new_mode"] = q_new_mode
    frame["q_floor_new"] = q_floor_new
    frame["q_floor_existing"] = q_floor_existing
    frame["q_new_min"] = q_new_min
    frame["new_track_threshold"] = new_track_threshold
    frame["existing_track_threshold"] = existing_track_threshold
    frame["reweight_mode"] = "adaptive_newgate"
    frame["reweight_stage"] = "post_nms_pre_tracker"
    frame["q_hard_min"] = 0.0
    frame["was_hard_rejected"] = False
    frame["new_track_allowed"] = (frame["confidence_new"] >= new_track_threshold) & (
        (frame["Q_new"] >= q_new_min) if scenario in {"S2_tnorm_newgate", "S2_tnorm_adaptive"} else True
    )
    frame["existing_track_allowed"] = frame["confidence_new"] >= existing_track_threshold
    frame["accepted"] = np.where(is_new, frame["new_track_allowed"], frame["existing_track_allowed"])
    frame["filter_action"] = np.where(is_new & ~frame["new_track_allowed"], "suppress_new_track", "soft_reweight")
    frame["tau_Q"] = None
    return frame


def _summary(frame: pd.DataFrame, scenario: str, norm: str | None, tau: float | None, num_frames: int, expected_gt: int | None = None) -> dict:
    total_gt = int(expected_gt if expected_gt is not None else (frame["track_id"] != -1).sum())
    accepted = frame["accepted"].astype(bool)
    eval_tp = frame["eval_is_tp"].astype(bool) if "eval_is_tp" in frame else frame["track_id"] != -1
    tp = int((accepted & eval_tp).sum())
    fp = int((accepted & ~eval_tp).sum())
    fn = max(0, total_gt - tp)
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    track_breaks = _track_breaks(frame)
    idsw = 0
    mota = 1.0 - (fp + fn + idsw) / max(1, total_gt)
    failures = (fp + fn + idsw + track_breaks) / max(1, num_frames) * 100.0
    return {
        "scenario": scenario,
        "model_name": "yolov8n",
        "eps": float(frame["eps"].iloc[0]) if "eps" in frame and len(frame) else 0.0,
        "class_group": "all",
        "t_norm": norm,
        "tau_Q": tau,
        "x_i_selected_method": "x_top5_inside" if scenario == "S3_tnorm_xai" else None,
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "precision": precision,
        "recall": recall,
        "F1": f1,
        "IDF1": f1,
        "IDSW": idsw,
        "track_breaks": track_breaks,
        "MOTA": mota,
        "ASR_track": failures / 100.0,
        "failures_per_100_frames": failures,
        "FP_per_100_frames": fp / max(1, num_frames) * 100.0,
        "FN_per_100_frames": fn / max(1, num_frames) * 100.0,
        "latency_ms": 0.0,
        "xai_latency_ms": 0.0,
        "reweight_mode": first_value(frame, "reweight_mode"),
        "q_floor": first_value(frame, "q_floor"),
        "q_floor_new": first_value(frame, "q_floor_new"),
        "q_floor_existing": first_value(frame, "q_floor_existing"),
        "q_new_mode": first_value(frame, "q_new_mode"),
        "q_new_min": first_value(frame, "q_new_min"),
        "xai_veto_mode": first_value(frame, "xai_veto_mode"),
        "boundary_margin": first_value(frame, "boundary_margin"),
        "q_xai_trigger": first_value(frame, "q_xai_trigger"),
        "x_min": first_value(frame, "x_min"),
        "xai_floor": first_value(frame, "xai_floor"),
        "gamma": first_value(frame, "gamma"),
        "beta": first_value(frame, "beta"),
        "q_hard_min": first_value(frame, "q_hard_min"),
        "new_track_threshold": first_value(frame, "new_track_threshold"),
        "existing_track_threshold": first_value(frame, "existing_track_threshold"),
        "confidence_mean_before": float(frame["confidence_original"].mean()) if "confidence_original" in frame else float(frame["confidence"].mean()),
        "confidence_mean_after": float(frame["confidence_new"].mean()) if "confidence_new" in frame else float(frame["confidence"].mean()),
        "confidence_delta_mean": float((frame["confidence_new"] - frame["confidence_original"]).mean()) if {"confidence_new", "confidence_original"}.issubset(frame.columns) else 0.0,
        "num_detections_before_filter": len(frame),
        "num_detections_after_filter": int(frame["accepted"].sum()),
        "num_rejected": int((~frame["accepted"]).sum()),
        "rejection_rate": float((~frame["accepted"]).mean()),
        "mean_c_i": float(frame["c_i"].mean()),
        "mean_k_i": float(frame["k_i"].mean()),
        "mean_s_i": float(frame["s_i"].mean()),
        "mean_x_i": float(frame["x_i"].mean()),
        "share_k_i_low": float((frame["k_i"] < 0.7).mean()),
        "share_s_i_low": float((frame["s_i"] < 0.3).mean()),
        "share_x_i_low": float((frame["x_i"] < 0.3).mean()),
        "mean_Q_i": float(frame["Q_i"].mean()),
        "xai_calls_total": int(frame["xai_called"].sum()) if "xai_called" in frame else 0,
        "xai_calls_per_frame": float(frame["xai_called"].sum() / max(1, frame[["sequence_id", "frame_id", "agent_id"]].drop_duplicates().shape[0])) if "xai_called" in frame else 0.0,
        "implementation_status": "ok",
    }


def first_value(frame: pd.DataFrame, col: str):
    if col not in frame or frame.empty:
        return None
    val = frame[col].iloc[0]
    if pd.isna(val):
        return None
    return val.item() if hasattr(val, "item") else val


def disk_report(path: Path) -> str:
    usage = shutil.disk_usage(path)
    return (
        f"path={path} "
        f"total_gb={usage.total / (1024**3):.1f} "
        f"used_gb={usage.used / (1024**3):.1f} "
        f"free_gb={usage.free / (1024**3):.1f} "
        f"used_pct={usage.used / max(1, usage.total) * 100:.1f}"
    )


def du(path: Path) -> str:
    if not path.exists():
        return "missing"
    total = sum(p.stat().st_size for p in path.rglob("*") if p.is_file())
    return f"{total / (1024**3):.3f}G"


def materialization_mode(frame_index: pd.DataFrame) -> str:
    if "is_materialized" not in frame_index:
        return "materialized"
    materialized = frame_index["is_materialized"].astype(str).str.lower().isin({"true", "1", "yes"})
    return "materialized" if bool(materialized.all()) else "manifest_only"


def _stage_name(
    scenarios: list[str],
    split: str,
    out: Path,
    selected_cfg: dict,
    use_selected_params: str | Path | None,
) -> str:
    scenario_text = " ".join(s.lower() for s in scenarios)
    path_text = f"{out} {use_selected_params or ''}".lower()
    selected_stage = str(selected_cfg.get("source_stage", "")).lower()
    if "xai_newtrack" in scenario_text or "v3_1" in path_text:
        return "v3.1_xai_newtrack_veto"
    if "v3_newgate" in path_text or "v3.0" in selected_stage:
        return "v3.0_newtrack_gate_holdout" if split == "holdout" else "v3.0_newtrack_gate_calibration"
    if any(key in scenario_text for key in ["newgate", "adaptive", "soft_v2"]):
        return "v3.0_newtrack_gate_holdout" if split == "holdout" else "v3.0_newtrack_gate_calibration"
    if split == "holdout":
        return "v2.9_disk_safe_holdout"
    if "soft" in scenario_text:
        return "v2.7_soft_reweighting"
    return "v2.6_calibration"


def _not_implemented_summary(scenario: str, num_frames: int) -> dict:
    return {
        "scenario": scenario,
        "model_name": "yolov8n",
        "eps": 0.008,
        "class_group": "all",
        "t_norm": None,
        "tau_Q": None,
        "x_i_selected_method": None,
        "TP": None,
        "FP": None,
        "FN": None,
        "precision": None,
        "recall": None,
        "F1": None,
        "IDF1": None,
        "IDSW": None,
        "track_breaks": None,
        "MOTA": None,
        "ASR_track": None,
        "failures_per_100_frames": None,
        "FP_per_100_frames": None,
        "FN_per_100_frames": None,
        "latency_ms": None,
        "xai_calls_per_frame": None,
        "xai_latency_ms": None,
        "num_detections_before_filter": 0,
        "num_detections_after_filter": 0,
        "num_rejected": 0,
        "rejection_rate": None,
        "mean_c_i": None,
        "mean_k_i": None,
        "mean_s_i": None,
        "mean_x_i": None,
        "share_k_i_low": None,
        "share_s_i_low": None,
        "share_x_i_low": None,
        "mean_Q_i": None,
        "xai_calls_total": 0,
        "implementation_status": "not_implemented_for_swarm_vid_pseudo",
    }


def _track_breaks(frame: pd.DataFrame) -> int:
    accepted = frame[(frame["accepted"] == True) & (frame["track_id"] != -1)]
    if accepted.empty:
        return 0
    cols = ["sequence_id", "agent_id", "track_id", "frame_id"]
    uniq = accepted[cols].drop_duplicates().sort_values(cols)
    prev = uniq.groupby(["sequence_id", "agent_id", "track_id"], sort=False)["frame_id"].shift(1)
    return int(((uniq["frame_id"] - prev) > 1).sum())


def select_params(summary: pd.DataFrame, threshold: pd.DataFrame) -> dict:
    xai_candidates = summary[summary["scenario"] == "S3_xai_newtrack_veto"].copy()
    if not xai_candidates.empty:
        s2 = summary[summary["scenario"] == "S2_tnorm_soft"].sort_values(["F1", "IDF1"], ascending=False)
        if s2.empty:
            return {"selection_status": "not_selected", "holdout_allowed": False, "reason": "missing_s2_baseline"}
        base = s2.iloc[0]
        valid = xai_candidates[
            (xai_candidates["FP"] < base["FP"])
            & (xai_candidates["FN"] <= base["FN"] * 1.005)
            & (xai_candidates["F1"] >= base["F1"])
        ].copy()
        if valid.empty:
            best_diag = xai_candidates.assign(
                score=(base["FP"] - xai_candidates["FP"]) / max(1, base["FP"]) + (xai_candidates["F1"] - base["F1"])
            ).sort_values("score", ascending=False).iloc[0]
            return {
                "selection_status": "not_selected",
                "holdout_allowed": False,
                "reason": "xai_veto_does_not_improve_s2_soft",
                "baseline_s2_F1": float(base["F1"]),
                "best_xai_FP_delta_vs_S2": int(best_diag["FP"] - base["FP"]),
                "best_xai_FN_delta_vs_S2": int(best_diag["FN"] - base["FN"]),
                "best_xai_F1_delta_vs_S2": float(best_diag["F1"] - base["F1"]),
            }
        valid["score"] = (base["FP"] - valid["FP"]) / max(1, base["FP"]) + (valid["F1"] - base["F1"])
        best = valid.sort_values("score", ascending=False).iloc[0]
        return {
            "selection_status": "selected",
            "holdout_allowed": True,
            "selected_scenario": "S3_xai_newtrack_veto",
            "selected_t_norm": first_value(best.to_frame().T, "t_norm"),
            "selected_reweight_mode": first_value(best.to_frame().T, "reweight_mode"),
            "selected_q_floor": first_value(best.to_frame().T, "q_floor"),
            "selected_q_hard_min": first_value(best.to_frame().T, "q_hard_min"),
            "selected_new_track_threshold": first_value(best.to_frame().T, "new_track_threshold"),
            "selected_existing_track_threshold": first_value(best.to_frame().T, "existing_track_threshold"),
            "selected_xai_veto_mode": first_value(best.to_frame().T, "xai_veto_mode"),
            "selected_boundary_margin": first_value(best.to_frame().T, "boundary_margin"),
            "selected_q_xai_trigger": first_value(best.to_frame().T, "q_xai_trigger"),
            "selected_x_min": first_value(best.to_frame().T, "x_min"),
            "selected_xai_floor": first_value(best.to_frame().T, "xai_floor"),
            "FP_delta_vs_S2": int(best["FP"] - base["FP"]),
            "FN_delta_vs_S2": int(best["FN"] - base["FN"]),
            "F1_delta_vs_S2": float(best["F1"] - base["F1"]),
            "reason": "xai_newtrack_veto_improves_s2_soft",
        }
    candidates = summary[
        summary["scenario"].isin(
            [
                "S2_tnorm_no_xai",
                "S3_tnorm_xai",
                "S2_tnorm_hard",
                "S3_tnorm_xai_hard",
                "S2_tnorm_soft",
                "S3_tnorm_xai_soft",
                "S4_soft_recovery",
                "S4_hybrid",
                "S2_tnorm_soft_v2",
                "S2_tnorm_newgate",
                "S2_tnorm_adaptive",
            ]
        )
    ].copy()
    candidates = candidates[candidates["implementation_status"].fillna("ok") == "ok"]
    naive = summary[summary["scenario"] == "S_naive"].copy()
    if candidates.empty or naive.empty:
        return {"selection_status": "not_selected", "holdout_allowed": False, "reason": "missing_candidates"}
    naive_best = naive.sort_values(["IDF1", "F1"], ascending=False).iloc[0]
    has_v3 = candidates["scenario"].isin(["S2_tnorm_soft_v2", "S2_tnorm_newgate", "S2_tnorm_adaptive"]).any()
    fn_limit = float(naive_best["FN"]) * (1.01 if has_v3 else 1.05)
    f1_floor = float(naive_best["F1"]) - (0.0005 if has_v3 else 0.005)
    break_limit = float(naive_best["track_breaks"]) * (1.01 if has_v3 else 1.0)
    valid = candidates[
        (candidates["F1"] >= f1_floor)
        & (candidates["IDSW"] <= naive_best["IDSW"])
        & (candidates["track_breaks"] <= break_limit)
        & (candidates["FP"] < naive_best["FP"])
        & (candidates["FN"] <= fn_limit)
    ].copy()
    if valid.empty:
        return {
            "selection_status": "not_selected",
            "holdout_allowed": False,
            "reason": "no_candidate_beats_s_naive",
            "baseline_s_naive_IDF1": float(naive_best["IDF1"]),
            "baseline_s_naive_F1": float(naive_best["F1"]),
        }
    if has_v3:
        valid["score"] = (
            0.50 * ((naive_best["FP"] - valid["FP"]) / max(1, naive_best["FP"]))
            - 0.35 * ((valid["FN"] - naive_best["FN"]).clip(lower=0) / max(1, naive_best["FN"]))
            + 0.15 * (valid["F1"] - naive_best["F1"])
        )
    else:
        valid["score"] = (
            0.30 * valid["IDF1"]
            + 0.25 * valid["F1"]
            - 0.15 * valid["IDSW"].fillna(0) / max(1, valid["num_detections_before_filter"].max())
            - 0.15 * valid["track_breaks"].fillna(0) / max(1, valid["num_detections_before_filter"].max())
            - 0.10 * valid["FP_per_100_frames"].fillna(0) / 100.0
        )
    best = valid.sort_values("score", ascending=False).iloc[0]
    return {
        "selection_status": "selected",
        "holdout_allowed": True,
        "selected_scenario": best["scenario"],
        "selected_t_norm": best["t_norm"],
        "selected_tau_Q": None if pd.isna(best.get("tau_Q")) else float(best["tau_Q"]),
        "selected_reweight_mode": first_value(best.to_frame().T, "reweight_mode"),
        "selected_q_floor": first_value(best.to_frame().T, "q_floor"),
        "selected_q_floor_new": first_value(best.to_frame().T, "q_floor_new"),
        "selected_q_floor_existing": first_value(best.to_frame().T, "q_floor_existing"),
        "selected_q_new_mode": first_value(best.to_frame().T, "q_new_mode"),
        "selected_q_new_min": first_value(best.to_frame().T, "q_new_min"),
        "selected_gamma": first_value(best.to_frame().T, "gamma"),
        "selected_q_hard_min": first_value(best.to_frame().T, "q_hard_min"),
        "selected_new_track_threshold": first_value(best.to_frame().T, "new_track_threshold"),
        "selected_existing_track_threshold": first_value(best.to_frame().T, "existing_track_threshold"),
        "x_i_selected_method": "x_top5_inside",
        "xai_method": "eigencam",
        "score": float(best["score"]),
        "FP_delta_vs_S_naive": int(best["FP"] - naive_best["FP"]),
        "FN_delta_vs_S_naive": int(best["FN"] - naive_best["FN"]),
        "F1_delta_vs_S_naive": float(best["F1"] - naive_best["F1"]),
        "reason": "adaptive_new_track_gating_reduces_fp_without_recall_collapse" if has_v3 else "soft_reweighting_reduces_fp_without_recall_collapse",
    }


def build_ablation(features: pd.DataFrame, num_frames: int, expected_gt: int | None = None) -> pd.DataFrame:
    variants = {
        "c_only": ["c_i"],
        "c_k": ["c_i", "k_i"],
        "c_s": ["c_i", "s_i"],
        "c_k_s": ["c_i", "k_i", "s_i"],
        "c_k_x": ["c_i", "k_i", "x_i"],
        "c_s_x": ["c_i", "s_i", "x_i"],
        "c_k_s_x": ["c_i", "k_i", "s_i", "x_i"],
    }
    rows = []
    for name, cols in variants.items():
        frame = features.copy()
        frame["scenario"] = name
        frame["Q_i"] = frame[cols].min(axis=1)
        frame["accepted"] = frame["Q_i"] >= 0.30
        row = _summary(frame, name, "min", 0.30, num_frames, expected_gt)
        row["variant"] = name
        row["features_used"] = ",".join(cols)
        row["interpretation"] = "pseudo-swarm calibration ablation"
        rows.append(row)
    return pd.DataFrame(rows)


def new_track_suppression(summary: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    base_new = features["track_status"].eq("new_candidate")
    total_new = int(base_new.sum())
    total_new_fp = int((base_new & ~features["eval_is_tp"].astype(bool)).sum()) if "eval_is_tp" in features else 0
    rows = []
    for _, row in summary[summary["scenario"].str.contains("soft", case=False, na=False)].iterrows():
        suppressed = int(row["num_rejected"])
        fp_frac = total_new_fp / max(1, total_new)
        suppressed_fp = int(round(suppressed * fp_frac))
        rows.append(
            {
                "scenario": row["scenario"],
                "reweight_mode": row.get("reweight_mode"),
                "q_floor": row.get("q_floor"),
                "new_track_threshold": row.get("new_track_threshold"),
                "num_new_track_candidates": total_new,
                "num_new_tracks_created": max(0, total_new - suppressed),
                "num_new_tracks_suppressed": suppressed,
                "suppressed_TP": max(0, suppressed - suppressed_fp),
                "suppressed_FP": suppressed_fp,
                "suppressed_FP_rate": suppressed_fp / max(1, suppressed),
                "suppressed_TP_rate": max(0, suppressed - suppressed_fp) / max(1, suppressed),
            }
        )
    return pd.DataFrame(rows)


def new_track_gate_summary(summary: pd.DataFrame, feature_frames: list[pd.DataFrame]) -> pd.DataFrame:
    naive = summary[summary["scenario"] == "S_naive"].sort_values(["F1", "IDF1"], ascending=False)
    base = naive.iloc[0] if len(naive) else None
    expected_gt = _expected_gt_from_summary(summary)
    rows = []
    for frame in feature_frames:
        if frame.empty or "scenario" not in frame:
            continue
        scenario = str(frame["scenario"].iloc[0])
        if scenario not in {"S2_tnorm_soft_v2", "S2_tnorm_newgate", "S2_tnorm_adaptive"}:
            continue
        is_new = frame["track_status"].eq("new_candidate")
        is_tp = frame["eval_is_tp"].astype(bool)
        suppressed = is_new & ~frame["accepted"].astype(bool)
        metric = _summary(
            frame,
            scenario,
            first_value(frame, "t_norm"),
            None,
            int(frame[["sequence_id", "frame_id"]].drop_duplicates().shape[0]),
            expected_gt,
        )
        rows.append(
            {
                "scenario": scenario,
                "q_new_mode": first_value(frame, "q_new_mode"),
                "q_floor_new": first_value(frame, "q_floor_new"),
                "q_floor_existing": first_value(frame, "q_floor_existing"),
                "new_track_threshold": first_value(frame, "new_track_threshold"),
                "existing_track_threshold": first_value(frame, "existing_track_threshold"),
                "q_new_min": first_value(frame, "q_new_min"),
                "num_new_track_candidates": int(is_new.sum()),
                "new_track_TP_candidates": int((is_new & is_tp).sum()),
                "new_track_FP_candidates": int((is_new & ~is_tp).sum()),
                "suppressed_TP": int((suppressed & is_tp).sum()),
                "suppressed_FP": int((suppressed & ~is_tp).sum()),
                "suppressed_FP_rate": int((suppressed & ~is_tp).sum()) / max(1, int(suppressed.sum())),
                "suppressed_TP_rate": int((suppressed & is_tp).sum()) / max(1, int(suppressed.sum())),
                "created_tracks": int((is_new & frame["accepted"].astype(bool)).sum()),
                "FP_delta_vs_S_naive": None if base is None else int(metric["FP"] - base["FP"]),
                "FN_delta_vs_S_naive": None if base is None else int(metric["FN"] - base["FN"]),
                "F1_delta_vs_S_naive": None if base is None else float(metric["F1"] - base["F1"]),
            }
        )
    return pd.DataFrame(rows)


def candidate_score_distribution(feature_frames: list[pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for frame in feature_frames:
        if frame.empty or "scenario" not in frame or "Q_new" not in frame:
            continue
        scenario = str(frame["scenario"].iloc[0])
        if scenario not in {"S2_tnorm_soft_v2", "S2_tnorm_newgate", "S2_tnorm_adaptive"}:
            continue
        for (status, is_tp, q_mode), group in frame.groupby(["track_status", "eval_is_tp", "q_new_mode"], dropna=False):
            q = group["Q_new"]
            rows.append(
                {
                    "scenario": scenario,
                    "track_status": status,
                    "eval_is_tp": bool(is_tp),
                    "q_new_mode": q_mode,
                    "mean_Q_new": float(q.mean()),
                    "p05_Q_new": float(q.quantile(0.05)),
                    "p25_Q_new": float(q.quantile(0.25)),
                    "p50_Q_new": float(q.quantile(0.50)),
                    "p75_Q_new": float(q.quantile(0.75)),
                    "p95_Q_new": float(q.quantile(0.95)),
                    "mean_confidence": float(group["confidence_original"].mean() if "confidence_original" in group else group["confidence"].mean()),
                    "mean_confidence_new": float(group["confidence_new"].mean() if "confidence_new" in group else group["confidence"].mean()),
                }
            )
    return pd.DataFrame(rows)


def xai_newtrack_veto_summary(summary: pd.DataFrame, feature_frames: list[pd.DataFrame]) -> pd.DataFrame:
    s2 = summary[summary["scenario"] == "S2_tnorm_soft"].sort_values(["F1", "IDF1"], ascending=False)
    base = s2.iloc[0] if len(s2) else None
    expected_gt = _expected_gt_from_summary(summary)
    selected = select_params(summary, pd.DataFrame())
    rows = []
    for frame in feature_frames:
        if frame.empty or "scenario" not in frame or str(frame["scenario"].iloc[0]) != "S3_xai_newtrack_veto":
            continue
        called = frame["xai_called"].astype(bool) if "xai_called" in frame else pd.Series(False, index=frame.index)
        vetoed = called & ~frame["accepted"].astype(bool)
        is_tp = frame["eval_is_tp"].astype(bool)
        metric = _summary(
            frame,
            "S3_xai_newtrack_veto",
            first_value(frame, "t_norm"),
            None,
            int(frame[["sequence_id", "frame_id"]].drop_duplicates().shape[0]),
            expected_gt,
        )
        row = {
            "xai_veto_mode": first_value(frame, "xai_veto_mode"),
            "boundary_margin": first_value(frame, "boundary_margin"),
            "q_xai_trigger": first_value(frame, "q_xai_trigger"),
            "x_min": first_value(frame, "x_min"),
            "xai_floor": first_value(frame, "xai_floor"),
            "xai_max_per_frame": first_value(frame, "xai_max_per_frame"),
            "num_xai_called": int(called.sum()),
            "xai_calls_per_frame": int(called.sum()) / max(1, int(frame[["sequence_id", "frame_id"]].drop_duplicates().shape[0])),
            "num_vetoed": int(vetoed.sum()),
            "vetoed_TP": int((vetoed & is_tp).sum()),
            "vetoed_FP": int((vetoed & ~is_tp).sum()),
            "FP_delta_vs_S2": None if base is None else int(metric["FP"] - base["FP"]),
            "FN_delta_vs_S2": None if base is None else int(metric["FN"] - base["FN"]),
            "F1_delta_vs_S2": None if base is None else float(metric["F1"] - base["F1"]),
            "latency_ms": 0.0,
        }
        row["selected_candidate"] = _xai_summary_row_selected(row, selected)
        rows.append(row)
    return pd.DataFrame(rows)


def _xai_summary_row_selected(row: dict, selected: dict) -> bool:
    if selected.get("selected_scenario") != "S3_xai_newtrack_veto":
        return False
    checks = [
        ("xai_veto_mode", "selected_xai_veto_mode"),
        ("boundary_margin", "selected_boundary_margin"),
        ("q_xai_trigger", "selected_q_xai_trigger"),
        ("x_min", "selected_x_min"),
        ("xai_floor", "selected_xai_floor"),
    ]
    for row_key, selected_key in checks:
        if not _same_value(row.get(row_key), selected.get(selected_key)):
            return False
    return True


def _same_value(a, b) -> bool:
    if a is None and b is None:
        return True
    if pd.isna(a) and (b is None or pd.isna(b)):
        return True
    if b is None and pd.isna(a):
        return True
    try:
        return abs(float(a) - float(b)) < 1e-9
    except (TypeError, ValueError):
        return str(a) == str(b)


def _expected_gt_from_summary(summary: pd.DataFrame) -> int | None:
    if summary.empty or not {"TP", "FN"}.issubset(summary.columns):
        return None
    totals = (summary["TP"] + summary["FN"]).dropna()
    return int(totals.max()) if len(totals) else None


def recall_preservation(summary: pd.DataFrame) -> pd.DataFrame:
    naive = summary[summary["scenario"] == "S_naive"].sort_values(["F1", "IDF1"], ascending=False)
    if naive.empty:
        return pd.DataFrame()
    base = naive.iloc[0]
    rows = []
    for _, row in summary[summary["scenario"].str.contains("soft|tnorm", case=False, na=False)].iterrows():
        rows.append(
            {
                "scenario": row["scenario"],
                "reweight_mode": row.get("reweight_mode"),
                "q_floor": row.get("q_floor"),
                "TP": row["TP"],
                "FN": row["FN"],
                "FN_delta_vs_S_naive": row["FN"] - base["FN"],
                "recall": row["recall"],
                "recall_delta_vs_S_naive": row["recall"] - base["recall"],
                "num_existing_tracks_kept": None,
                "num_existing_tracks_broken": row["track_breaks"],
            }
        )
    return pd.DataFrame(rows)


def write_selected_params(path: Path, selected: dict) -> None:
    path.write_text(yaml.safe_dump(selected, sort_keys=False, allow_unicode=True), encoding="utf-8")


def inject_pseudo_attack(detections: pd.DataFrame, attack_cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    cfg = attack_cfg.get("attack_model", attack_cfg)
    rng = np.random.default_rng(int(cfg.get("seed", 42)))
    out = detections.copy().reset_index(drop=True)
    out["source_error_type"] = "clean_tp"
    out["is_corrupted"] = False
    out["dropped_by_attack"] = False
    out["eval_is_tp"] = out["track_id"] != -1
    events: list[dict] = []
    max_det_id = int(out["det_id"].max()) if len(out) else 0
    enabled = set(cfg.get("enabled_error_types", []))

    if "bbox_shift_tp" in enabled and cfg.get("bbox_shift_tp", {}).get("enabled", False):
        frac = float(cfg["bbox_shift_tp"].get("target_fraction", 0.15))
        idxs = sample_tp_indices(out, frac, rng)
        shift_lo, shift_hi = cfg["bbox_shift_tp"].get("shift_ratio_range", [0.15, 0.40])
        conf_lo, conf_hi = cfg["bbox_shift_tp"].get("confidence_multiplier_range", [0.80, 1.00])
        for idx in idxs:
            before = out.loc[idx].copy()
            box = row_box(before)
            ratio = float(rng.uniform(shift_lo, shift_hi))
            dx = rng.choice([-1.0, 1.0]) * box_w(box) * ratio
            dy = rng.choice([-1.0, 1.0]) * box_h(box) * ratio
            out.loc[idx, ["x1", "y1", "x2", "y2"]] = [box[0] + dx, box[1] + dy, box[2] + dx, box[3] + dy]
            out.loc[idx, "confidence"] = float(out.loc[idx, "confidence"]) * float(rng.uniform(conf_lo, conf_hi))
            out.loc[idx, "eval_is_tp"] = _iou(row_box(before), row_box(out.loc[idx])) >= 0.5
            out.loc[idx, "source_error_type"] = "bbox_shift_tp"
            out.loc[idx, "is_corrupted"] = True
            events.append(event_row("bbox_shift_tp", before, out.loc[idx]))

    if "inter_agent_misalignment" in enabled and cfg.get("inter_agent_misalignment", {}).get("enabled", False):
        frac = float(cfg["inter_agent_misalignment"].get("target_fraction", 0.10))
        idxs = sample_one_agent_track_indices(out, frac, rng)
        shift_lo, shift_hi = cfg["inter_agent_misalignment"].get("bbox_shift_ratio_range", [0.10, 0.30])
        for idx in idxs:
            before = out.loc[idx].copy()
            box = row_box(before)
            ratio = float(rng.uniform(shift_lo, shift_hi))
            dx = rng.choice([-1.0, 1.0]) * box_w(box) * ratio
            dy = rng.choice([-1.0, 1.0]) * box_h(box) * ratio
            out.loc[idx, ["x1", "y1", "x2", "y2"]] = [box[0] + dx, box[1] + dy, box[2] + dx, box[3] + dy]
            out.loc[idx, "source_error_type"] = "inter_agent_misalignment"
            out.loc[idx, "is_corrupted"] = True
            out.loc[idx, "eval_is_tp"] = _iou(row_box(before), row_box(out.loc[idx])) >= 0.5
            events.append(event_row("inter_agent_misalignment", before, out.loc[idx], is_agent_specific=True))

    if "class_confusion" in enabled and cfg.get("class_confusion", {}).get("enabled", False):
        frac = float(cfg["class_confusion"].get("target_fraction", 0.05))
        idxs = sample_tp_indices(out, frac, rng)
        conf_lo, conf_hi = cfg["class_confusion"].get("confidence_multiplier_range", [0.85, 1.00])
        for idx in idxs:
            before = out.loc[idx].copy()
            new_class = confused_class(int(before["class_id"]))
            out.loc[idx, "class_id"] = new_class
            out.loc[idx, "class_name"] = VISDRONE_CLASSES.get(new_class, "unknown")
            out.loc[idx, "confidence"] = float(out.loc[idx, "confidence"]) * float(rng.uniform(conf_lo, conf_hi))
            out.loc[idx, "source_error_type"] = "class_confusion"
            out.loc[idx, "is_corrupted"] = True
            out.loc[idx, "eval_is_tp"] = False
            events.append(event_row("class_confusion", before, out.loc[idx]))

    if "agent_specific_drop" in enabled and cfg.get("agent_specific_drop", {}).get("enabled", False):
        frac = float(cfg["agent_specific_drop"].get("target_fraction", 0.10))
        idxs = sample_one_agent_track_indices(out, frac, rng)
        for idx in idxs:
            before = out.loc[idx].copy()
            out.loc[idx, "dropped_by_attack"] = True
            out.loc[idx, "source_error_type"] = "agent_specific_drop"
            out.loc[idx, "is_corrupted"] = True
            events.append(event_row("agent_specific_drop", before, None, is_agent_specific=True, is_dropped=True))

    if "high_conf_fp" in enabled and cfg.get("high_conf_fp", {}).get("enabled", False):
        fp_cfg = cfg["high_conf_fp"]
        rate = float(fp_cfg.get("per_frame_rate", 0.10))
        conf_lo, conf_hi = fp_cfg.get("confidence_range", [0.55, 0.90])
        frame_keys = out[["sequence_id", "frame_id", "agent_id"]].drop_duplicates()
        for rec in frame_keys.itertuples(index=False):
            if rng.random() > rate:
                continue
            group = out[(out.sequence_id == rec.sequence_id) & (out.frame_id == rec.frame_id) & (out.agent_id == rec.agent_id) & (out.track_id != -1)]
            if group.empty:
                continue
            base = group.sample(1, random_state=int(rng.integers(0, 2**31 - 1))).iloc[0]
            box = make_fp_box(base, group, rng)
            max_det_id += 1
            row = base.copy()
            row["det_id"] = max_det_id
            row["track_id"] = -1
            row["confidence"] = float(rng.uniform(conf_lo, conf_hi))
            row[["x1", "y1", "x2", "y2"]] = box
            row["source_error_type"] = "high_conf_fp"
            row["is_corrupted"] = True
            row["dropped_by_attack"] = False
            row["eval_is_tp"] = False
            out = pd.concat([out, pd.DataFrame([row])], ignore_index=True)
            events.append(event_row("high_conf_fp", None, row, is_high_conf_fp=True))

    out = out[out["dropped_by_attack"] == False].copy()
    return out.reset_index(drop=True), pd.DataFrame(events)


def sample_tp_indices(frame: pd.DataFrame, fraction: float, rng: np.random.Generator) -> list[int]:
    idxs = frame.index[(frame["track_id"] != -1) & (frame.get("dropped_by_attack", False) == False)].to_numpy()
    n = min(len(idxs), int(round(len(idxs) * fraction)))
    return sorted(rng.choice(idxs, size=n, replace=False).tolist()) if n > 0 else []


def sample_one_agent_track_indices(frame: pd.DataFrame, fraction: float, rng: np.random.Generator) -> list[int]:
    keys = frame[frame["track_id"] != -1].groupby(["sequence_id", "frame_id", "track_id"]).filter(lambda g: g["agent_id"].nunique() > 1)
    groups = list(keys.groupby(["sequence_id", "frame_id", "track_id"]).groups.values())
    n = min(len(groups), int(round(len(groups) * fraction)))
    if n <= 0:
        return []
    chosen = rng.choice(np.arange(len(groups)), size=n, replace=False)
    return [int(rng.choice(list(groups[i]))) for i in chosen]


def confused_class(class_id: int) -> int:
    return {1: 2, 2: 1, 3: 10, 10: 3}.get(class_id, 2 if class_id != 2 else 1)


def make_fp_box(base: pd.Series, group: pd.DataFrame, rng: np.random.Generator) -> tuple[float, float, float, float]:
    box = row_box(base)
    w, h = box_w(box), box_h(box)
    cx = (box[0] + box[2]) / 2.0 + rng.choice([-1.0, 1.0]) * w * float(rng.uniform(1.2, 2.0))
    cy = (box[1] + box[3]) / 2.0 + rng.choice([-1.0, 1.0]) * h * float(rng.uniform(1.2, 2.0))
    for _ in range(20):
        candidate = (cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0)
        if max(_iou(candidate, row_box(r)) for _, r in group.iterrows()) < 0.20:
            return candidate
        cx += rng.choice([-1.0, 1.0]) * w
        cy += rng.choice([-1.0, 1.0]) * h
    return (box[2] + w, box[3] + h, box[2] + 2 * w, box[3] + 2 * h)


def event_row(event_type: str, before: pd.Series | None, after: pd.Series | None, is_high_conf_fp: bool = False, is_agent_specific: bool = False, is_dropped: bool = False) -> dict:
    src = after if after is not None else before
    before_box = row_box(before) if before is not None else None
    after_box = row_box(after) if after is not None else None
    iou_after = 0.0 if is_high_conf_fp or after is None or before is None else _iou(before_box, after_box)
    return {
        "sequence_id": src["sequence_id"],
        "frame_id": int(src["frame_id"]),
        "agent_id": src["agent_id"],
        "event_type": event_type,
        "original_det_id": None if before is None else int(before["det_id"]),
        "new_det_id": None if after is None else int(after["det_id"]),
        "gt_id": None if src["track_id"] == -1 else int(src["track_id"]),
        "class_before": None if before is None else before.get("class_name"),
        "class_after": None if after is None else after.get("class_name"),
        "confidence_before": None if before is None else float(before["confidence"]),
        "confidence_after": None if after is None else float(after["confidence"]),
        "bbox_before": None if before_box is None else json.dumps(list(before_box)),
        "bbox_after": None if after_box is None else json.dumps(list(after_box)),
        "iou_before_gt": 1.0 if before is not None and before["track_id"] != -1 else 0.0,
        "iou_after_gt": iou_after,
        "is_high_conf_fp": is_high_conf_fp,
        "is_shifted_tp": event_type == "bbox_shift_tp",
        "is_dropped_tp": is_dropped,
        "is_class_confused": event_type == "class_confusion",
        "is_agent_specific": is_agent_specific,
    }


def attack_event_summary(events: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame(columns=["event_type", "num_events"])
    rows = []
    for event_type, group in events.groupby("event_type", sort=False):
        rows.append(
            {
                "event_type": event_type,
                "num_events": len(group),
                "mean_confidence_before": float(group["confidence_before"].dropna().mean()) if group["confidence_before"].notna().any() else None,
                "mean_confidence_after": float(group["confidence_after"].dropna().mean()) if group["confidence_after"].notna().any() else None,
                "mean_iou_before_gt": float(group["iou_before_gt"].dropna().mean()),
                "mean_iou_after_gt": float(group["iou_after_gt"].dropna().mean()),
                "num_high_conf": int((group["confidence_after"].fillna(0) >= 0.5).sum()),
                "num_agent_specific": int(group["is_agent_specific"].sum()),
            }
        )
    return pd.DataFrame(rows)


def error_type_breakdown(summary: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame(columns=["error_type", "scenario", "TP", "FP", "FN", "F1", "IDF1", "num_events", "num_fixed", "num_missed", "num_over_filtered"])
    rows = []
    counts = events["event_type"].value_counts().to_dict()
    for event_type, num in counts.items():
        for _, row in summary[summary["scenario"].isin(["S1_fgsm", "S_naive", "S2_tnorm_no_xai", "S3_tnorm_xai"])].iterrows():
            rows.append(
                {
                    "error_type": event_type,
                    "scenario": row["scenario"],
                    "TP": row.get("TP"),
                    "FP": row.get("FP"),
                    "FN": row.get("FN"),
                    "F1": row.get("F1"),
                    "IDF1": row.get("IDF1"),
                    "num_events": int(num),
                    "num_fixed": None,
                    "num_missed": None,
                    "num_over_filtered": None,
                }
            )
    return pd.DataFrame(rows)


def row_box(row: pd.Series) -> tuple[float, float, float, float]:
    return (float(row["x1"]), float(row["y1"]), float(row["x2"]), float(row["y2"]))


def box_w(box: tuple[float, float, float, float]) -> float:
    return max(1.0, box[2] - box[0])


def box_h(box: tuple[float, float, float, float]) -> float:
    return max(1.0, box[3] - box[1])


def _load_yaml(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}
