from __future__ import annotations

import json
import math
import subprocess
import time
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from defense4uavswarm.v8_sim import (
    EPS,
    add_features,
    box_iou,
    evaluate_experiment,
    feature_matrix,
    metric_summary,
    read_json,
    scenario_acceptance,
    train_rf_model,
    vector_iou,
    write_json,
)


V9_SCENARIOS = [
    "s2_tnorm_temporal",
    "s2_tnorm_temporal_logodds",
    "s2_tnorm_temporal_logodds_maha",
    "s2_tnorm_temporal_logodds_maha_world",
    "s2_tnorm_temporal_logodds_maha_world_epi",
    "s2_v9_selected",
]


def load_config(path: str | Path = "configs/v9_geomdyn_config.yaml") -> dict[str, Any]:
    try:
        import yaml

        return yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    except Exception:
        return {}


def run_v9_experiment(
    manifest_path: str | Path,
    gt_2d_path: str | Path,
    detections_path: str | Path,
    scenarios: list[str],
    holdout_scenes: list[str],
    output_dir: str | Path,
    config_path: str | Path = "configs/v9_geomdyn_config.yaml",
) -> dict[str, Any]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    cfg = load_config(config_path)
    manifest = read_json(manifest_path)
    gt = pd.DataFrame(read_json(gt_2d_path)["boxes"])
    det = pd.DataFrame(read_json(detections_path)["detections"])
    if holdout_scenes:
        gt = gt[gt["scene_id"].isin(holdout_scenes)].copy()
        det = det[det["scene_id"].isin(holdout_scenes)].copy()
    det = prepare_v9_features(det, manifest, cfg)
    expected_gt = len(gt)
    num_frames = gt[["scene_id", "frame_id"]].drop_duplicates().shape[0]
    runtime_base = max(EPS, 0.001 * len(det) / max(1, num_frames))
    rf_model = train_rf_model(det, gt) if any(s.lower() == "s2_learned_fp_gate" for s in scenarios) else None

    rows = []
    events = []
    for scenario in scenarios:
        start = time.perf_counter()
        accepted, extra = v9_acceptance(det, scenario, cfg, rf_model)
        row = metric_summary(gt, det, accepted, scenario, expected_gt, num_frames, manifest, runtime_base)
        row["scenario"] = normalize_v9_name(str(row["scenario"]))
        row.update(v9_flags(scenario))
        row["runtime_ms_per_frame"] = (time.perf_counter() - start) / max(1, num_frames) * 1000.0 + row.get("runtime_ms_per_frame", 0.0)
        rows.append(row)
        events.extend(extra.get("events", []))
    main = pd.DataFrame(rows)
    main.to_csv(out / "main_comparison_table.csv", index=False)
    write_ablation(main, out)
    write_selected(main, out, cfg)
    write_diagnostics(det, out)
    pd.DataFrame(events).to_csv(out / "track_events.csv", index=False)
    main[main["scenario"].isin(["S2_learned_fp_gate", "S2_v9_selected"])].to_csv(out / "rf_comparison_summary.csv", index=False)
    return {"summary": main, "features": det}


def prepare_v9_features(det: pd.DataFrame, manifest: dict[str, Any], cfg: dict[str, Any], modifiers: dict[str, float] | None = None) -> pd.DataFrame:
    d = add_features(det, modifiers or {})
    d["track_key_v9"] = d["_track_key"].astype(str)
    d = add_world_projection_features(d, manifest, cfg)
    d = add_mahalanobis_features(d, cfg)
    d = add_epipolar_features(d, cfg)
    return d


def add_world_projection_features(d: pd.DataFrame, manifest: dict[str, Any], cfg: dict[str, Any]) -> pd.DataFrame:
    agent_meta = {}
    for frame in manifest.get("frames", []):
        for agent in frame.get("agents", []):
            agent_meta[(frame["scene_id"], int(frame["frame_id"]), agent["agent_id"])] = agent
    wx = np.full(len(d), np.nan)
    wy = np.full(len(d), np.nan)
    available = np.zeros(len(d), dtype=bool)
    for pos, row in enumerate(d.itertuples()):
        agent = agent_meta.get((row.scene_id, int(row.frame_id), row.agent_id))
        if not agent:
            continue
        projected = image_bottom_center_to_ground((float(row.x1 + row.x2) / 2.0, float(row.y2)), agent)
        if projected is None:
            continue
        wx[pos], wy[pos] = projected
        available[pos] = True
    d = d.copy()
    d["world_x_proj"] = wx
    d["world_y_proj"] = wy
    d["world_available"] = available

    support_radius = float(cfg.get("world", {}).get("support_radius_m", 5.0))
    support = np.zeros(len(d), dtype=int)
    world_min_dist = np.full(len(d), 999.0)
    for _, group in d[d["world_available"]].groupby(["scene_id", "frame_id", "class_id"], sort=False):
        idx = group.index.to_numpy()
        pts = group[["world_x_proj", "world_y_proj"]].to_numpy(dtype=float)
        agents = group["agent_id"].to_numpy()
        for local_i, row_idx in enumerate(idx):
            support_agents = set()
            for local_j, other_idx in enumerate(idx):
                if local_i == local_j or agents[local_i] == agents[local_j]:
                    continue
                dist = float(np.linalg.norm(pts[local_i] - pts[local_j]))
                world_min_dist[row_idx] = min(world_min_dist[row_idx], dist)
                if dist <= support_radius:
                    support_agents.add(agents[local_j])
            support[row_idx] = len(support_agents)
    sigma = float(cfg.get("world", {}).get("sigma_world_m", 5.0))
    s_world = np.exp(-(world_min_dist**2) / max(EPS, 2.0 * sigma**2))
    s_world[world_min_dist >= 999.0] = 0.0
    max_agents = max(1, int(d["agent_id"].nunique()) - 1)
    d["world_support_count"] = support
    d["world_support_ratio"] = support / max_agents
    d["s_world"] = s_world
    d["s_support"] = np.minimum(1.0, support / max(1, int(cfg.get("world", {}).get("required_support_count", 1))))
    d["s_geom_world"] = np.maximum.reduce([d["s_i"].to_numpy(), d["s_world"].to_numpy(), 0.5 * d["s_support"].to_numpy()])
    return d


def image_bottom_center_to_ground(point: tuple[float, float], agent: dict[str, Any]) -> tuple[float, float] | None:
    intr = agent.get("intrinsics") or {}
    ext = agent.get("extrinsics") or {}
    r_wc = np.asarray(ext.get("rotation_world_to_camera"), dtype=float)
    t_wc = np.asarray(ext.get("translation_world_to_camera"), dtype=float)
    if r_wc.shape != (3, 3) or t_wc.shape != (3,):
        return None
    u, v = point
    fx, fy, cx, cy = float(intr.get("fx", 0)), float(intr.get("fy", 0)), float(intr.get("cx", 0)), float(intr.get("cy", 0))
    if fx <= 0 or fy <= 0:
        return None
    ray_cam = np.array([(u - cx) / fx, (v - cy) / fy, 1.0], dtype=float)
    ray_world = r_wc.T @ ray_cam
    cam_center = -(r_wc.T @ t_wc)
    if abs(ray_world[2]) < 1e-6:
        return None
    scale = -cam_center[2] / ray_world[2]
    if scale <= 0:
        return None
    point_world = cam_center + scale * ray_world
    return float(point_world[0]), float(point_world[1])


def add_mahalanobis_features(d: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    base = float(cfg.get("mahalanobis", {}).get("proxy_sigma_base_px", 40.0))
    sigma_min = float(cfg.get("mahalanobis", {}).get("proxy_sigma_min_px", 8.0))
    sigma_max = float(cfg.get("mahalanobis", {}).get("proxy_sigma_max_px", 40.0))
    meas = float(cfg.get("mahalanobis", {}).get("measurement_noise_px", 20.0))
    scale = float(cfg.get("mahalanobis", {}).get("maha_scale", 1.0))
    clip = float(cfg.get("mahalanobis", {}).get("max_maha_d2_clip", 25.0))
    d2 = np.full(len(d), np.nan)
    available = np.zeros(len(d), dtype=bool)
    for _, group in d.groupby(["scene_id", "agent_id", "class_id", "_track_key"], sort=False):
        prev = None
        prev2 = None
        for idx, row in group.sort_values("frame_id").iterrows():
            z = np.array([(row.x1 + row.x2) / 2.0, (row.y1 + row.y2) / 2.0], dtype=float)
            if prev is None:
                prev2 = prev
                prev = (int(row.frame_id), z)
                continue
            if prev2 is not None:
                dt_prev = max(1, prev[0] - prev2[0])
                vel = (prev[1] - prev2[1]) / dt_prev
            else:
                vel = np.zeros(2)
            dt = max(1, int(row.frame_id) - prev[0])
            pred = prev[1] + vel * dt
            age = float(row.temporal_age)
            sigma = min(sigma_max, max(sigma_min, base / math.sqrt(age + 1.0)))
            var = sigma**2 + meas**2
            val = float(np.sum((z - pred) ** 2) / max(EPS, var))
            d2[idx] = min(clip, val)
            available[idx] = True
            prev2 = prev
            prev = (int(row.frame_id), z)
    d = d.copy()
    d["maha_d2"] = d2
    d["maha_available"] = available
    d["k_maha"] = np.where(available, np.exp(-0.5 * np.nan_to_num(d2, nan=clip) / max(EPS, scale)), d["k_i"])
    return d


def add_epipolar_features(d: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    # Full epipolar geometry is approximated with cross-agent horizontal line
    # agreement in this controlled pinhole setup. It is soft support only.
    sigma = float(cfg.get("epipolar", {}).get("sigma_epi_px", 25.0))
    dist = np.full(len(d), np.nan)
    available = np.zeros(len(d), dtype=bool)
    for _, group in d.groupby(["scene_id", "frame_id", "class_id"], sort=False):
        idx = group.index.to_numpy()
        centers = np.column_stack(((group["x1"] + group["x2"]) / 2.0, (group["y1"] + group["y2"]) / 2.0))
        agents = group["agent_id"].to_numpy()
        for i, row_idx in enumerate(idx):
            other = [j for j in range(len(idx)) if agents[j] != agents[i]]
            if not other:
                continue
            vals = np.abs(centers[other, 1] - centers[i, 1])
            dist[row_idx] = float(np.min(vals))
            available[row_idx] = True
    d = d.copy()
    d["epi_distance"] = dist
    d["epi_available"] = available
    d["s_epi"] = np.where(available, np.exp(-(np.nan_to_num(dist, nan=999.0) ** 2) / max(EPS, 2.0 * sigma**2)), 0.0)
    d["s_geom_epi"] = np.maximum.reduce([d["s_geom_world"].to_numpy(), d["s_epi"].to_numpy()])
    return d


def v9_acceptance(det: pd.DataFrame, scenario: str, cfg: dict[str, Any], rf_model: Any = None) -> tuple[pd.Series, dict[str, Any]]:
    s = scenario.lower()
    if s in {"s_naive", "s2_tnorm_soft", "s2_tnorm_temporal", "s2_ema_confidence_gate", "s2_support_count_gate"}:
        from defense4uavswarm.v8_sim import load_temporal_params

        return scenario_acceptance(det, s, load_temporal_params("configs/selected_s2_temporal.yaml"), rf_model)
    if s == "s2_learned_fp_gate" and rf_model is not None:
        fp_prob = rf_model.predict_proba(feature_matrix(det))[:, 1]
        return pd.Series(fp_prob < 0.55, index=det.index), {}
    q_floor = float(cfg.get("trust", {}).get("q_floor", 0.6))
    new_thr = float(cfg.get("trust", {}).get("new_track_threshold", 0.5))
    k = det["k_i"].copy()
    support = det["s_i"].copy()
    selected = s == "s2_v9_selected"
    if "maha" in s or selected:
        k = np.maximum(k, det["k_maha"])
    if "world" in s or selected:
        support = np.maximum(support, det["s_geom_world"])
    if "epi" in s or selected:
        support = np.maximum(support, det["s_geom_epi"])
    q = np.minimum.reduce([det["c_i"].to_numpy(), np.asarray(k), np.asarray(support)])
    conf_rw = det["confidence"] * (q_floor + (1.0 - q_floor) * q)
    if "logodds" in s or selected:
        return logodds_acceptance(det, conf_rw, cfg, scenario=s)
    return pd.Series(conf_rw >= new_thr, index=det.index), {}


def logodds_acceptance(det: pd.DataFrame, conf_rw: pd.Series | np.ndarray, cfg: dict[str, Any], scenario: str) -> tuple[pd.Series, dict[str, Any]]:
    if "track_key_v9" not in det.columns:
        det = det.copy()
        if "_track_key" in det.columns:
            det["track_key_v9"] = det["_track_key"].astype(str)
        else:
            det["track_key_v9"] = det.get("det_id", pd.Series(range(len(det)), index=det.index)).astype(str)
    log_cfg = cfg.get("logodds", {})
    trust_cfg = cfg.get("trust", {})
    strong_reward = float(log_cfg.get("strong_match_reward", 0.9))
    weak_reward = float(log_cfg.get("weak_match_reward", 0.4))
    miss_penalty = float(log_cfg.get("miss_penalty", 0.25))
    confirm_log = float(log_cfg.get("confirm_log_odds", 1.2))
    min_hits = int(log_cfg.get("min_hits_to_confirm", 2))
    max_age = int(log_cfg.get("max_pending_age", 5))
    min_support = int(trust_cfg.get("selected_min_support", 1))
    min_age = int(trust_cfg.get("selected_min_temporal_age", 2))
    min_conf = float(trust_cfg.get("selected_min_confidence", 0.55))
    conf = pd.Series(conf_rw, index=det.index)
    accepted = pd.Series(False, index=det.index)
    events = []
    for _, group in det.groupby(["scene_id", "agent_id", "class_id", "track_key_v9"], sort=False):
        log_odds = 0.0
        hits = 0
        misses = 0
        for row in group.sort_values("frame_id").itertuples():
            idx = row.Index
            before = log_odds
            match_type = "none"
            reward = 0.0
            if conf.loc[idx] >= 0.40:
                reward = strong_reward
                match_type = "strong"
            elif conf.loc[idx] >= 0.30:
                reward = weak_reward
                match_type = "weak"
            else:
                reward = -miss_penalty
                misses += 1
            if reward > 0:
                hits += 1
            log_odds += reward
            age = int(row.temporal_age)
            support_ok = int(getattr(row, "world_support_count", getattr(row, "support_count", 0))) >= min_support
            memory_ok = age >= min_age and conf.loc[idx] >= min_conf and log_odds >= confirm_log and hits >= min_hits
            # Geometry support is the high-recall path; log-odds memory recovers
            # single-view persistent objects without opening random one-frame FP.
            confirmed = bool(support_ok or memory_ok)
            if confirmed:
                accepted.loc[idx] = True
            if age <= max_age:
                events.append(
                    {
                        "scenario": normalize_v9_name(scenario),
                        "scene_id": row.scene_id,
                        "frame_id": int(row.frame_id),
                        "agent_id": row.agent_id,
                        "pending_id": row.det_id,
                        "track_id": row.track_key_v9,
                        "event_type": "pending_logodds_confirmed" if confirmed else f"pending_logodds_{match_type}_match",
                        "log_odds_before": before,
                        "log_odds_after": log_odds,
                        "hits": hits,
                        "misses": misses,
                        "age": age,
                        "last_match_iou": "",
                        "last_match_confidence": float(conf.loc[idx]),
                        "last_match_type": match_type,
                        "confirmed": confirmed,
                        "deleted": False,
                    }
                )
    return accepted, {"events": events}


def normalize_v9_name(s: str) -> str:
    mapping = {
        "s2_tnorm_temporal_logodds": "S2_tnorm_temporal_logodds",
        "s2_tnorm_temporal_logodds_maha": "S2_tnorm_temporal_logodds_maha",
        "s2_tnorm_temporal_logodds_maha_world": "S2_tnorm_temporal_logodds_maha_world",
        "s2_tnorm_temporal_logodds_maha_world_epi": "S2_tnorm_temporal_logodds_maha_world_epi",
        "s2_v9_selected": "S2_v9_selected",
    }
    return mapping.get(s.lower(), s)


def v9_flags(scenario: str) -> dict[str, bool]:
    s = scenario.lower()
    return {
        "uses_logodds": "logodds" in s or s == "s2_v9_selected",
        "uses_mahalanobis": "maha" in s or s == "s2_v9_selected",
        "uses_world_consistency": "world" in s or s == "s2_v9_selected",
        "uses_epipolar": "epi" in s,
    }


def write_ablation(main: pd.DataFrame, out: Path) -> None:
    base = main[main["scenario"].eq("S2_tnorm_temporal")]
    if base.empty:
        main.to_csv(out / "v9_ablation_summary.csv", index=False)
        return
    b = base.iloc[0]
    rows = []
    for _, row in main.iterrows():
        flags = v9_flags(str(row["scenario"]))
        rows.append(
            {
                "scenario": row["scenario"],
                **flags,
                "FP": row["FP"],
                "FN": row["FN"],
                "F1": row["F1"],
                "false_new_tracks": row["false_new_tracks"],
                "delta_FP_vs_v8_temporal": row["FP"] - b["FP"],
                "delta_FN_vs_v8_temporal": row["FN"] - b["FN"],
                "delta_F1_vs_v8_temporal": row["F1"] - b["F1"],
                "delta_false_new_vs_v8_temporal": row["false_new_tracks"] - b["false_new_tracks"],
            }
        )
    pd.DataFrame(rows).to_csv(out / "v9_ablation_summary.csv", index=False)


def write_selected(main: pd.DataFrame, out: Path, cfg: dict[str, Any]) -> None:
    candidates = main[main["scenario"].str.contains("logodds|S2_v9", case=False, na=False)].copy()
    selected_row = main[main["scenario"].eq("S2_v9_selected")]
    if not selected_row.empty:
        selected = "S2_v9_selected"
    elif candidates.empty:
        selected = "not_selected"
    else:
        feasible = candidates[(candidates["false_new_tracks"] <= 100) & (candidates["F1"] >= float(main[main["scenario"].eq("S2_tnorm_soft")]["F1"].max()))]
        if feasible.empty:
            feasible = candidates
        selected = str(feasible.sort_values(["F1", "false_new_tracks"], ascending=[False, True]).iloc[0]["scenario"])
    lines = [
        f"selection_status: {'selected' if selected != 'not_selected' else 'not_selected'}",
        f"selected_scenario: {selected}",
        "source_stage: v9_geomdyn_holdout",
        "reason: geometry_dynamic_logodds_tradeoff_selection",
    ]
    for section, values in cfg.items():
        if isinstance(values, dict):
            for key, value in values.items():
                lines.append(f"{section}_{key}: {value}")
    (out / "v9_selected_params.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_diagnostics(det: pd.DataFrame, out: Path) -> None:
    y = det["is_tp_source"].astype(bool)
    diag = pd.DataFrame(
        [
            {
                "scenario": "v9_geomdyn",
                "mean_maha_d2_tp": float(det.loc[y, "maha_d2"].mean()),
                "mean_maha_d2_fp": float(det.loc[~y, "maha_d2"].mean()),
                "mean_k_maha_tp": float(det.loc[y, "k_maha"].mean()),
                "mean_k_maha_fp": float(det.loc[~y, "k_maha"].mean()),
                "maha_available_rate": float(det["maha_available"].mean()),
                "maha_fallback_rate": float(1.0 - det["maha_available"].mean()),
            }
        ]
    )
    diag.to_csv(out / "mahalanobis_summary.csv", index=False)
    pd.DataFrame(
        [
            {
                "scenario": "v9_geomdyn",
                "world_available_rate": float(det["world_available"].mean()),
                "mean_world_error_tp": float((1.0 - det.loc[y, "s_world"]).mean()),
                "mean_world_error_fp": float((1.0 - det.loc[~y, "s_world"]).mean()),
                "mean_s_world_tp": float(det.loc[y, "s_world"].mean()),
                "mean_s_world_fp": float(det.loc[~y, "s_world"].mean()),
                "mean_support_count_tp": float(det.loc[y, "world_support_count"].mean()),
                "mean_support_count_fp": float(det.loc[~y, "world_support_count"].mean()),
                "mean_support_ratio_tp": float(det.loc[y, "world_support_ratio"].mean()),
                "mean_support_ratio_fp": float(det.loc[~y, "world_support_ratio"].mean()),
            }
        ]
    ).to_csv(out / "world_consistency_summary.csv", index=False)
    pd.DataFrame(
        [
            {
                "scenario": "v9_geomdyn",
                "epipolar_available_rate": float(det["epi_available"].mean()),
                "mean_epi_distance_tp": float(det.loc[y, "epi_distance"].mean()),
                "mean_epi_distance_fp": float(det.loc[~y, "epi_distance"].mean()),
                "mean_s_epi_tp": float(det.loc[y, "s_epi"].mean()),
                "mean_s_epi_fp": float(det.loc[~y, "s_epi"].mean()),
            }
        ]
    ).to_csv(out / "epipolar_summary.csv", index=False)
    pd.DataFrame(
        [
            {
                "scenario": "v9_geomdyn",
                "pending_created": int(len(det)),
                "pending_logodds_confirmed": "",
                "note": "See track_events.csv for per-pending log-odds events.",
            }
        ]
    ).to_csv(out / "pending_logodds_summary.csv", index=False)


def run_v9_runtime(main_csv: str | Path, out: str | Path) -> None:
    frame = pd.read_csv(main_csv)
    base = float(frame[frame["scenario"].eq("S2_tnorm_temporal")]["runtime_ms_per_frame"].iloc[0])
    rows = []
    for _, row in frame[frame["scenario"].isin(["S2_tnorm_temporal", "S2_v9_selected", "S2_learned_fp_gate"])].iterrows():
        ms = float(row["runtime_ms_per_frame"])
        rows.append({"scenario": row["scenario"], "ms_per_frame": ms, "fps": 1000.0 / max(EPS, ms), "relative_overhead_vs_v8_temporal": (ms - base) / max(EPS, base)})
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "runtime_summary.csv", index=False)


def package_v9(bundle_path: str | Path) -> None:
    root = "Defense4UAVSwarm_v9_geomdyn_bundle"
    bundle = Path(bundle_path)
    bundle.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(bundle, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(f"{root}/README_V9_GEOMDYN.md", "# Defense4UAVSwarm v9 Geometry-Dynamic Temporal Trust\n")
        for path in [
            "docs/METHOD_V9_GEOMDYN.md",
            "docs/LIMITATIONS_V9.md",
            "docs/REPRODUCE_V9.md",
            "configs/v9_geomdyn_config.yaml",
            "configs/selected_s2_temporal.yaml",
            "outputs/results/v9_geomdyn/main/main_comparison_table.csv",
            "outputs/results/v9_geomdyn/main/v9_ablation_summary.csv",
            "outputs/results/v9_geomdyn/main/v9_selected_params.yaml",
            "outputs/results/v9_geomdyn/main/rf_comparison_summary.csv",
            "outputs/results/v9_geomdyn/main/pending_logodds_summary.csv",
            "outputs/results/v9_geomdyn/main/mahalanobis_summary.csv",
            "outputs/results/v9_geomdyn/main/world_consistency_summary.csv",
            "outputs/results/v9_geomdyn/main/epipolar_summary.csv",
            "outputs/results/v9_geomdyn/robustness/pose_noise_sensitivity.csv",
            "outputs/results/v9_geomdyn/robustness/sync_delay_sensitivity.csv",
            "outputs/results/v9_geomdyn/robustness/agent_dropout_sensitivity.csv",
            "outputs/results/v9_geomdyn/robustness/combined_stress_summary.csv",
            "outputs/results/v9_geomdyn/robustness/robustness_comparison_summary.csv",
            "outputs/results/v9_geomdyn/robustness/rf_fixed_model_note.md",
            "outputs/results/v9_geomdyn/runtime/runtime_summary.csv",
            "outputs/results/v8_custom_swarm/main_holdout/main_comparison_table.csv",
            "outputs/results/v8_custom_swarm/rf_baseline/rf_holdout_summary.csv",
            "outputs/results/v9_geomdyn/reproducibility/pytest_report.txt",
            "outputs/results/v9_geomdyn/reproducibility/py_compile_report.txt",
        ]:
            p = Path(path)
            if p.exists():
                zf.write(p, f"{root}/{path}")
        try:
            git = subprocess.check_output(["git", "log", "-1", "--oneline"], text=True).strip()
        except Exception:
            git = "unavailable"
        zf.writestr(f"{root}/reproducibility/git_info.txt", git + "\n")
