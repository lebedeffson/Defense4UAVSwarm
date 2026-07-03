from __future__ import annotations

import json
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
) -> None:
    swarm_cfg = _load_yaml(swarm_config)
    root = Path(swarm_cfg["output_root"]) / split
    frame_index_path = root / "metadata" / "frame_index.csv"
    if not frame_index_path.exists():
        raise FileNotFoundError(f"Build pseudo-swarm first: missing {frame_index_path}")
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    frame_index = load_frame_index(root)
    sequences = load_split_sequences(split_config, split) or sorted(frame_index["sequence_id"].unique())
    if limit_sequences:
        sequences = sequences[:limit_sequences]
    frame_index = frame_index[frame_index["sequence_id"].isin(sequences)].copy()
    ds = VisDroneDataset(swarm_cfg["source_root"], "val", subset_hint="VID")
    gt = ds.all_annotations(sequences)
    detections = _pseudo_detections(gt, frame_index)
    features, matches = compute_inter_agent_consistency(detections, frame_index, iou_min=0.3, s_missing_policy="neutral")
    features["scenario"] = "S1_fgsm"
    features["model_name"] = "pseudo_yolo"
    features["eps"] = eps_values[0] if eps_values else 0.0
    features = add_kinematics(features)
    features["k_i"] = features["k_i_selected"]
    features["x_i"] = 1.0
    features["x_i_available"] = False
    features["xai_called"] = False
    features["xai_method"] = None
    xai_audit = pd.DataFrame()
    xai_summary = pd.DataFrame()
    if any(s.lower() in {"s3_tnorm_xai", "s3"} for s in scenarios):
        features, xai_audit, xai_summary = apply_xai_smoke(features, frame_index, xai_method, xai_max_per_frame, semantic_max_frames)
    tau_q_grid = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]
    tau_conf_grid = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60]
    rows = []
    threshold_rows = []
    feature_frames = []
    total_frames = int(frame_index[["sequence_id", "frame_id"]].drop_duplicates().shape[0])
    for scenario in scenarios:
        if scenario.lower() in {"s0", "s0_clean", "s1", "s1_fgsm"}:
            frame = features.copy()
            frame["scenario"] = "S0_clean" if scenario.lower().startswith("s0") else "S1_fgsm"
            frame["Q_i"] = frame["c_i"]
            frame["t_norm"] = None
            frame["tau_Q"] = None
            frame["accepted"] = True
            rows.append(_summary(frame, str(frame["scenario"].iloc[0]), None, None, total_frames))
            feature_frames.append(frame)
        elif scenario.lower() in {"s_naive", "snaive"}:
            for tau in tau_conf_grid:
                frame = features.copy()
                frame["scenario"] = "S_naive"
                frame["Q_i"] = frame["c_i"]
                frame["t_norm"] = "confidence"
                frame["tau_Q"] = tau
                frame["accepted"] = frame["Q_i"] >= tau
                rows.append(_summary(frame, "S_naive", "confidence", tau, total_frames))
                feature_frames.append(frame)
                threshold_rows.append({"scenario": "S_naive", "t_norm": "confidence", "tau_Q": tau, "selected": False})
        elif scenario.lower() in {"s2_tnorm_no_xai", "s2"}:
            for norm in t_norms:
                for tau in tau_q_grid:
                    frame = features.copy()
                    frame["scenario"] = "S2_tnorm_no_xai"
                    frame["x_i"] = 1.0
                    frame["t_norm"] = norm
                    frame["tau_Q"] = tau
                    frame["Q_i"] = _q(frame, norm)
                    frame["accepted"] = frame["Q_i"] >= tau
                    rows.append(_summary(frame, "S2_tnorm_no_xai", norm, tau, total_frames))
                    feature_frames.append(frame)
                    threshold_rows.append({"scenario": "S2_tnorm_no_xai", "t_norm": norm, "tau_Q": tau, "selected": False})
        elif scenario.lower() in {"s3_tnorm_xai", "s3"}:
            for norm in t_norms:
                for tau in tau_q_grid:
                    frame = features.copy()
                    frame["scenario"] = "S3_tnorm_xai"
                    frame["t_norm"] = norm
                    frame["tau_Q"] = tau
                    frame["Q_i"] = _q(frame, norm)
                    frame["accepted"] = frame["Q_i"] >= tau
                    rows.append(_summary(frame, "S3_tnorm_xai", norm, tau, total_frames))
                    feature_frames.append(frame)
                    threshold_rows.append({"scenario": "S3_tnorm_xai", "t_norm": norm, "tau_Q": tau, "selected": False})
        elif scenario.lower() in {"s3_safe_recovery", "s4_hybrid"}:
            rows.append(_not_implemented_summary("S3_safe_recovery" if scenario.lower() == "s3_safe_recovery" else "S4_hybrid", total_frames))
    audit = features.copy()
    audit["scenario"] = "S1_fgsm_base_features"
    audit["Q_i"] = audit["c_i"]
    audit["t_norm"] = None
    audit["tau_Q"] = None
    audit["accepted"] = True
    audit.to_csv(out / "swarm_feature_audit.csv", index=False)
    matches.to_csv(out / "inter_agent_matching_audit.csv", index=False)
    if len(xai_audit):
        xai_audit.to_csv(out / "xai_feature_audit.csv", index=False)
        xai_summary.to_csv(out / "xai_summary.csv", index=False)
        xai_score_auc(xai_audit).to_csv(out / "xai_score_auc.csv", index=False)
    summary = pd.DataFrame(rows)
    threshold = pd.DataFrame(threshold_rows)
    selected = select_params(summary, threshold)
    if len(threshold) and selected.get("selection_status") == "selected":
        mask = (
            (threshold["scenario"] == selected["selected_scenario"])
            & (threshold["t_norm"] == selected["selected_t_norm"])
            & (threshold["tau_Q"] == selected["selected_tau_Q"])
        )
        threshold.loc[mask, "selected"] = True
    summary.to_csv(out / "summary_metrics.csv", index=False)
    summary.to_csv(out / "research_matrix.csv", index=False)
    threshold.to_csv(out / "threshold_selection.csv", index=False)
    summary[summary["scenario"].isin(["S2_tnorm_no_xai", "S3_tnorm_xai", "S4_hybrid"])].to_csv(out / "tnorm_comparison.csv", index=False)
    build_ablation(features, total_frames).to_csv(out / "ablation_summary.csv", index=False)
    summary.to_csv(out / "robustness_summary.csv", index=False)
    write_selected_params(out / "selected_params.yaml", selected)
    (out / "metadata.json").write_text(
        json.dumps(
            {
                "stage": "v2.5_calibration",
                "dataset_type": "synthetic pseudo-swarm",
                "swarm_dataset_type": "synthetic pseudo-swarm",
                "source_dataset": "VisDrone2019-VID-val",
                "stress_transforms_enabled": "stress" in str(swarm_config),
                "limitation": "pseudo-swarm approximates multi-agent observations using transformed views of the same source frame",
                "split": split,
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
                "s3_safe_recovery_status": "not implemented for swarm_vid pseudo-swarm in v2.5",
                "s4_hybrid_status": "not implemented for swarm_vid pseudo-swarm in v2.5",
            },
            indent=2,
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


def _summary(frame: pd.DataFrame, scenario: str, norm: str | None, tau: float | None, num_frames: int) -> dict:
    total_gt = int((frame["track_id"] != -1).sum())
    accepted = frame["accepted"].astype(bool)
    tp = int((accepted & (frame["track_id"] != -1)).sum())
    fp = int((accepted & (frame["track_id"] == -1)).sum())
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
    candidates = summary[summary["scenario"].isin(["S2_tnorm_no_xai", "S3_tnorm_xai", "S4_hybrid"])].copy()
    candidates = candidates[candidates["implementation_status"].fillna("ok") == "ok"]
    naive = summary[summary["scenario"] == "S_naive"].copy()
    if candidates.empty or naive.empty:
        return {"selection_status": "not_selected", "holdout_allowed": False, "reason": "missing_candidates"}
    naive_best = naive.sort_values(["IDF1", "F1"], ascending=False).iloc[0]
    valid = candidates[
        (candidates["IDF1"] >= naive_best["IDF1"])
        & (candidates["IDSW"] <= naive_best["IDSW"])
        & (candidates["track_breaks"] <= naive_best["track_breaks"])
        & ((candidates["FP"] - naive_best["FP"]) / max(1, naive_best["FP"]) <= 0.15)
    ].copy()
    if valid.empty:
        return {
            "selection_status": "not_selected",
            "holdout_allowed": False,
            "reason": "no_candidate_beats_s_naive",
            "baseline_s_naive_IDF1": float(naive_best["IDF1"]),
        }
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
        "selected_tau_Q": float(best["tau_Q"]),
        "x_i_selected_method": "x_top5_inside",
        "xai_method": "eigencam",
        "score": float(best["score"]),
    }


def build_ablation(features: pd.DataFrame, num_frames: int) -> pd.DataFrame:
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
        row = _summary(frame, name, "min", 0.30, num_frames)
        row["variant"] = name
        row["features_used"] = ",".join(cols)
        row["interpretation"] = "pseudo-swarm calibration ablation"
        rows.append(row)
    return pd.DataFrame(rows)


def write_selected_params(path: Path, selected: dict) -> None:
    path.write_text(yaml.safe_dump(selected, sort_keys=False, allow_unicode=True), encoding="utf-8")


def _load_yaml(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}
