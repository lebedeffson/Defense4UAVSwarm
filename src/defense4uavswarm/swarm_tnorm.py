from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yaml

from defense4uavswarm.datasets.visdrone import VISDRONE_CLASSES, VisDroneDataset
from defense4uavswarm.matrix import load_split_sequences
from defense4uavswarm.swarm import compute_inter_agent_consistency, load_frame_index, transform_bbox
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
    features["k_i"] = 1.0
    features["x_i"] = 1.0
    features["x_i_available"] = False
    features["xai_called"] = False
    features["xai_method"] = None
    xai_audit = pd.DataFrame()
    xai_summary = pd.DataFrame()
    if any(s.lower() in {"s3_tnorm_xai", "s3"} for s in scenarios):
        features, xai_audit, xai_summary = apply_xai_smoke(features, frame_index, xai_method, xai_max_per_frame, semantic_max_frames)
    rows = []
    threshold_rows = []
    feature_frames = []
    for scenario in scenarios:
        if scenario.lower() in {"s0", "s0_clean", "s1", "s1_fgsm"}:
            frame = features.copy()
            frame["scenario"] = "S0_clean" if scenario.lower().startswith("s0") else "S1_fgsm"
            frame["Q_i"] = frame["c_i"]
            frame["t_norm"] = None
            frame["tau_Q"] = None
            frame["accepted"] = True
            rows.append(_summary(frame, str(frame["scenario"].iloc[0]), None, None))
            feature_frames.append(frame)
        elif scenario.lower() in {"s_naive", "snaive"}:
            frame = features.copy()
            frame["scenario"] = "S_naive"
            frame["Q_i"] = frame["c_i"]
            frame["t_norm"] = "confidence"
            frame["tau_Q"] = 0.5
            frame["accepted"] = frame["Q_i"] >= 0.5
            rows.append(_summary(frame, "S_naive", "confidence", 0.5))
            feature_frames.append(frame)
            threshold_rows.append({"scenario": "S_naive", "t_norm": "confidence", "tau_Q": 0.5, "selected": True})
        elif scenario.lower() in {"s2_tnorm_no_xai", "s2"}:
            for norm in t_norms:
                frame = features.copy()
                frame["scenario"] = "S2_tnorm_no_xai"
                frame["t_norm"] = norm
                frame["tau_Q"] = 0.3
                frame["Q_i"] = _q(frame, norm)
                frame["accepted"] = frame["Q_i"] >= 0.3
                rows.append(_summary(frame, "S2_tnorm_no_xai", norm, 0.3))
                feature_frames.append(frame)
                threshold_rows.append({"scenario": "S2_tnorm_no_xai", "t_norm": norm, "tau_Q": 0.3, "selected": True})
        elif scenario.lower() in {"s3_tnorm_xai", "s3"}:
            for norm in t_norms:
                frame = features.copy()
                frame["scenario"] = "S3_tnorm_xai"
                frame["t_norm"] = norm
                frame["tau_Q"] = 0.3
                frame["Q_i"] = _q(frame, norm)
                frame["accepted"] = frame["Q_i"] >= 0.3
                rows.append(_summary(frame, "S3_tnorm_xai", norm, 0.3))
                feature_frames.append(frame)
                threshold_rows.append({"scenario": "S3_tnorm_xai", "t_norm": norm, "tau_Q": 0.3, "selected": True})
    audit = pd.concat(feature_frames, ignore_index=True) if feature_frames else features
    audit.to_csv(out / "swarm_feature_audit.csv", index=False)
    matches.to_csv(out / "inter_agent_matching_audit.csv", index=False)
    if len(xai_audit):
        xai_audit.to_csv(out / "xai_feature_audit.csv", index=False)
        xai_summary.to_csv(out / "xai_summary.csv", index=False)
    pd.DataFrame(rows).to_csv(out / "summary_metrics.csv", index=False)
    pd.DataFrame(rows).to_csv(out / "research_matrix.csv", index=False)
    pd.DataFrame(threshold_rows).to_csv(out / "threshold_selection.csv", index=False)
    pd.DataFrame([r for r in rows if r["scenario"] == "S2_tnorm_no_xai"]).to_csv(out / "tnorm_comparison.csv", index=False)
    (out / "metadata.json").write_text(
        json.dumps(
            {
                "stage": "v2.3_xai_smoke" if len(xai_audit) else "v2_smoke_s2",
                "dataset_type": "pseudo_swarm",
                "split": split,
                "num_agents": int(frame_index["agent_id"].nunique()),
                "num_synchronized_frames": int(frame_index.groupby(["sequence_id", "frame_id"])["agent_id"].nunique().eq(frame_index["agent_id"].nunique()).sum()),
                "xai_enabled": bool(len(xai_audit)),
                "xai_method_requested": xai_method,
                "xai_method_used": xai_method if len(xai_audit) else None,
                "gradcam_status": "not_attempted_or_failed" if xai_method == "eigencam" else "fallback_not_used",
                "xai_candidate_policy": {"c_low": 0.5, "sigma_k": 0.3, "sigma_s": 0.3, "tau_pre": 0.3, "M_xai": xai_max_per_frame},
                "x_i_formula": "sum(CAM inside bbox) / sum(CAM over image)",
                "x_i_neutral_when_not_called": 1.0,
                "s_missing_policy": "neutral",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"output: {out}")
    print(f"mean_s_i: {features['s_i'].mean():.4f}")
    print(f"s2_rejected: {int((audit[audit['scenario'] == 'S2_tnorm_no_xai']['accepted'] == False).sum())}")
    if len(xai_summary):
        print(f"num_xai_called: {int(xai_summary.iloc[0]['num_xai_called'])}")
        print(f"mean_x_i: {float(xai_summary.iloc[0]['mean_x_i']):.4f}")


def apply_xai_smoke(features: pd.DataFrame, frame_index: pd.DataFrame, method: str, max_per_frame: int, max_frames: int | None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    out = features.copy()
    out["Q_pre"] = out[["c_i", "k_i", "s_i"]].min(axis=1)
    out["risk"] = (1.0 - out["c_i"]) + (1.0 - out["k_i"]) + (1.0 - out["s_i"])
    image_paths = frame_index.set_index(["sequence_id", "frame_id", "agent_id"])["image_path"].to_dict()
    audit_rows = []
    frame_count = 0
    for key, group in out.groupby(["sequence_id", "frame_id", "agent_id"], sort=False):
        if max_frames is not None and frame_count >= max_frames:
            break
        candidates = select_xai_candidates(group, max_per_frame=max_per_frame)
        if candidates.empty:
            candidates = group.sort_values("risk", ascending=False).head(1)
        if candidates.empty:
            continue
        cam, latency = compute_cam(image_paths[key], method=method)
        for rank, (idx, row) in enumerate(candidates.iterrows(), start=1):
            x_i, inside, total = compute_xai_score(cam, (row.x1, row.y1, row.x2, row.y2))
            available = x_i is not None
            value = float(x_i) if available else 1.0
            out.loc[idx, "x_i"] = value
            out.loc[idx, "x_i_available"] = available
            out.loc[idx, "xai_called"] = True
            out.loc[idx, "xai_method"] = method
            audit_rows.append(
                {
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
                    "x_i": value,
                    "xai_energy_inside_bbox": inside,
                    "xai_total_energy": total,
                    "xai_latency_ms": latency,
                    "is_TP": row.track_id != -1,
                    "is_FP": row.track_id == -1,
                    "matched_gt_id": row.track_id if row.track_id != -1 else None,
                    "matched_gt_iou": 1.0 if row.track_id != -1 else 0.0,
                }
            )
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
                "share_x_i_lt_0_3": float((available["x_i"] < 0.3).mean()) if len(available) else 0.0,
                "share_x_i_lt_0_5": float((available["x_i"] < 0.5).mean()) if len(available) else 0.0,
                "mean_xai_latency_ms": float(audit["xai_latency_ms"].mean()) if len(audit) else 0.0,
                "xai_calls_per_frame": len(called) / max(1, frame_count),
            }
        ]
    )
    return out, audit, summary


def _pseudo_detections(gt: pd.DataFrame, frame_index: pd.DataFrame) -> pd.DataFrame:
    transforms = frame_index.set_index(["sequence_id", "frame_id", "agent_id"])["transform_from_reference_matrix"].to_dict()
    rows = []
    det_id = 0
    for frame_row in frame_index[["sequence_id", "frame_id", "agent_id"]].drop_duplicates().itertuples(index=False):
        frame_gt = gt[(gt.sequence_id == frame_row.sequence_id) & (gt.frame_id == frame_row.frame_id)]
        matrix = transforms[(frame_row.sequence_id, frame_row.frame_id, frame_row.agent_id)]
        for g in frame_gt.itertuples(index=False):
            box = transform_bbox((g.x1, g.y1, g.x2, g.y2), matrix)
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


def _summary(frame: pd.DataFrame, scenario: str, norm: str | None, tau: float | None) -> dict:
    return {"scenario": scenario, "t_norm": norm, "tau_Q": tau, "num_detections_before_filter": len(frame), "num_detections_after_filter": int(frame["accepted"].sum()), "num_rejected": int((~frame["accepted"]).sum()), "rejection_rate": float((~frame["accepted"]).mean()), "mean_s_i": float(frame["s_i"].mean()), "share_s_i_low": float((frame["s_i"] < 0.3).mean()), "mean_Q_i": float(frame["Q_i"].mean())}


def _load_yaml(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}
