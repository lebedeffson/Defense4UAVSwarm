from __future__ import annotations

import json
from itertools import product
from pathlib import Path

import pandas as pd
import yaml

from defense4uavswarm.metrics.detection import match_frame
from defense4uavswarm.pipeline import evaluate, save
from defense4uavswarm.tracking.simple import iou


def apply_track_recovery(
    pred: pd.DataFrame,
    gt: pd.DataFrame,
    mode: str = "constant_velocity",
    horizon: int = 3,
    decay: float = 0.8,
    confirm_age: int = 3,
    max_recovered_per_frame: int = 10,
    conf_floor: float = 0.03,
) -> pd.DataFrame:
    if pred.empty:
        return pred.copy()
    out_rows = []
    states: dict[tuple[str, int], dict] = {}
    all_frames = sorted(gt[["sequence_id", "frame_id"]].drop_duplicates().itertuples(index=False, name=None))
    by_frame = {k: v for k, v in pred.groupby(["sequence_id", "frame_id"])}
    template_cols = list(pred.columns)
    for seq, frame_id in all_frames:
        frame = by_frame.get((seq, frame_id), pred.iloc[0:0])
        present = set()
        for row in frame.to_dict("records"):
            out_rows.append(row)
            tid = row.get("pred_track_id")
            if pd.isna(tid):
                continue
            tid = int(tid)
            key = (seq, tid)
            present.add(key)
            box = (float(row["x1"]), float(row["y1"]), float(row["x2"]), float(row["y2"]))
            prev = states.get(key)
            if prev is None:
                vbox = (0.0, 0.0, 0.0, 0.0)
                age = 1
            else:
                last = prev["bbox"]
                vbox = tuple(box[i] - last[i] for i in range(4))
                age = int(prev["track_age"]) + 1
            confidence = row.get("confidence_original")
            if pd.isna(confidence):
                confidence = row.get("confidence", 0.0)
            states[key] = {
                "bbox": box,
                "vbox": vbox,
                "last_confidence": float(confidence),
                "class_name": row.get("class_name"),
                "class_id": row.get("class_id"),
                "track_age": age,
                "missed_frames": 0,
                "recovery_age": 0,
                "last_row": row,
            }
        recovered = []
        for key, state in list(states.items()):
            if key[0] != seq or key in present:
                continue
            if int(state["track_age"]) < int(confirm_age):
                continue
            recovery_age = int(state["recovery_age"]) + 1
            if recovery_age > int(horizon):
                continue
            box = state["bbox"] if mode == "hold_last" else tuple(state["bbox"][i] + state["vbox"][i] for i in range(4))
            if not _valid_box(box):
                continue
            conf = max(float(conf_floor), float(state["last_confidence"]) * (float(decay) ** recovery_age))
            if conf <= float(conf_floor):
                continue
            row = dict(state["last_row"])
            row.update(
                {
                    "frame_id": frame_id,
                    "x1": box[0],
                    "y1": box[1],
                    "x2": box[2],
                    "y2": box[3],
                    "confidence": conf,
                    "confidence_original": state["last_confidence"],
                    "c_i": conf,
                    "accepted": True,
                    "is_recovered": True,
                    "recovery_age": recovery_age,
                    "recovery_mode": mode,
                    "recovery_source_track_id": key[1],
                    "recovery_confidence": conf,
                    "recovery_horizon": horizon,
                    "recovery_decay": decay,
                    "filter_mode": "track_recovery",
                    "scenario": "S3_track_recovery",
                }
            )
            recovered.append(row)
        recovered = sorted(recovered, key=lambda r: r["recovery_confidence"], reverse=True)[: int(max_recovered_per_frame)]
        for row in recovered:
            out_rows.append(row)
            key = (row["sequence_id"], int(row["pred_track_id"]))
            states[key]["bbox"] = (row["x1"], row["y1"], row["x2"], row["y2"])
            states[key]["recovery_age"] = int(row["recovery_age"])
            states[key]["missed_frames"] = int(row["recovery_age"])
    out = pd.DataFrame(out_rows)
    for col in template_cols:
        if col not in out:
            out[col] = None
    if "is_recovered" not in out:
        out["is_recovered"] = False
    out["is_recovered"] = out["is_recovered"].map(lambda value: bool(value) if pd.notna(value) else False)
    return out


def run_track_recovery_calibration(
    gt: pd.DataFrame,
    s0: pd.DataFrame,
    s1: pd.DataFrame,
    cfg: dict,
    model_name: str,
    eps: float,
    results: Path,
    modes: list[str],
    horizons: list[int],
    decays: list[float],
    confirm_ages: list[int],
    max_recovered_values: list[int],
) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    base_s1 = evaluate(gt, s1, "S1", eps, cfg=cfg, include_map=False, include_tracking=True)
    base_s0 = evaluate(gt, s0, "S0", 0.0, cfg=cfg, include_map=False, include_tracking=True)
    rows = []
    frames: dict[tuple, pd.DataFrame] = {}
    clean_frames: dict[tuple, pd.DataFrame] = {}
    for mode, horizon, decay, confirm_age, max_rec in product(modes, horizons, decays, confirm_ages, max_recovered_values):
        key = (mode, int(horizon), float(decay), int(confirm_age), int(max_rec))
        s3 = apply_track_recovery(s1, gt, mode, int(horizon), float(decay), int(confirm_age), int(max_rec))
        s0r = apply_track_recovery(s0, gt, mode, int(horizon), float(decay), int(confirm_age), int(max_rec))
        frames[key] = s3
        clean_frames[key] = s0r
        m = evaluate(gt, s3, "S3_track_recovery", eps, cfg=cfg, include_map=False, include_tracking=True)
        cm = evaluate(gt, s0r, "S3_clean_recovery", 0.0, cfg=cfg, include_map=False, include_tracking=False)
        audit = recovery_audit(gt, s3, model_name, eps, mode, int(horizon), float(decay), int(confirm_age), int(max_rec))
        rec_tp = int(audit["is_TP_recovered"].sum()) if len(audit) else 0
        rec_fp = int(audit["is_FP_recovered"].sum()) if len(audit) else 0
        frames_n = max(1, len(gt[["sequence_id", "frame_id"]].drop_duplicates()))
        failure_delta = _failure_intensity(m, frames_n) - _failure_intensity(base_s1, frames_n)
        clean_drop = base_s0["F1"] - cm["F1"]
        fp_delta = m["FP"] - base_s1["FP"]
        fn_delta = m["FN"] - base_s1["FN"]
        idsw_delta = m["IDSW"] - base_s1["IDSW"]
        break_delta = m["track_breaks"] - base_s1["track_breaks"]
        score = (
            0.35 * (m["IDF1"] or 0)
            - 0.20 * (m["track_breaks"] or 0) / frames_n
            - 0.15 * (m["IDSW"] or 0) / frames_n
            - 0.15 * (m["FN"] / max(1, m["TP"] + m["FN"]))
            - 0.10 * (m["FP"] / frames_n)
        )
        rows.append(
            {
                "model_name": model_name,
                "eps": eps,
                "recovery_mode": mode,
                "recovery_horizon": horizon,
                "decay": decay,
                "A_confirm": confirm_age,
                "max_recovered_tracks_per_frame": max_rec,
                "num_recovered_boxes": len(audit),
                "recovered_TP": rec_tp,
                "recovered_FP": rec_fp,
                "recovered_TP_rate": rec_tp / max(1, len(audit)),
                "recovered_FP_rate": rec_fp / max(1, len(audit)),
                "FN_reduced": -fn_delta,
                "FP_added": fp_delta,
                "IDF1": m["IDF1"],
                "IDF1_delta_vs_S1": m["IDF1"] - base_s1["IDF1"],
                "IDSW_delta_vs_S1": idsw_delta,
                "track_break_delta_vs_S1": break_delta,
                "failure_intensity_delta_vs_S1": failure_delta,
                "clean_F1_drop": clean_drop,
                "FP_delta_vs_S1": fp_delta,
                "FN_delta_vs_S1": fn_delta,
                "recovery_score": score,
            }
        )
    summary = pd.DataFrame(rows)
    strict = summary[
        (summary["clean_F1_drop"] <= 0.02)
        & (summary["FP_delta_vs_S1"] <= base_s1["FP"] * 0.10)
        & (summary["IDSW_delta_vs_S1"] <= 0)
        & (summary["track_break_delta_vs_S1"] <= 0)
        & (summary["recovered_FP_rate"] <= 0.50)
    ]
    fallback = summary[
        (summary["FP_delta_vs_S1"] <= base_s1["FP"] * 0.15)
        & (summary["IDSW_delta_vs_S1"] <= 2)
        & (summary["track_break_delta_vs_S1"] <= 0)
    ]
    if len(strict):
        status = "selected"
        best = strict.sort_values("recovery_score", ascending=False).iloc[0].to_dict()
    elif len(fallback):
        status = "fallback_recovery_tolerance"
        best = fallback.sort_values("recovery_score", ascending=False).iloc[0].to_dict()
    else:
        status = "not_selected"
        best = summary.sort_values("recovery_score", ascending=False).iloc[0].to_dict()
    best_key = (best["recovery_mode"], int(best["recovery_horizon"]), float(best["decay"]), int(best["A_confirm"]), int(best["max_recovered_tracks_per_frame"]))
    selected_frame = frames[best_key]
    save(selected_frame, results / f"vid_{model_name}_s3_track_recovery_eps_{eps}.csv")
    audit = recovery_audit(gt, selected_frame, model_name, eps, best["recovery_mode"], int(best["recovery_horizon"]), float(best["decay"]), int(best["A_confirm"]), int(best["max_recovered_tracks_per_frame"]))
    save(audit, results / "recovery_audit.csv")
    summary["selected"] = False
    mask = (
        (summary["recovery_mode"] == best["recovery_mode"])
        & (summary["recovery_horizon"] == best["recovery_horizon"])
        & (summary["decay"] == best["decay"])
        & (summary["A_confirm"] == best["A_confirm"])
        & (summary["max_recovered_tracks_per_frame"] == best["max_recovered_tracks_per_frame"])
    )
    summary.loc[mask, "selected"] = status == "selected"
    summary["selection_status"] = status
    save(summary, results / "recovery_summary.csv")
    payload = {
        "selection_scope": "model_eps_level",
        "selection_status": status,
        "holdout_allowed": status == "selected",
        "reason": None if status == "selected" else "no_safe_recovery_candidate",
        "selected": {model_name: {f"eps_{eps}": best}},
    }
    (results / "selected_defense_params.yaml").write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    metadata = {"stage": "v1.6", "scenario": "S3_track_recovery", "holdout_run": False, "holdout_allowed": status == "selected", "selection_status": status}
    (results / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return selected_frame, best, summary


def recovery_audit(gt: pd.DataFrame, pred: pd.DataFrame, model_name: str, eps: float, mode: str, horizon: int, decay: float, confirm_age: int, max_rec: int) -> pd.DataFrame:
    rec = pred[pred.get("is_recovered", False) == True] if "is_recovered" in pred else pred.iloc[0:0]
    rows = []
    gt_frames = {k: v for k, v in gt.groupby(["sequence_id", "frame_id"])}
    for idx, row in rec.iterrows():
        gt_f = gt_frames.get((row["sequence_id"], row["frame_id"]), gt.iloc[0:0])
        best_iou, best_gt = 0.0, None
        for _, g in gt_f.iterrows():
            val = iou((row.x1, row.y1, row.x2, row.y2), (g.x1, g.y1, g.x2, g.y2))
            if val > best_iou:
                best_iou, best_gt = val, g
        is_tp = best_iou >= 0.5
        rows.append(
            {
                "sequence_id": row["sequence_id"],
                "frame_id": row["frame_id"],
                "pred_track_id": row["pred_track_id"],
                "model_name": model_name,
                "eps": eps,
                "recovery_mode": mode,
                "recovery_horizon": horizon,
                "decay": decay,
                "A_confirm": confirm_age,
                "max_recovered_tracks_per_frame": max_rec,
                "recovery_age": row.get("recovery_age"),
                "bbox_x1": row.x1,
                "bbox_y1": row.y1,
                "bbox_x2": row.x2,
                "bbox_y2": row.y2,
                "confidence_recovered": row.get("recovery_confidence", row.get("confidence")),
                "matched_gt_track_id": None if best_gt is None else best_gt.get("gt_track_id"),
                "matched_gt_iou": best_iou,
                "is_TP_recovered": is_tp,
                "is_FP_recovered": not is_tp,
            }
        )
    return pd.DataFrame(rows)


def write_recovery_reports(gt: pd.DataFrame, s0: pd.DataFrame, s1: pd.DataFrame, s3: pd.DataFrame, cfg: dict, eps: float, results: Path) -> None:
    summary = [
        evaluate(gt, s0, "S0", 0.0, cfg=cfg, include_map=False, include_tracking=True),
        evaluate(gt, s1, "S1", eps, cfg=cfg, include_map=False, include_tracking=True),
        evaluate(gt, s3, "S3_track_recovery", eps, cfg=cfg, include_map=False, include_tracking=True),
    ]
    frame = pd.DataFrame(summary)
    save(frame, results / "summary_metrics.csv")
    save(frame, results / "summary_metrics_tracking_selected.csv")
    frames_n = max(1, len(gt[["sequence_id", "frame_id"]].drop_duplicates()))
    err = frame.copy()
    err["failures_per_100_frames"] = (err["FP"] + err["FN"] + err["IDSW"].fillna(0) + err["track_breaks"].fillna(0)) / frames_n * 100.0
    save(err, results / "error_intensity_summary.csv")


def _valid_box(box: tuple[float, float, float, float]) -> bool:
    x1, y1, x2, y2 = box
    return all(pd.notna(v) for v in box) and x2 > x1 and y2 > y1 and x2 > 0 and y2 > 0


def _failure_intensity(m: dict, frames: int) -> float:
    return (m["FP"] + m["FN"] + (m["IDSW"] or 0) + (m["track_breaks"] or 0)) / frames * 100.0
