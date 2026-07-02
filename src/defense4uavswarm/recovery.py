from __future__ import annotations

import json
from itertools import product
from pathlib import Path

import pandas as pd
import yaml

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
    scenario_name: str = "S3_track_recovery",
    safe: bool = False,
    min_recent_confidence: float = 0.0,
    min_mean_track_confidence: float = 0.0,
    max_missed_before_recovery: int = 1,
    weak_detection_support: bool = False,
    weak_confidence_min: float = 0.01,
    weak_iou_min: float = 0.3,
    weak_lambda: float = 0.7,
    recovery_cooldown: int = 0,
    max_center_displacement_ratio: float = 1.5,
    min_bbox_inside_ratio: float = 0.7,
) -> pd.DataFrame:
    if pred.empty:
        return pred.copy()
    out_rows = []
    audit_rows = []
    states: dict[tuple[str, int], dict] = {}
    all_frames = sorted(gt[["sequence_id", "frame_id"]].drop_duplicates().itertuples(index=False, name=None))
    by_frame = {k: v for k, v in pred.groupby(["sequence_id", "frame_id"])}
    frame_bounds = _frame_bounds(gt, pred)
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
            box = _box(row)
            prev = states.get(key)
            if prev is None:
                vbox = (0.0, 0.0, 0.0, 0.0)
                age = 1
                conf_hist = []
                diag_hist = []
            else:
                last = prev["bbox"]
                vbox = tuple(box[i] - last[i] for i in range(4))
                age = int(prev["track_age"]) + 1
                conf_hist = list(prev["conf_history"])
                diag_hist = list(prev["diag_history"])
            confidence = _confidence(row)
            conf_hist.append(confidence)
            diag_hist.append(_diag(box))
            states[key] = {
                "bbox": box,
                "vbox": vbox,
                "last_confidence": confidence,
                "mean_confidence": sum(conf_hist) / max(1, len(conf_hist)),
                "conf_history": conf_hist[-10:],
                "diag_history": diag_hist[-10:],
                "class_name": row.get("class_name"),
                "class_id": row.get("class_id"),
                "track_age": age,
                "missed_frames": 0,
                "recovery_age": 0,
                "cooldown_remaining": max(0, int(prev.get("cooldown_remaining", 0)) - 1) if prev else 0,
                "last_row": row,
            }
        recovered = []
        blocked = []
        for key, state in list(states.items()):
            if key[0] != seq or key in present:
                continue
            recovery_age = int(state["recovery_age"]) + 1
            box_pred = state["bbox"] if mode == "hold_last" else tuple(state["bbox"][i] + state["vbox"][i] for i in range(4))
            reason = _block_reason(
                state,
                box_pred,
                frame,
                frame_bounds.get(seq),
                safe=safe,
                horizon=horizon,
                confirm_age=confirm_age,
                conf_floor=conf_floor,
                decay=decay,
                recovery_age=recovery_age,
                min_recent_confidence=min_recent_confidence,
                min_mean_track_confidence=min_mean_track_confidence,
                max_missed_before_recovery=max_missed_before_recovery,
                weak_detection_support=weak_detection_support,
                weak_confidence_min=weak_confidence_min,
                weak_iou_min=weak_iou_min,
                recovery_cooldown=recovery_cooldown,
                max_center_displacement_ratio=max_center_displacement_ratio,
                min_bbox_inside_ratio=min_bbox_inside_ratio,
            )
            weak = _weak_support(box_pred, frame, weak_confidence_min, weak_iou_min)
            box = _blend_box(box_pred, weak["box"], weak_lambda) if weak["has_support"] and weak_detection_support else box_pred
            metrics = _box_metrics(state["bbox"], box, frame_bounds.get(seq))
            audit = _candidate_audit_row(
                seq,
                frame_id,
                key[1],
                state,
                mode,
                horizon,
                decay,
                confirm_age,
                max_recovered_per_frame,
                recovery_age,
                box,
                metrics,
                weak,
                reason,
            )
            if reason:
                blocked.append(audit)
                states[key]["recovery_age"] = recovery_age
                states[key]["missed_frames"] = int(state.get("missed_frames", 0)) + 1
                if reason in {"horizon_exceeded", "too_many_missed", "low_recovered_confidence"} or recovery_age >= int(horizon):
                    del states[key]
                continue
            conf = float(state["last_confidence"]) * (float(decay) ** recovery_age)
            if conf < float(conf_floor):
                audit["recovery_block_reason"] = "low_recovered_confidence"
                blocked.append(audit)
                continue
            row = dict(state["last_row"])
            recovery_type = "weak_detection_refined" if weak["has_support"] and weak_detection_support else "pure_prediction"
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
                    "recovery_type": recovery_type,
                    "recovery_source_track_id": key[1],
                    "recovery_confidence": conf,
                    "recovery_horizon": horizon,
                    "recovery_decay": decay,
                    "track_age": state["track_age"],
                    "mean_track_confidence": state["mean_confidence"],
                    "bbox_inside_ratio": metrics["bbox_inside_ratio"],
                    "center_displacement_ratio": metrics["center_displacement_ratio"],
                    "bbox_area_ratio": metrics["bbox_area_ratio"],
                    "aspect_ratio_change": metrics["aspect_ratio_change"],
                    "has_weak_detection_support": weak["has_support"],
                    "weak_detection_confidence": weak["confidence"],
                    "weak_detection_iou": weak["iou"],
                    "filter_mode": "safe_track_recovery" if safe else "track_recovery",
                    "scenario": scenario_name,
                }
            )
            audit.update({"recovery_allowed": True, "recovery_block_reason": None})
            recovered.append(row)
            blocked.append(audit)
        recovered = sorted(recovered, key=lambda r: r["recovery_confidence"], reverse=True)
        allowed_keys = {(r["sequence_id"], int(r["pred_track_id"]), int(r["frame_id"])) for r in recovered[: int(max_recovered_per_frame)]}
        for row in recovered[int(max_recovered_per_frame) :]:
            for audit in blocked:
                if audit["sequence_id"] == row["sequence_id"] and audit["frame_id"] == row["frame_id"] and audit["pred_track_id"] == row["pred_track_id"]:
                    audit["recovery_allowed"] = False
                    audit["recovery_block_reason"] = "max_recovered_per_frame"
        for row in recovered[: int(max_recovered_per_frame)]:
            out_rows.append(row)
            key = (row["sequence_id"], int(row["pred_track_id"]))
            states[key]["bbox"] = _box(row)
            states[key]["recovery_age"] = int(row["recovery_age"])
            states[key]["missed_frames"] = int(row["recovery_age"])
            states[key]["cooldown_remaining"] = int(recovery_cooldown)
        audit_rows.extend(blocked)
    out = pd.DataFrame(out_rows)
    for col in template_cols:
        if col not in out:
            out[col] = None
    for col in [
        "is_recovered",
        "recovery_type",
        "mean_track_confidence",
        "bbox_inside_ratio",
        "center_displacement_ratio",
        "bbox_area_ratio",
        "aspect_ratio_change",
        "has_weak_detection_support",
        "weak_detection_confidence",
        "weak_detection_iou",
    ]:
        if col not in out:
            out[col] = False if col.startswith(("is_", "has_")) else None
    out["is_recovered"] = out["is_recovered"].map(lambda value: bool(value) if pd.notna(value) else False)
    audit = pd.DataFrame(audit_rows)
    if len(audit):
        audit = _attach_gt_recovery_labels(gt, audit)
    out.attrs["recovery_audit"] = audit
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
    scenario_name: str = "S3_track_recovery",
    safe: bool = False,
    min_recent_confidences: list[float] | None = None,
    min_mean_track_confidences: list[float] | None = None,
    recovered_conf_floors: list[float] | None = None,
    weak_detection_support_values: list[bool] | None = None,
    weak_confidence_mins: list[float] | None = None,
    weak_iou_mins: list[float] | None = None,
    recovery_cooldowns: list[int] | None = None,
) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    base_s1 = evaluate(gt, s1, "S1", eps, cfg=cfg, include_map=False, include_tracking=True)
    base_s0 = evaluate(gt, s0, "S0", 0.0, cfg=cfg, include_map=False, include_tracking=True)
    min_recent_confidences = min_recent_confidences or [0.0]
    min_mean_track_confidences = min_mean_track_confidences or [0.0]
    recovered_conf_floors = recovered_conf_floors or [0.03]
    weak_detection_support_values = weak_detection_support_values or [False]
    weak_confidence_mins = weak_confidence_mins or [0.01]
    weak_iou_mins = weak_iou_mins or [0.3]
    recovery_cooldowns = recovery_cooldowns or [0]
    rows = []
    frames: dict[tuple, pd.DataFrame] = {}
    audits: dict[tuple, pd.DataFrame] = {}
    grid = product(
        modes,
        horizons,
        decays,
        confirm_ages,
        max_recovered_values,
        recovered_conf_floors,
        min_recent_confidences,
        min_mean_track_confidences,
        weak_detection_support_values,
        weak_confidence_mins,
        weak_iou_mins,
        recovery_cooldowns,
    )
    for mode, horizon, decay, confirm_age, max_rec, conf_floor, recent_conf, mean_conf, weak_enabled, weak_conf, weak_iou, cooldown in grid:
        key = (mode, int(horizon), float(decay), int(confirm_age), int(max_rec), float(conf_floor), float(recent_conf), float(mean_conf), bool(weak_enabled), float(weak_conf), float(weak_iou), int(cooldown))
        s3 = apply_track_recovery(
            s1,
            gt,
            mode,
            int(horizon),
            float(decay),
            int(confirm_age),
            int(max_rec),
            float(conf_floor),
            scenario_name=scenario_name,
            safe=safe,
            min_recent_confidence=float(recent_conf),
            min_mean_track_confidence=float(mean_conf),
            weak_detection_support=bool(weak_enabled),
            weak_confidence_min=float(weak_conf),
            weak_iou_min=float(weak_iou),
            recovery_cooldown=int(cooldown),
        )
        s0r = apply_track_recovery(s0, gt, mode, int(horizon), float(decay), int(confirm_age), int(max_rec), float(conf_floor), scenario_name=f"{scenario_name}_clean", safe=safe, min_recent_confidence=float(recent_conf), min_mean_track_confidence=float(mean_conf), weak_detection_support=bool(weak_enabled), weak_confidence_min=float(weak_conf), weak_iou_min=float(weak_iou), recovery_cooldown=int(cooldown))
        frames[key] = s3
        audits[key] = s3.attrs.get("recovery_audit", pd.DataFrame())
        m = evaluate(gt, s3, scenario_name, eps, cfg=cfg, include_map=False, include_tracking=True)
        cm = evaluate(gt, s0r, f"{scenario_name}_clean", 0.0, cfg=cfg, include_map=False, include_tracking=False)
        audit = audits[key]
        allowed = audit[audit.get("recovery_allowed", False) == True] if len(audit) else audit
        rec_tp = int(allowed["is_TP_recovered"].sum()) if len(allowed) else 0
        rec_fp = int(allowed["is_FP_recovered"].sum()) if len(allowed) else 0
        frames_n = max(1, len(gt[["sequence_id", "frame_id"]].drop_duplicates()))
        fp_delta = m["FP"] - base_s1["FP"]
        fn_delta = m["FN"] - base_s1["FN"]
        idsw_delta = m["IDSW"] - base_s1["IDSW"]
        break_delta = m["track_breaks"] - base_s1["track_breaks"]
        fp_pct = fp_delta / max(1, base_s1["FP"])
        score = (
            0.35 * (m["IDF1"] or 0)
            - 0.20 * (m["track_breaks"] or 0) / frames_n
            - 0.15 * (m["IDSW"] or 0) / frames_n
            - 0.15 * (m["FN"] / max(1, m["TP"] + m["FN"]))
            - 0.15 * (m["FP"] / frames_n)
        )
        rows.append(
            {
                "model_name": model_name,
                "eps": eps,
                "scenario": scenario_name,
                "recovery_mode": mode,
                "recovery_horizon": horizon,
                "decay": decay,
                "recovered_conf_floor": conf_floor,
                "min_track_age": confirm_age,
                "A_confirm": confirm_age,
                "min_recent_confidence": recent_conf,
                "min_mean_track_confidence": mean_conf,
                "weak_detection_support": bool(weak_enabled),
                "weak_confidence_min": weak_conf,
                "weak_iou_min": weak_iou,
                "recovery_cooldown": cooldown,
                "max_recovered_tracks_per_frame": max_rec,
                "num_recovery_candidates": len(audit),
                "num_recovery_allowed": int(audit["recovery_allowed"].sum()) if len(audit) else 0,
                "num_recovery_blocked": int((~audit["recovery_allowed"]).sum()) if len(audit) else 0,
                "num_recovered_boxes": len(allowed),
                "recovered_TP": rec_tp,
                "recovered_FP": rec_fp,
                "recovered_TP_rate": rec_tp / max(1, len(allowed)),
                "recovered_FP_rate": rec_fp / max(1, len(allowed)),
                "TP_gain_vs_S1": m["TP"] - base_s1["TP"],
                "FP_gain_vs_S1": fp_delta,
                "FN_delta_vs_S1": fn_delta,
                "FN_reduced": -fn_delta,
                "FP_added": fp_delta,
                "IDF1": m["IDF1"],
                "IDF1_delta_vs_S1": m["IDF1"] - base_s1["IDF1"],
                "IDSW_delta_vs_S1": idsw_delta,
                "track_break_delta_vs_S1": break_delta,
                "failure_intensity_delta_vs_S1": _failure_intensity(m, frames_n) - _failure_intensity(base_s1, frames_n),
                "clean_F1_drop": base_s0["F1"] - cm["F1"],
                "FP_delta_vs_S1": fp_delta,
                "FP_delta_percent_vs_S1": fp_pct,
                "recovery_score": score,
            }
        )
    summary = pd.DataFrame(rows)
    strict = summary[
        (summary["IDF1_delta_vs_S1"] > 0)
        & (summary["track_break_delta_vs_S1"] < 0)
        & (summary["IDSW_delta_vs_S1"] <= 0)
        & (summary["FN_delta_vs_S1"] < 0)
        & (summary["FP_delta_percent_vs_S1"] <= 0.10)
        & (summary["recovered_FP_rate"] <= 0.35)
    ]
    fallback = summary[
        (summary["IDF1_delta_vs_S1"] > 0)
        & (summary["track_break_delta_vs_S1"] < 0)
        & (summary["IDSW_delta_vs_S1"] <= 0)
        & (summary["FN_delta_vs_S1"] < 0)
        & (summary["FP_delta_percent_vs_S1"] <= 0.15)
        & (summary["recovered_FP_rate"] <= 0.45)
    ]
    if len(strict):
        status = "selected"
        best = strict.sort_values("recovery_score", ascending=False).iloc[0].to_dict()
    elif len(fallback):
        status = "fallback_selected"
        best = fallback.sort_values("recovery_score", ascending=False).iloc[0].to_dict()
    else:
        status = "not_selected"
        best = summary.sort_values("recovery_score", ascending=False).iloc[0].to_dict()
    best_key = _best_key(best)
    selected_frame = frames[best_key]
    selected_audit = audits[best_key]
    save(selected_frame, results / f"vid_{model_name}_{scenario_name.lower()}_eps_{eps}.csv")
    save(selected_audit, results / "recovery_audit.csv")
    summary["selected"] = False
    summary.loc[_summary_mask(summary, best), "selected"] = status in {"selected", "fallback_selected"}
    summary["selection_status"] = status
    save(summary, results / "recovery_summary.csv")
    save(summary, results / "recovery_safety_summary.csv")
    save(_block_summary(selected_audit), results / "recovery_block_summary.csv")
    payload = {
        "selection_scope": "model_eps_level",
        "selection_status": status,
        "holdout_allowed": status in {"selected", "fallback_selected"},
        "reason": None if status in {"selected", "fallback_selected"} else "no_safe_recovery_candidate",
        "selected": {model_name: {f"eps_{eps}": best}},
    }
    (results / "selected_defense_params.yaml").write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    metadata = {"stage": "v1.7" if safe else "v1.6", "scenario": scenario_name, "holdout_run": False, "holdout_allowed": payload["holdout_allowed"], "selection_status": status}
    (results / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return selected_frame, best, summary


def write_recovery_reports(gt: pd.DataFrame, s0: pd.DataFrame, s1: pd.DataFrame, recovered: list[tuple[str, pd.DataFrame]], cfg: dict, eps: float, results: Path) -> None:
    summary = [
        evaluate(gt, s0, "S0", 0.0, cfg=cfg, include_map=False, include_tracking=True),
        evaluate(gt, s1, "S1", eps, cfg=cfg, include_map=False, include_tracking=True),
    ]
    for scenario, frame in recovered:
        summary.append(evaluate(gt, frame, scenario, eps, cfg=cfg, include_map=False, include_tracking=True))
    frame = pd.DataFrame(summary)
    save(frame, results / "summary_metrics.csv")
    save(frame, results / "summary_metrics_tracking_selected.csv")
    frames_n = max(1, len(gt[["sequence_id", "frame_id"]].drop_duplicates()))
    err = frame.copy()
    err["failures_per_100_frames"] = (err["FP"] + err["FN"] + err["IDSW"].fillna(0) + err["track_breaks"].fillna(0)) / frames_n * 100.0
    save(err, results / "error_intensity_summary.csv")


def _confidence(row: dict) -> float:
    value = row.get("confidence_original")
    if pd.isna(value):
        value = row.get("confidence", 0.0)
    return float(value)


def _box(row: dict | pd.Series) -> tuple[float, float, float, float]:
    return (float(row["x1"]), float(row["y1"]), float(row["x2"]), float(row["y2"]))


def _diag(box: tuple[float, float, float, float]) -> float:
    return max(1.0, ((box[2] - box[0]) ** 2 + (box[3] - box[1]) ** 2) ** 0.5)


def _area(box: tuple[float, float, float, float]) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def _aspect(box: tuple[float, float, float, float]) -> float:
    return max(1e-6, box[2] - box[0]) / max(1e-6, box[3] - box[1])


def _valid_box(box: tuple[float, float, float, float]) -> bool:
    x1, y1, x2, y2 = box
    return all(pd.notna(v) for v in box) and x2 > x1 and y2 > y1 and x2 > 0 and y2 > 0


def _frame_bounds(gt: pd.DataFrame, pred: pd.DataFrame) -> dict[str, tuple[float, float]]:
    both = pd.concat([gt[["sequence_id", "x2", "y2"]], pred[["sequence_id", "x2", "y2"]]], ignore_index=True)
    return {str(seq): (float(group["x2"].max()), float(group["y2"].max())) for seq, group in both.groupby("sequence_id")}


def _inside_ratio(box: tuple[float, float, float, float], bounds: tuple[float, float] | None) -> float:
    if bounds is None:
        return 1.0
    width, height = bounds
    clipped = (max(0.0, box[0]), max(0.0, box[1]), min(width, box[2]), min(height, box[3]))
    return _area(clipped) / max(1e-6, _area(box))


def _box_metrics(prev: tuple[float, float, float, float], box: tuple[float, float, float, float], bounds: tuple[float, float] | None) -> dict:
    cx0, cy0 = (prev[0] + prev[2]) / 2, (prev[1] + prev[3]) / 2
    cx1, cy1 = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    displacement = ((cx1 - cx0) ** 2 + (cy1 - cy0) ** 2) ** 0.5
    return {
        "bbox_inside_ratio": _inside_ratio(box, bounds),
        "center_displacement_ratio": displacement / _diag(prev),
        "bbox_area_ratio": _area(box) / max(1e-6, _area(prev)),
        "aspect_ratio_change": _aspect(box) / max(1e-6, _aspect(prev)),
    }


def _weak_support(box: tuple[float, float, float, float], frame: pd.DataFrame, conf_min: float, iou_min: float) -> dict:
    best = {"has_support": False, "confidence": None, "iou": 0.0, "box": None}
    for row in frame.to_dict("records"):
        conf = _confidence(row)
        if conf < conf_min:
            continue
        rb = _box(row)
        val = iou(box, rb)
        if val > float(best["iou"]):
            best = {"has_support": val >= iou_min, "confidence": conf, "iou": val, "box": rb}
    return best


def _blend_box(pred: tuple[float, float, float, float], weak: tuple[float, float, float, float] | None, weight: float) -> tuple[float, float, float, float]:
    if weak is None:
        return pred
    return tuple(float(weight) * pred[i] + (1.0 - float(weight)) * weak[i] for i in range(4))


def _block_reason(state: dict, box: tuple[float, float, float, float], frame: pd.DataFrame, bounds: tuple[float, float] | None, **kw) -> str | None:
    if int(state["track_age"]) < int(kw["confirm_age"]):
        return "not_confirmed"
    recovery_age = int(kw["recovery_age"])
    if recovery_age > int(kw["horizon"]):
        return "horizon_exceeded"
    if not _valid_box(box):
        return "motion_implausible"
    conf = float(state["last_confidence"]) * (float(kw["decay"]) ** recovery_age)
    if conf < float(kw["conf_floor"]):
        return "low_recovered_confidence"
    if not kw["safe"]:
        return None
    if float(state["last_confidence"]) < float(kw["min_recent_confidence"]):
        return "low_last_confidence"
    if float(state["mean_confidence"]) < float(kw["min_mean_track_confidence"]):
        return "low_mean_confidence"
    if int(state.get("missed_frames", 0)) > int(kw["max_missed_before_recovery"]):
        return "too_many_missed"
    if int(state.get("cooldown_remaining", 0)) > 0 and int(state.get("recovery_age", 0)) == 0:
        return "cooldown_active"
    metrics = _box_metrics(state["bbox"], box, bounds)
    if metrics["bbox_inside_ratio"] < float(kw["min_bbox_inside_ratio"]):
        return "bbox_out_of_frame"
    if metrics["center_displacement_ratio"] > float(kw["max_center_displacement_ratio"]):
        return "motion_implausible"
    if not 0.5 <= metrics["bbox_area_ratio"] <= 2.0 or not 0.5 <= metrics["aspect_ratio_change"] <= 2.0:
        return "motion_implausible"
    if kw["weak_detection_support"] and not _weak_support(box, frame, kw["weak_confidence_min"], kw["weak_iou_min"])["has_support"]:
        return "no_weak_support"
    return None


def _candidate_audit_row(seq: str, frame_id: int, pred_track_id: int, state: dict, mode: str, horizon: int, decay: float, confirm_age: int, max_rec: int, recovery_age: int, box: tuple[float, float, float, float], metrics: dict, weak: dict, reason: str | None) -> dict:
    return {
        "sequence_id": seq,
        "frame_id": frame_id,
        "pred_track_id": pred_track_id,
        "recovery_mode": mode,
        "recovery_type": "weak_detection_refined" if weak["has_support"] else "pure_prediction",
        "recovery_horizon": horizon,
        "decay": decay,
        "A_confirm": confirm_age,
        "max_recovered_tracks_per_frame": max_rec,
        "recovery_age": recovery_age,
        "track_age": state["track_age"],
        "last_confidence": state["last_confidence"],
        "mean_track_confidence": state["mean_confidence"],
        "missed_frames": state.get("missed_frames", 0),
        "bbox_x1": box[0],
        "bbox_y1": box[1],
        "bbox_x2": box[2],
        "bbox_y2": box[3],
        "confidence_recovered": state["last_confidence"] * (decay ** recovery_age),
        "bbox_inside_ratio": metrics["bbox_inside_ratio"],
        "center_displacement_ratio": metrics["center_displacement_ratio"],
        "bbox_area_ratio": metrics["bbox_area_ratio"],
        "aspect_ratio_change": metrics["aspect_ratio_change"],
        "has_weak_detection_support": weak["has_support"],
        "weak_detection_confidence": weak["confidence"],
        "weak_detection_iou": weak["iou"],
        "recovery_allowed": reason is None,
        "recovery_block_reason": reason,
    }


def _attach_gt_recovery_labels(gt: pd.DataFrame, audit: pd.DataFrame) -> pd.DataFrame:
    gt_frames = {k: v for k, v in gt.groupby(["sequence_id", "frame_id"])}
    rows = []
    for row in audit.to_dict("records"):
        gt_f = gt_frames.get((row["sequence_id"], row["frame_id"]), gt.iloc[0:0])
        best_iou, best_gt = 0.0, None
        for _, g in gt_f.iterrows():
            val = iou((row["bbox_x1"], row["bbox_y1"], row["bbox_x2"], row["bbox_y2"]), (g.x1, g.y1, g.x2, g.y2))
            if val > best_iou:
                best_iou, best_gt = val, g
        is_tp = best_iou >= 0.5
        row.update(
            {
                "matched_gt_track_id": None if best_gt is None else best_gt.get("gt_track_id"),
                "matched_gt_iou": best_iou,
                "is_TP_recovered": is_tp,
                "is_FP_recovered": not is_tp,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _block_summary(audit: pd.DataFrame) -> pd.DataFrame:
    if audit.empty or "recovery_block_reason" not in audit:
        return pd.DataFrame(columns=["block_reason", "count", "would_have_been_TP", "would_have_been_FP", "blocked_FP_rate", "blocked_TP_rate"])
    blocked = audit[audit["recovery_allowed"] == False].copy()
    if blocked.empty:
        return pd.DataFrame(columns=["block_reason", "count", "would_have_been_TP", "would_have_been_FP", "blocked_FP_rate", "blocked_TP_rate"])
    rows = []
    for reason, group in blocked.groupby("recovery_block_reason"):
        tp = int(group["is_TP_recovered"].sum())
        fp = int(group["is_FP_recovered"].sum())
        rows.append({"block_reason": reason, "count": len(group), "would_have_been_TP": tp, "would_have_been_FP": fp, "blocked_FP_rate": fp / max(1, len(group)), "blocked_TP_rate": tp / max(1, len(group))})
    return pd.DataFrame(rows)


def _best_key(best: dict) -> tuple:
    return (
        best["recovery_mode"],
        int(best["recovery_horizon"]),
        float(best["decay"]),
        int(best["A_confirm"]),
        int(best["max_recovered_tracks_per_frame"]),
        float(best["recovered_conf_floor"]),
        float(best["min_recent_confidence"]),
        float(best["min_mean_track_confidence"]),
        bool(best["weak_detection_support"]),
        float(best["weak_confidence_min"]),
        float(best["weak_iou_min"]),
        int(best["recovery_cooldown"]),
    )


def _summary_mask(summary: pd.DataFrame, best: dict) -> pd.Series:
    return (
        (summary["recovery_mode"] == best["recovery_mode"])
        & (summary["recovery_horizon"] == best["recovery_horizon"])
        & (summary["decay"] == best["decay"])
        & (summary["A_confirm"] == best["A_confirm"])
        & (summary["max_recovered_tracks_per_frame"] == best["max_recovered_tracks_per_frame"])
        & (summary["recovered_conf_floor"] == best["recovered_conf_floor"])
        & (summary["min_recent_confidence"] == best["min_recent_confidence"])
        & (summary["min_mean_track_confidence"] == best["min_mean_track_confidence"])
        & (summary["weak_detection_support"] == best["weak_detection_support"])
        & (summary["weak_confidence_min"] == best["weak_confidence_min"])
        & (summary["weak_iou_min"] == best["weak_iou_min"])
        & (summary["recovery_cooldown"] == best["recovery_cooldown"])
    )


def _failure_intensity(m: dict, frames: int) -> float:
    return (m["FP"] + m["FN"] + (m["IDSW"] or 0) + (m["track_breaks"] or 0)) / frames * 100.0
