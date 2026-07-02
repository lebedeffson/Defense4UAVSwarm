from __future__ import annotations

import math

import pandas as pd

from defense4uavswarm.tracking.simple import iou


def t_norm(name: str, c: float, k: float, s: float = 1.0, x: float = 1.0) -> float:
    key = name.lower().replace("t_", "")
    vals = [max(0.0, min(1.0, v)) for v in (c, k, s, x)]
    if key == "min":
        return min(vals)
    if key == "prod":
        q = 1.0
        for v in vals:
            q *= v
        return q
    if key == "lukasiewicz":
        return max(0.0, sum(vals) - len(vals) + 1)
    raise ValueError(f"Unknown T-norm: {name}")


def _kinematics_value(row, k_variant: str, alpha_scale: float) -> float:
    if k_variant == "iou" and hasattr(row, "k_iou"):
        return float(row.k_iou)
    if k_variant == "gate" and hasattr(row, "d_center"):
        alpha = max(1.0, float(getattr(row, "alpha_base", 50.0)) * alpha_scale)
        r = float(row.d_center) / alpha
        return 1.0 / (1.0 + r * r)
    if k_variant == "combined" and hasattr(row, "k_iou") and hasattr(row, "d_center"):
        alpha = max(1.0, float(getattr(row, "alpha_base", 50.0)) * alpha_scale)
        center = max(0.0, min(1.0, math.exp(-float(row.d_center) / alpha)))
        iou_val = max(1e-6, min(1.0, float(row.k_iou)))
        return (center * iou_val) ** 0.5
    if k_variant == "robust_min" and hasattr(row, "k_iou") and hasattr(row, "k_acc"):
        return min(float(row.k_center), float(row.k_iou), float(row.k_acc))
    if hasattr(row, "d_center"):
        alpha = max(1.0, float(getattr(row, "alpha_base", 50.0)) * alpha_scale)
        return math.exp(-float(row.d_center) / alpha)
    return float(row.k_i)


def apply_tnorm(
    df,
    name: str,
    tau: float,
    mode: str = "hard",
    k_variant: str = "center",
    alpha_scale: float = 1.0,
    beta: float | None = None,
    preassociation: bool = False,
    gamma_assoc: float = 0.2,
    k_new_track: float = 1.0,
    tau_soft_update: float | None = None,
    max_missed_frames: int = 5,
    soft_conf_floor: float = 0.0,
    reject_patience: int = 1,
    tau_existing: float | None = None,
    tau_new: float | None = None,
    confirm_age: int = 3,
    max_confirm_missed: int = 1,
    confirmed_conf_floor: float = 0.03,
    risk_tau: float = 0.7,
    new_conf_tau: float = 0.2,
    risk_weights: tuple[float, float, float] = (0.4, 0.3, 0.3),
):
    if preassociation:
        return _apply_tnorm_preassociation(
            df,
            name,
            tau,
            mode=mode,
            k_variant=k_variant,
            alpha_scale=alpha_scale,
            beta=beta,
            gamma_assoc=gamma_assoc,
            k_new_track=k_new_track,
            tau_soft_update=tau if tau_soft_update is None else tau_soft_update,
            max_missed_frames=max_missed_frames,
            soft_conf_floor=soft_conf_floor,
            reject_patience=reject_patience,
            tau_existing=tau_existing,
            tau_new=tau_new,
            confirm_age=confirm_age,
            max_confirm_missed=max_confirm_missed,
            confirmed_conf_floor=confirmed_conf_floor,
            risk_tau=risk_tau,
            new_conf_tau=new_conf_tau,
            risk_weights=risk_weights,
        )

    out = df.copy()
    mode = {"hard_filter": "hard", "soft_reweight": "soft"}.get(mode, mode)
    out["k_variant"] = k_variant
    out["alpha_scale"] = alpha_scale
    out["beta"] = beta
    out["k_i"] = [_kinematics_value(r, k_variant, alpha_scale) for r in out.itertuples()]
    out["Q_raw"] = [t_norm(name, float(r.c_i), float(r.k_i), float(r.s_i), float(r.x_i)) for r in out.itertuples()]
    out["Q_smooth"] = out["Q_raw"]
    if beta is not None and len(out):
        out = out.sort_values(["sequence_id", "pred_track_id", "frame_id"], na_position="last").copy()
        smoothed = []
        for _, group in out.groupby(["sequence_id", "pred_track_id"], dropna=False, sort=False):
            prev = None
            for value in group["Q_raw"].astype(float):
                current = value if prev is None else beta * value + (1.0 - beta) * prev
                smoothed.append(current)
                prev = current
        out["Q_smooth"] = smoothed
        out = out.sort_index()
    out["Q_i"] = out["Q_smooth"]
    out["t_norm"] = name
    out["tau"] = tau
    out["filter_mode"] = "soft_reweight" if mode == "soft" else "hard_filter"
    out["tau_existing"] = tau if tau_existing is None else tau_existing
    out["tau_new"] = tau if tau_new is None else tau_new
    out["confirmed_conf_floor"] = confirmed_conf_floor
    out["confirm_age"] = confirm_age
    out["risk_tau"] = risk_tau
    out["new_conf_tau"] = new_conf_tau
    if "track_status" not in out:
        out["track_status"] = [
            _status_from_existing_row(r, int(confirm_age), int(max_confirm_missed))
            for r in out.itertuples()
        ]
    out["missed_frames"] = out.get("missed_frames", 0)
    out["assoc_score"] = out.get("assoc_score", out["k_i"])
    out["suspicious"] = out["Q_i"] < out["tau_existing"]
    out["filter_action"] = "keep"
    if mode == "soft":
        out["confidence_original"] = out["confidence"]
        out["confidence"] = out["confidence"].astype(float) * out["Q_i"].astype(float)
        out["c_i"] = out["confidence"]
        out["accepted"] = out["confidence"] >= tau
    else:
        out["accepted"] = out["Q_i"] >= tau
    return out


def _apply_tnorm_preassociation(
    df: pd.DataFrame,
    name: str,
    tau: float,
    mode: str,
    k_variant: str,
    alpha_scale: float,
    beta: float | None,
    gamma_assoc: float,
    k_new_track: float,
    tau_soft_update: float,
    max_missed_frames: int,
    soft_conf_floor: float,
    reject_patience: int,
    tau_existing: float | None,
    tau_new: float | None,
    confirm_age: int,
    max_confirm_missed: int,
    confirmed_conf_floor: float,
    risk_tau: float,
    new_conf_tau: float,
    risk_weights: tuple[float, float, float],
) -> pd.DataFrame:
    out = df.copy()
    if out.empty:
        return out
    mode = {"hard_filter": "hard", "soft_reweight": "soft", "delayed_hard_filter": "delayed"}.get(mode, mode)
    tau_existing = tau if tau_existing is None else float(tau_existing)
    tau_new = tau if tau_new is None else float(tau_new)
    out["confidence_original"] = out["confidence"]
    for col in [
        "defense_track_id",
        "assoc_score",
        "assoc_status",
        "gamma_assoc",
        "k_variant",
        "alpha_scale",
        "alpha",
        "k_center",
        "k_iou",
        "k_combined",
        "k_gate",
        "k_acc",
        "k_robust_min",
        "k_i",
        "d_center",
        "Q_raw",
        "Q_smooth",
        "Q_i",
        "beta",
        "soft_conf_floor",
        "reject_patience",
        "low_Q_streak",
        "track_status",
        "missed_frames",
        "suspicious",
        "filter_action",
        "risk_new",
        "risk_tau",
        "new_conf_tau",
    ]:
        out[col] = None
    mode_label = "soft_reweight" if mode == "soft" else "delayed_hard_filter" if mode == "delayed" else mode
    out["filter_mode"] = mode_label
    out["t_norm"] = name
    out["tau"] = tau
    out["tau_existing"] = tau_existing
    out["tau_new"] = tau_new
    out["confirmed_conf_floor"] = confirmed_conf_floor
    out["confirm_age"] = confirm_age
    out["risk_tau"] = risk_tau
    out["new_conf_tau"] = new_conf_tau
    out["gamma_assoc"] = gamma_assoc
    out["soft_conf_floor"] = soft_conf_floor
    out["reject_patience"] = reject_patience
    out["k_variant"] = k_variant
    out["alpha_scale"] = alpha_scale
    out["beta"] = beta

    alpha_base_by_seq = _alpha_base_by_sequence(out)
    next_track_id = 1
    states: dict[str, dict[int, dict]] = {}
    for (seq, frame), frame_df in out.sort_values(["sequence_id", "frame_id", "confidence"], ascending=[True, True, False]).groupby(["sequence_id", "frame_id"], sort=False):
        seq_states = states.setdefault(seq, {})
        for tid in list(seq_states):
            if int(frame) - int(seq_states[tid].get("last_frame_id", frame)) > max_missed_frames:
                del seq_states[tid]
        used_tracks: set[int] = set()
        for idx, det in frame_df.iterrows():
            alpha_base = alpha_base_by_seq.get(seq, 50.0)
            alpha = max(1.0, alpha_scale * alpha_base)
            box = _box(det)
            center = _center(box)
            best_track_id, best_score, best_pred = None, -1.0, None
            for tid, state in seq_states.items():
                if tid in used_tracks:
                    continue
                pred_box = _predict_box(state)
                pred_center = _center(pred_box)
                d = math.hypot(center[0] - pred_center[0], center[1] - pred_center[1])
                k_center = math.exp(-d / alpha)
                k_iou = iou(pred_box, box)
                score = 0.5 * k_iou + 0.5 * k_center
                if score > best_score:
                    best_track_id, best_score, best_pred = tid, score, (pred_box, pred_center, d, k_center, k_iou, state)
            if best_track_id is None or best_score < gamma_assoc:
                tid = next_track_id
                next_track_id += 1
                d_center = 0.0
                k_center = k_iou = k_combined = k_gate = k_acc = k_robust_min = k_new_track
                assoc_score = 0.0
                state = None
                assoc_status = "new_track" if best_track_id is None else "low_assoc"
                track_status = "unmatched_detection" if pd.isna(getattr(det, "pred_track_id", None)) else "new_candidate"
                track_age = 1
                missed_frames = 0
            else:
                tid = int(best_track_id)
                used_tracks.add(tid)
                _, _, d_center, k_center, k_iou, state = best_pred
                k_combined = math.sqrt(max(k_center, 1e-6) * max(k_iou, 1e-6))
                r = d_center / alpha
                k_gate = 1.0 / (1.0 + r * r)
                k_acc = _acceleration_consistency(state, center, alpha)
                k_robust_min = min(k_center, k_iou, k_acc)
                assoc_score = best_score
                assoc_status = "matched_existing_track"
                track_age = int(state.get("age", 0)) + 1
                missed_frames = max(0, int(frame) - int(state.get("last_frame_id", frame)) - 1)
                if track_age >= int(confirm_age) and missed_frames <= int(max_confirm_missed):
                    track_status = "confirmed_existing"
                else:
                    track_status = "tentative_existing"
            k_values = {
                "center": k_center,
                "iou": k_iou,
                "combined": k_combined,
                "gate": k_gate,
                "acc": k_acc,
                "robust_min": k_robust_min,
            }
            k_i = float(k_values.get(k_variant, k_center))
            q_raw = t_norm(name, float(det.c_i), k_i, float(det.s_i), float(det.x_i))
            last_q = None if state is None else state.get("last_Q")
            q_smooth = q_raw if beta is None or last_q is None else beta * q_raw + (1.0 - beta) * float(last_q)
            low_q_streak = 0 if state is None else int(state.get("low_Q_streak", 0))
            if q_smooth < tau:
                low_q_streak += 1
            else:
                low_q_streak = 0
            suspicious = q_smooth < (tau_existing if track_status in {"confirmed_existing", "tentative_existing"} else tau_new)
            filter_action = "keep"
            risk_new = _risk_new(float(det.confidence), k_i, assoc_score, risk_weights)
            if mode == "risk_gated_new_suppression":
                if track_status in {"confirmed_existing", "tentative_existing"}:
                    accepted = True
                    should_update = True
                    filter_action = "keep_confirmed" if track_status == "confirmed_existing" else "keep_tentative"
                else:
                    accepted = risk_new < float(risk_tau)
                    should_update = accepted
                    suspicious = risk_new >= float(risk_tau)
                    filter_action = "keep" if accepted else "reject_new"
            elif mode == "low_conf_new_suppression":
                if track_status in {"confirmed_existing", "tentative_existing"}:
                    accepted = True
                    should_update = True
                    filter_action = "keep_confirmed" if track_status == "confirmed_existing" else "keep_tentative"
                else:
                    accepted = float(det.confidence) >= float(new_conf_tau)
                    should_update = accepted
                    suspicious = not accepted
                    filter_action = "keep" if accepted else "reject_new"
            elif mode == "suspicious_label_only":
                accepted = True
                should_update = True
                suspicious = bool(suspicious or (track_status in {"new_candidate", "unmatched_detection"} and risk_new >= float(risk_tau)))
                filter_action = "label_suspicious" if suspicious else "keep"
            elif mode == "track_aware":
                if track_status == "confirmed_existing":
                    accepted = True
                    if suspicious:
                        new_conf = max(float(det.confidence) * q_smooth, float(confirmed_conf_floor))
                        out.at[idx, "confidence"] = new_conf
                        out.at[idx, "c_i"] = new_conf
                        filter_action = "soft_downweight"
                    should_update = True
                elif track_status == "tentative_existing":
                    accepted = low_q_streak < max(1, reject_patience)
                    should_update = accepted
                    filter_action = "delayed_keep" if accepted else "reject_tentative"
                else:
                    accepted = q_smooth >= tau_new
                    should_update = accepted
                    filter_action = "keep" if accepted else "reject_new"
            elif mode == "new_track_suppression":
                if track_status == "confirmed_existing":
                    accepted = True
                    should_update = True
                    filter_action = "keep"
                elif track_status == "tentative_existing":
                    accepted = low_q_streak < max(1, reject_patience)
                    should_update = accepted
                    filter_action = "delayed_keep" if accepted else "reject_tentative"
                else:
                    accepted = q_smooth >= tau_new
                    should_update = accepted
                    filter_action = "keep" if accepted else "reject_new"
            elif mode == "soft":
                new_conf = max(soft_conf_floor, float(det.confidence) * q_smooth)
                accepted = new_conf >= tau
                out.at[idx, "confidence"] = new_conf
                out.at[idx, "c_i"] = new_conf
                should_update = q_smooth >= tau_soft_update
                filter_action = "soft_downweight" if q_smooth < tau else "keep"
            elif mode == "delayed":
                accepted = low_q_streak < max(1, reject_patience)
                should_update = accepted
                filter_action = "delayed_keep" if accepted else "reject_tentative"
            else:
                accepted = q_smooth >= tau
                should_update = accepted
                filter_action = "keep" if accepted else "reject_new"
            out.at[idx, "defense_track_id"] = tid
            out.at[idx, "track_status"] = track_status
            out.at[idx, "assoc_score"] = assoc_score
            out.at[idx, "assoc_status"] = assoc_status
            out.at[idx, "missed_frames"] = missed_frames
            out.at[idx, "alpha_base"] = alpha_base
            out.at[idx, "alpha"] = alpha
            out.at[idx, "d_center"] = d_center
            out.at[idx, "k_center"] = k_center
            out.at[idx, "k_iou"] = k_iou
            out.at[idx, "k_combined"] = k_combined
            out.at[idx, "k_gate"] = k_gate
            out.at[idx, "k_acc"] = k_acc
            out.at[idx, "k_robust_min"] = k_robust_min
            out.at[idx, "k_i"] = k_i
            out.at[idx, "Q_raw"] = q_raw
            out.at[idx, "Q_smooth"] = q_smooth
            out.at[idx, "Q_i"] = q_smooth
            out.at[idx, "risk_new"] = risk_new
            out.at[idx, "risk_tau"] = risk_tau
            out.at[idx, "new_conf_tau"] = new_conf_tau
            out.at[idx, "confidence_new"] = out.at[idx, "confidence"]
            out.at[idx, "low_Q_streak"] = low_q_streak
            out.at[idx, "suspicious"] = bool(suspicious)
            out.at[idx, "filter_action"] = filter_action
            out.at[idx, "accepted"] = bool(accepted)
            if should_update:
                seq_states[tid] = _updated_state(seq_states.get(tid), box, center, int(frame), q_smooth, bool(accepted), low_q_streak)
    return out


def _alpha_base_by_sequence(df: pd.DataFrame) -> dict[str, float]:
    out = {}
    for seq, group in df.groupby("sequence_id"):
        diag = ((group["x2"] - group["x1"]) ** 2 + (group["y2"] - group["y1"]) ** 2) ** 0.5
        value = float(diag.median()) if len(diag.dropna()) else 50.0
        out[seq] = max(1.0, value)
    return out


def _status_from_existing_row(row, confirm_age: int, max_confirm_missed: int) -> str:
    pred_id = getattr(row, "pred_track_id", None)
    if pd.isna(pred_id):
        return "unmatched_detection"
    age = getattr(row, "track_age", None)
    age = 1 if pd.isna(age) else int(age)
    missed = getattr(row, "missed_frames", 0)
    missed = 0 if pd.isna(missed) else int(missed)
    if age >= confirm_age and missed <= max_confirm_missed:
        return "confirmed_existing"
    if age >= 1:
        return "tentative_existing"
    return "new_candidate"


def _risk_new(confidence: float, k_i: float, assoc_score: float, weights: tuple[float, float, float]) -> float:
    w_conf, w_k, w_assoc = weights
    conf = max(0.0, min(1.0, confidence))
    k = max(0.0, min(1.0, k_i))
    assoc = max(0.0, min(1.0, assoc_score))
    return w_conf * (1.0 - conf) + w_k * (1.0 - k) + w_assoc * (1.0 - assoc)


def _box(row) -> tuple[float, float, float, float]:
    return (float(row.x1), float(row.y1), float(row.x2), float(row.y2))


def _center(box: tuple[float, float, float, float]) -> tuple[float, float]:
    return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)


def _predict_box(state: dict) -> tuple[float, float, float, float]:
    if state.get("bbox_t_minus_2") is None:
        return state["bbox_t_minus_1"]
    b1 = state["bbox_t_minus_1"]
    b2 = state["bbox_t_minus_2"]
    return tuple(b1[i] + (b1[i] - b2[i]) for i in range(4))


def _acceleration_consistency(state: dict | None, center: tuple[float, float], alpha: float) -> float:
    if state is None or state.get("center_t_minus_2") is None:
        return 1.0
    c1 = state["center_t_minus_1"]
    c2 = state["center_t_minus_2"]
    ax = center[0] - 2.0 * c1[0] + c2[0]
    ay = center[1] - 2.0 * c1[1] + c2[1]
    return math.exp(-math.hypot(ax, ay) / alpha)


def _updated_state(state: dict | None, box: tuple[float, float, float, float], center: tuple[float, float], frame: int, q: float, accepted: bool, low_q_streak: int = 0) -> dict:
    return {
        "last_frame_id": frame,
        "bbox_t_minus_1": box,
        "bbox_t_minus_2": None if state is None else state.get("bbox_t_minus_1"),
        "center_t_minus_1": center,
        "center_t_minus_2": None if state is None else state.get("center_t_minus_1"),
        "last_Q": q,
        "last_accepted": accepted,
        "low_Q_streak": low_q_streak,
        "age": 1 if state is None else int(state.get("age", 0)) + 1,
        "missed_frames": 0,
    }


def apply_conf_threshold(df, tau: float):
    out = df.copy()
    out["Q_i"] = out["confidence"]
    out["tau"] = tau
    out["accepted"] = out["confidence"] >= tau
    out["t_norm"] = "confidence"
    return out
