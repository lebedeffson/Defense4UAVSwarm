from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from defense4uavswarm.datasets.visdrone import VisDroneDataset
from defense4uavswarm.filtering.tnorms import apply_tnorm
from defense4uavswarm.metrics.detection import match_frame
from defense4uavswarm.pipeline import save
from defense4uavswarm.tracking.simple import iou


def build_semantic_diagnostics(
    cfg: dict,
    results_dir: str | Path,
    split_config: str | Path | None,
    split: str | None,
    model_name: str,
    eps: float,
    features: list[str],
    augmentation_count: int = 3,
    xai_max_per_frame: int = 3,
) -> None:
    results = Path(results_dir)
    s1_path = results / f"vid_{model_name}_s1_fgsm_eps_{eps}.csv"
    if not s1_path.exists():
        raise FileNotFoundError(f"Missing S1 predictions for semantic diagnostics: {s1_path}")
    sequences = _load_sequences(split_config, split)
    gt = VisDroneDataset(cfg["dataset"]["root"], "val", subset_hint="VID").all_annotations(sequences)
    s1 = pd.read_csv(s1_path)
    scored = apply_tnorm(
        s1,
        "T_min",
        0.1,
        mode="suspicious_label_only",
        k_variant="robust_min",
        alpha_scale=0.5,
        preassociation=True,
        gamma_assoc=0.05,
        risk_tau=0.85,
        risk_weights=tuple(cfg["filtering"].get("risk_weights", [0.4, 0.3, 0.3])),
    )
    audit = semantic_feature_audit(gt, scored, cfg, features, augmentation_count, xai_max_per_frame)
    save(audit, results / "semantic_feature_audit.csv")
    auc = semantic_score_auc(audit)
    save(auc, results / "semantic_score_auc.csv")
    save(semantic_suspicious_summary(audit), results / "semantic_suspicious_summary.csv")
    best = auc.dropna(subset=["roc_auc"]).sort_values("roc_auc", ascending=False).head(1)
    best_score = None if best.empty else str(best.iloc[0]["score_name"])
    best_auc = None if best.empty else float(best.iloc[0]["roc_auc"])
    best_p5 = None if best.empty else float(best.iloc[0]["precision_at_5pct_rejection"])
    metadata = {
        "stage": "v1.5",
        "model": model_name,
        "eps": eps,
        "split": split,
        "semantic_features_requested": features,
        "augmentation_consistency_enabled": "augmentation" in features,
        "augmentation_consistency_num_augments": augmentation_count,
        "augmentation_consistency_candidate_only": True,
        "xai_enabled": "eigencam" in features,
        "xai_max_per_frame": xai_max_per_frame,
        "best_score": best_score,
        "best_auc": best_auc,
        "best_precision_at_5pct_rejection": best_p5,
        "selection_status": "diagnostic_passed" if (best_auc or 0.0) >= 0.70 or (best_p5 or 0.0) >= 0.50 else "diagnostic_failed",
        "reason": None if (best_auc or 0.0) >= 0.70 or (best_p5 or 0.0) >= 0.50 else "semantic_features_do_not_separate_fp_tp",
        "holdout_run": False,
        "holdout_allowed": False,
    }
    (results / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def semantic_feature_audit(gt: pd.DataFrame, pred: pd.DataFrame, cfg: dict, features: list[str], augmentation_count: int, xai_max_per_frame: int) -> pd.DataFrame:
    frame = label_tp_fp(gt, pred)
    frame["bbox_area"] = (frame["x2"].astype(float) - frame["x1"].astype(float)).clip(lower=0) * (frame["y2"].astype(float) - frame["y1"].astype(float)).clip(lower=0)
    frame["class_top1_conf"] = frame["confidence_original"] if "confidence_original" in frame else frame["confidence"]
    frame["class_top2_conf"] = np.nan
    frame["class_margin"] = np.nan
    frame["class_entropy"] = np.nan
    frame["m_i"] = np.nan
    if "temporal" in features:
        frame["t_i"] = temporal_stability(frame)
        frame["temporal_class_stability"] = frame["t_i"]
    else:
        frame["t_i"] = np.nan
        frame["temporal_class_stability"] = np.nan
    if "augmentation" in features:
        frame["a_i"] = augmentation_consistency(frame, cfg, augmentation_count)
        frame["augmentation_consistency"] = frame["a_i"]
    else:
        frame["a_i"] = np.nan
        frame["augmentation_consistency"] = np.nan
    frame["xai_energy_inside"] = np.nan
    frame["xai_peak_inside"] = np.nan
    frame["x_i"] = np.nan
    frame["semantic_risk_score"] = semantic_risk(frame)
    frame["semantic_suspicious"] = frame["semantic_risk_score"] >= frame["semantic_risk_score"].quantile(0.95)
    keep = [
        "sequence_id",
        "frame_id",
        "model_name",
        "eps",
        "scenario",
        "track_status",
        "pred_track_id",
        "gt_track_id_matched",
        "is_TP",
        "is_FP",
        "confidence",
        "k_i",
        "assoc_score",
        "Q_i",
        "class_margin",
        "class_entropy",
        "m_i",
        "augmentation_consistency",
        "a_i",
        "temporal_class_stability",
        "t_i",
        "xai_energy_inside",
        "xai_peak_inside",
        "x_i",
        "bbox_area",
        "class_name",
        "class_name_gt",
        "semantic_suspicious",
        "semantic_risk_score",
    ]
    out = frame[[c for c in keep if c in frame]].rename(columns={"class_name": "class_name_pred"})
    return out


def label_tp_fp(gt: pd.DataFrame, pred: pd.DataFrame) -> pd.DataFrame:
    out = pred.copy()
    pred_frames = {k: v for k, v in out.groupby(["sequence_id", "frame_id"])} if len(out) else {}
    labels = {}
    for (seq, frame_id), gt_f in gt.groupby(["sequence_id", "frame_id"]):
        pred_f = pred_frames.get((seq, frame_id), out.iloc[0:0])
        _, _, _, pairs = match_frame(gt_f, pred_f.assign(accepted=True))
        for gi, pi in pairs:
            labels[pi] = gi
    out["gt_track_id_matched"] = [gt.loc[labels[idx]].get("gt_track_id") if idx in labels else None for idx in out.index]
    out["class_name_gt"] = [gt.loc[labels[idx]].get("class_name") if idx in labels else None for idx in out.index]
    out["is_TP"] = [idx in labels for idx in out.index]
    out["is_FP"] = ~out["is_TP"]
    return out


def temporal_stability(frame: pd.DataFrame, window: int = 5) -> pd.Series:
    values = pd.Series(1.0, index=frame.index)
    track_col = "defense_track_id" if "defense_track_id" in frame else "pred_track_id"
    ordered = frame.sort_values(["sequence_id", track_col, "frame_id"], na_position="last")
    for _, g in ordered.groupby(["sequence_id", track_col], dropna=False):
        history: list[str] = []
        for idx, row in g.iterrows():
            cls = str(row.get("class_name", ""))
            if len(history) >= 2:
                recent = history[-window:]
                values.at[idx] = sum(x == cls for x in recent) / max(1, len(recent))
            history.append(cls)
    return values


def augmentation_consistency(frame: pd.DataFrame, cfg: dict, count: int) -> pd.Series:
    try:
        from ultralytics import YOLO
    except Exception:
        return pd.Series(np.nan, index=frame.index)
    candidates = frame[
        (pd.to_numeric(frame.get("risk_new", frame.get("semantic_risk_score", 0)), errors="coerce").fillna(0) > 0.4)
        | (frame["track_status"].isin(["new_candidate", "unmatched_detection"]))
        | (pd.to_numeric(frame["confidence"], errors="coerce").fillna(1) < 0.30)
    ]
    max_frames = int(cfg.get("semantic", {}).get("augmentation_max_frames", 80))
    candidate_keys = list(candidates[["sequence_id", "frame_id"]].drop_duplicates().itertuples(index=False, name=None))[:max_frames]
    if not candidate_keys:
        return pd.Series(np.nan, index=frame.index)
    model = YOLO(cfg["model"]["weights"])
    device = None if cfg.get("device") == "auto" else cfg.get("device")
    aug_names = ["brightness_plus", "brightness_minus", "resize_095"][: max(1, count)]
    out = pd.Series(np.nan, index=frame.index)
    by_key = {k: v for k, v in frame.groupby(["sequence_id", "frame_id"])}
    for key in candidate_keys:
        group = by_key.get(key)
        if group is None or not len(group):
            continue
        image_path = Path(str(group["image_path"].dropna().iloc[0]))
        img = cv2.imread(str(image_path))
        if img is None:
            continue
        aug_preds = []
        for aug in aug_names:
            pred = _predict_boxes(model, _augment_image(img, aug), cfg, device)
            aug_preds.append(pred)
        for idx, row in group.iterrows():
            ok = 0
            box = (float(row.x1), float(row.y1), float(row.x2), float(row.y2))
            cls = str(row.class_name)
            for pred in aug_preds:
                if any(iou(box, p[:4]) >= 0.5 and str(p[4]) == cls and float(p[5]) >= float(cfg["model"]["conf"]) for p in pred):
                    ok += 1
            out.at[idx] = ok / max(1, len(aug_preds))
    return out


def _augment_image(img, name: str):
    if name == "brightness_plus":
        return cv2.convertScaleAbs(img, alpha=1.0, beta=25)
    if name == "brightness_minus":
        return cv2.convertScaleAbs(img, alpha=1.0, beta=-25)
    if name == "resize_095":
        h, w = img.shape[:2]
        small = cv2.resize(img, (max(1, int(w * 0.95)), max(1, int(h * 0.95))))
        return cv2.resize(small, (w, h))
    return img


def _predict_boxes(model, img, cfg: dict, device):
    res = model.predict(source=img, imgsz=cfg["model"]["imgsz"], conf=cfg["model"]["conf"], iou=cfg["model"]["iou"], max_det=cfg["model"]["max_det"], device=device, verbose=False)[0]
    rows = []
    if res.boxes is None:
        return rows
    for b in res.boxes:
        x1, y1, x2, y2 = [float(v) for v in b.xyxy[0].tolist()]
        cls = int(b.cls[0])
        rows.append((x1, y1, x2, y2, res.names.get(cls, str(cls)), float(b.conf[0])))
    return rows


def semantic_risk(df: pd.DataFrame) -> pd.Series:
    conf = pd.to_numeric(df.get("confidence_original", df["confidence"]), errors="coerce").fillna(0).clip(0, 1)
    k = pd.to_numeric(df["k_i"], errors="coerce").fillna(1).clip(0, 1)
    assoc = pd.to_numeric(df["assoc_score"], errors="coerce").fillna(0).clip(0, 1)
    old = 0.4 * (1 - conf) + 0.3 * (1 - k) + 0.3 * (1 - assoc)
    aug = pd.to_numeric(df.get("a_i", np.nan), errors="coerce")
    temp = pd.to_numeric(df.get("t_i", np.nan), errors="coerce")
    risk = old.copy()
    risk = np.where(aug.notna(), 0.5 * old + 0.5 * (1 - aug.fillna(1)), risk)
    risk = np.where(temp.notna(), 0.8 * risk + 0.2 * (1 - temp.fillna(1)), risk)
    return pd.Series(risk, index=df.index).clip(0, 1)


def semantic_score_auc(audit: pd.DataFrame) -> pd.DataFrame:
    y = audit["is_FP"].astype(int)
    conf = pd.to_numeric(audit["confidence"], errors="coerce").fillna(0).clip(0, 1)
    k = pd.to_numeric(audit["k_i"], errors="coerce").fillna(1).clip(0, 1)
    assoc = pd.to_numeric(audit["assoc_score"], errors="coerce").fillna(0).clip(0, 1)
    q = pd.to_numeric(audit["Q_i"], errors="coerce").fillna(conf).clip(0, 1)
    old = 0.4 * (1 - conf) + 0.3 * (1 - k) + 0.3 * (1 - assoc)
    scores = {
        "old_score": old,
        "score_margin": 1 - pd.to_numeric(audit["m_i"], errors="coerce"),
        "score_aug": 1 - pd.to_numeric(audit["a_i"], errors="coerce"),
        "score_temporal": 1 - pd.to_numeric(audit["t_i"], errors="coerce"),
        "score_xai": 1 - pd.to_numeric(audit["x_i"], errors="coerce"),
        "score_old_aug": 0.5 * old + 0.5 * (1 - pd.to_numeric(audit["a_i"], errors="coerce")),
        "score_old_aug_temporal": 0.4 * old + 0.4 * (1 - pd.to_numeric(audit["a_i"], errors="coerce")) + 0.2 * (1 - pd.to_numeric(audit["t_i"], errors="coerce")),
        "score_old_aug_xai": 0.4 * old + 0.4 * (1 - pd.to_numeric(audit["a_i"], errors="coerce")) + 0.2 * (1 - pd.to_numeric(audit["x_i"], errors="coerce")),
        "score_conf_Q_assoc": 0.4 * (1 - conf) + 0.3 * (1 - q) + 0.3 * (1 - assoc),
    }
    rows = []
    for name, score in scores.items():
        valid = score.notna()
        rows.append({"score_name": name, **score_stats(y[valid].to_numpy(), score[valid].to_numpy())})
    return pd.DataFrame(rows)


def semantic_suspicious_summary(audit: pd.DataFrame) -> pd.DataFrame:
    auc = semantic_score_auc(audit)
    rows = []
    for _, score_row in auc.dropna(subset=["best_threshold"]).iterrows():
        name = score_row["score_name"]
        score = _score_by_name(audit, name)
        threshold = float(score_row["best_threshold"])
        mask = score >= threshold
        rows.append(
            {
                "score_name": name,
                "threshold": threshold,
                "num_suspicious": int(mask.sum()),
                "suspicious_TP": int((mask & audit["is_TP"]).sum()),
                "suspicious_FP": int((mask & audit["is_FP"]).sum()),
                "suspicious_FP_rate": float((mask & audit["is_FP"]).sum() / max(1, mask.sum())),
                "suspicious_TP_rate": float((mask & audit["is_TP"]).sum() / max(1, mask.sum())),
            }
        )
    return pd.DataFrame(rows)


def _score_by_name(audit: pd.DataFrame, name: str) -> pd.Series:
    conf = pd.to_numeric(audit["confidence"], errors="coerce").fillna(0).clip(0, 1)
    k = pd.to_numeric(audit["k_i"], errors="coerce").fillna(1).clip(0, 1)
    assoc = pd.to_numeric(audit["assoc_score"], errors="coerce").fillna(0).clip(0, 1)
    old = 0.4 * (1 - conf) + 0.3 * (1 - k) + 0.3 * (1 - assoc)
    if name == "old_score":
        return old
    if name == "score_aug":
        return 1 - pd.to_numeric(audit["a_i"], errors="coerce")
    if name == "score_temporal":
        return 1 - pd.to_numeric(audit["t_i"], errors="coerce")
    if name == "score_old_aug":
        return 0.5 * old + 0.5 * (1 - pd.to_numeric(audit["a_i"], errors="coerce"))
    if name == "score_old_aug_temporal":
        return 0.4 * old + 0.4 * (1 - pd.to_numeric(audit["a_i"], errors="coerce")) + 0.2 * (1 - pd.to_numeric(audit["t_i"], errors="coerce"))
    return audit["semantic_risk_score"]


def score_stats(y: np.ndarray, score: np.ndarray) -> dict:
    mask = np.isfinite(score)
    y, score = y[mask], score[mask]
    if len(y) == 0 or y.sum() == 0 or y.sum() == len(y):
        return {"roc_auc": None, "average_precision": None, "precision_at_1pct_rejection": None, "precision_at_5pct_rejection": None, "precision_at_10pct_rejection": None, "best_threshold": None, "best_FP_detection_F1": None}
    order = np.argsort(score)[::-1]
    y_sorted, s_sorted = y[order], score[order]
    tp = np.cumsum(y_sorted)
    fp = np.cumsum(1 - y_sorted)
    precision = tp / np.maximum(1, tp + fp)
    recall = tp / max(1, y.sum())
    f1 = 2 * precision * recall / np.maximum(1e-12, precision + recall)
    best_i = int(np.nanargmax(f1))
    ap = float(np.sum((recall - np.r_[0, recall[:-1]]) * precision))
    ranks = pd.Series(score).rank(method="average").to_numpy()
    n_pos, n_neg = y.sum(), len(y) - y.sum()
    auc = float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / max(1, n_pos * n_neg))
    def p_at(frac: float) -> float:
        n = max(1, int(np.ceil(frac * len(y_sorted))))
        return float(y_sorted[:n].mean())
    return {"roc_auc": auc, "average_precision": ap, "precision_at_1pct_rejection": p_at(0.01), "precision_at_5pct_rejection": p_at(0.05), "precision_at_10pct_rejection": p_at(0.10), "best_threshold": float(s_sorted[best_i]), "best_FP_detection_F1": float(f1[best_i])}


def _load_sequences(split_config: str | Path | None, split: str | None) -> list[str] | None:
    if not split_config or not split:
        return None
    import yaml

    with open(split_config, "r", encoding="utf-8") as f:
        return [str(x) for x in (yaml.safe_load(f) or {}).get(split, [])]
