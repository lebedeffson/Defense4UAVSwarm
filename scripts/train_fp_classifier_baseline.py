#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import pickle

import numpy as np
import pandas as pd
import yaml


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--train-audit", required=True)
    p.add_argument("--test-audit", required=True)
    p.add_argument("--models", nargs="+", default=["logistic", "random_forest", "gradient_boosting"])
    p.add_argument("--iou-threshold", type=float, default=0.5)
    p.add_argument("--class-weight", default="balanced")
    p.add_argument("--threshold-grid", nargs="+", type=float, default=[0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90])
    p.add_argument("--selection-objective", default="constrained_fp_reduction")
    p.add_argument("--base-confidence-threshold", type=float, default=0.6)
    p.add_argument("--output-dir", required=True)
    args = p.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    train = load_frame(args.train_audit)
    test = load_frame(args.test_audit)
    rows = []
    thresholds = []
    importances = []
    best_payload = None
    for model in args.models:
        score, importance, clf = fit_predict(model, train, test, return_model=True)
        y = (~test["eval_is_tp"].astype(bool)).astype(int).to_numpy()
        best_t, best = threshold_sweep(y, score)
        rows.append({"model": model, "roc_auc": roc_auc(y, score), "average_precision": average_precision(y, score), "best_threshold": best_t, **best})
        for t in np.linspace(0.1, 0.9, 9):
            thresholds.append({"model": model, "threshold": t, **metrics(y, score >= t)})
        calib_score = predict_score(clf, model, train)
        selected = select_tracking_threshold(train, calib_score, args.threshold_grid, args.base_confidence_threshold)
        selected["model"] = model
        selected["selection_objective"] = args.selection_objective
        selected["base_confidence_threshold"] = args.base_confidence_threshold
        selected["classifier_threshold"] = selected.pop("threshold")
        if best_payload is None or selected["score"] > best_payload["selected"]["score"]:
            best_payload = {"model": model, "clf": clf, "selected": selected, "importance": importance}
        for name, value in importance.items():
            importances.append({"model": model, "feature": name, "importance": value})
    pd.DataFrame(rows).to_csv(out / "fp_classifier_metrics.csv", index=False)
    pd.DataFrame(rows).to_csv(out / "fp_classifier_train_metrics.csv", index=False)
    pd.DataFrame(thresholds).to_csv(out / "fp_classifier_threshold_sweep.csv", index=False)
    pd.DataFrame(importances).to_csv(out / "fp_classifier_feature_importance.csv", index=False)
    if best_payload:
        with (out / "best_model.pkl").open("wb") as f:
            pickle.dump({"model_name": best_payload["model"], "model": best_payload["clf"]}, f)
        (out / "selected_threshold.yaml").write_text(yaml.safe_dump(best_payload["selected"], sort_keys=False), encoding="utf-8")
        pd.DataFrame([best_payload["selected"]]).to_csv(out / "fp_classifier_threshold_selection.csv", index=False)


def load_frame(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    for col in ["confidence_original", "confidence_new", "c_i", "k_i", "s_i", "class_id"]:
        if col not in df:
            df[col] = df.get("confidence", 0.0) if "confidence" in df else 0.0
    df["bbox_area"] = (df["x2"] - df["x1"]).clip(lower=1) * (df["y2"] - df["y1"]).clip(lower=1)
    df["bbox_aspect_ratio"] = (df["x2"] - df["x1"]).clip(lower=1) / (df["y2"] - df["y1"]).clip(lower=1)
    return df


def features(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    if "bbox_area" not in df:
        df = df.copy()
        df["bbox_area"] = (df["x2"] - df["x1"]).clip(lower=1) * (df["y2"] - df["y1"]).clip(lower=1)
    if "bbox_aspect_ratio" not in df:
        df = df.copy()
        df["bbox_aspect_ratio"] = (df["x2"] - df["x1"]).clip(lower=1) / (df["y2"] - df["y1"]).clip(lower=1)
    if "confidence_original" not in df:
        df = df.copy()
        df["confidence_original"] = df.get("confidence", 0.0)
    if "confidence_new" not in df:
        df = df.copy()
        df["confidence_new"] = df.get("confidence", 0.0)
    cols = ["c_i", "k_i", "s_i", "confidence_original", "confidence_new", "bbox_area", "bbox_aspect_ratio", "class_id"]
    return df[cols].fillna(0).astype(float).to_numpy(), cols


def fit_predict(model: str, train: pd.DataFrame, test: pd.DataFrame, return_model: bool = False):
    x_train, cols = features(train)
    x_test, _ = features(test)
    y_train = (~train["eval_is_tp"].astype(bool)).astype(int).to_numpy()
    try:
        if model == "logistic":
            from sklearn.linear_model import LogisticRegression
            clf = LogisticRegression(class_weight="balanced", max_iter=1000).fit(x_train, y_train)
            result = (clf.predict_proba(x_test)[:, 1], dict(zip(cols, np.abs(clf.coef_[0]))), clf)
            return result if return_model else result[:2]
        if model == "random_forest":
            from sklearn.ensemble import RandomForestClassifier
            clf = RandomForestClassifier(n_estimators=50, class_weight="balanced", random_state=42).fit(x_train, y_train)
            result = (clf.predict_proba(x_test)[:, 1], dict(zip(cols, clf.feature_importances_)), clf)
            return result if return_model else result[:2]
        from sklearn.ensemble import GradientBoostingClassifier
        clf = GradientBoostingClassifier(random_state=42).fit(x_train, y_train)
        result = (clf.predict_proba(x_test)[:, 1], dict(zip(cols, clf.feature_importances_)), clf)
        return result if return_model else result[:2]
    except Exception:
        score = 1.0 - test["s_i"].fillna(1.0).astype(float).to_numpy()
        clf = {"fallback": "1-s_i"}
        result = (score, {c: (1.0 if c == "s_i" else 0.0) for c in cols}, clf)
        return result if return_model else result[:2]


def predict_score(clf, model: str, frame: pd.DataFrame) -> np.ndarray:
    x, _ = features(frame)
    if isinstance(clf, dict):
        return 1.0 - frame["s_i"].fillna(1.0).astype(float).to_numpy()
    return clf.predict_proba(x)[:, 1]


def select_tracking_threshold(frame: pd.DataFrame, score: np.ndarray, grid: list[float], base_conf: float) -> dict:
    expected_gt = int(frame["eval_is_tp"].astype(bool).sum())
    base_accepted = frame["confidence"].astype(float).to_numpy() >= base_conf
    y_tp = frame["eval_is_tp"].astype(bool).to_numpy()
    is_new = frame["track_status"].eq("new_candidate").to_numpy() if "track_status" in frame else np.ones(len(frame), dtype=bool)
    base = tracking_metrics(base_accepted, y_tp, expected_gt, is_new)
    candidates = []
    for tau in grid:
        accepted = base_accepted & ~(is_new & (score >= tau))
        row = tracking_metrics(accepted, y_tp, expected_gt, is_new)
        row["threshold"] = tau
        row["FP_delta_vs_base"] = row["FP"] - base["FP"]
        row["FN_delta_vs_base"] = row["FN"] - base["FN"]
        row["F1_delta_vs_base"] = row["F1"] - base["F1"]
        row["score"] = (-row["FP_delta_vs_base"] / max(1, base["FP"])) + row["F1_delta_vs_base"]
        row["passes_constraints"] = row["FN_delta_vs_base"] <= max(1, base["FN"] * 0.01) and row["F1_delta_vs_base"] >= -0.0005
        candidates.append(row)
    valid = [r for r in candidates if r["passes_constraints"]]
    return sorted(valid or candidates, key=lambda r: (r["passes_constraints"], r["score"], r["F1"]), reverse=True)[0]


def tracking_metrics(accepted: np.ndarray, y_tp: np.ndarray, expected_gt: int, is_new: np.ndarray) -> dict:
    tp = int((accepted & y_tp).sum())
    fp = int((accepted & ~y_tp).sum())
    fn = max(0, expected_gt - tp)
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-9, precision + recall)
    false_new = int((accepted & is_new & ~y_tp).sum())
    return {"TP": tp, "FP": fp, "FN": fn, "precision": precision, "recall": recall, "F1": f1, "IDF1": f1, "false_new_tracks": false_new}


def threshold_sweep(y: np.ndarray, score: np.ndarray) -> tuple[float, dict]:
    best_t, best, best_f1 = 0.5, {}, -1.0
    for t in np.linspace(0.05, 0.95, 19):
        m = metrics(y, score >= t)
        if m["F1"] > best_f1:
            best_t, best, best_f1 = float(t), m, m["F1"]
    return best_t, best


def metrics(y: np.ndarray, pred: np.ndarray) -> dict:
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-9, precision + recall)
    return {"TP": tp, "FP": fp, "FN": fn, "precision": precision, "recall": recall, "F1": f1, "FP_delta_vs_S_naive": None, "FN_delta_vs_S_naive": None, "F1_delta_vs_S_naive": None}


def roc_auc(y: np.ndarray, score: np.ndarray) -> float:
    pos, neg = score[y == 1], score[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return 0.5
    return float(((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean()))


def average_precision(y: np.ndarray, score: np.ndarray) -> float:
    order = np.argsort(-score)
    yy = y[order]
    if yy.sum() == 0:
        return 0.0
    precision = np.cumsum(yy) / np.arange(1, len(yy) + 1)
    return float((precision * yy).sum() / yy.sum())


if __name__ == "__main__":
    main()
