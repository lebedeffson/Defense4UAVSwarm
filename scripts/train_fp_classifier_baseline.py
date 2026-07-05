#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--train-audit", required=True)
    p.add_argument("--test-audit", required=True)
    p.add_argument("--models", nargs="+", default=["logistic", "random_forest", "gradient_boosting"])
    p.add_argument("--iou-threshold", type=float, default=0.5)
    p.add_argument("--class-weight", default="balanced")
    p.add_argument("--output-dir", required=True)
    args = p.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    train = load_frame(args.train_audit)
    test = load_frame(args.test_audit)
    rows = []
    thresholds = []
    importances = []
    for model in args.models:
        score, importance = fit_predict(model, train, test)
        y = (~test["eval_is_tp"].astype(bool)).astype(int).to_numpy()
        best_t, best = threshold_sweep(y, score)
        rows.append({"model": model, "roc_auc": roc_auc(y, score), "average_precision": average_precision(y, score), "best_threshold": best_t, **best})
        for t in np.linspace(0.1, 0.9, 9):
            thresholds.append({"model": model, "threshold": t, **metrics(y, score >= t)})
        for name, value in importance.items():
            importances.append({"model": model, "feature": name, "importance": value})
    pd.DataFrame(rows).to_csv(out / "fp_classifier_metrics.csv", index=False)
    pd.DataFrame(thresholds).to_csv(out / "fp_classifier_threshold_sweep.csv", index=False)
    pd.DataFrame(importances).to_csv(out / "fp_classifier_feature_importance.csv", index=False)


def load_frame(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    for col in ["confidence_original", "confidence_new", "c_i", "k_i", "s_i", "class_id"]:
        if col not in df:
            df[col] = df.get("confidence", 0.0) if "confidence" in df else 0.0
    df["bbox_area"] = (df["x2"] - df["x1"]).clip(lower=1) * (df["y2"] - df["y1"]).clip(lower=1)
    df["bbox_aspect_ratio"] = (df["x2"] - df["x1"]).clip(lower=1) / (df["y2"] - df["y1"]).clip(lower=1)
    return df


def features(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    cols = ["c_i", "k_i", "s_i", "confidence_original", "confidence_new", "bbox_area", "bbox_aspect_ratio", "class_id"]
    return df[cols].fillna(0).astype(float).to_numpy(), cols


def fit_predict(model: str, train: pd.DataFrame, test: pd.DataFrame) -> tuple[np.ndarray, dict[str, float]]:
    x_train, cols = features(train)
    x_test, _ = features(test)
    y_train = (~train["eval_is_tp"].astype(bool)).astype(int).to_numpy()
    try:
        if model == "logistic":
            from sklearn.linear_model import LogisticRegression
            clf = LogisticRegression(class_weight="balanced", max_iter=1000).fit(x_train, y_train)
            return clf.predict_proba(x_test)[:, 1], dict(zip(cols, np.abs(clf.coef_[0])))
        if model == "random_forest":
            from sklearn.ensemble import RandomForestClassifier
            clf = RandomForestClassifier(n_estimators=50, class_weight="balanced", random_state=42).fit(x_train, y_train)
            return clf.predict_proba(x_test)[:, 1], dict(zip(cols, clf.feature_importances_))
        from sklearn.ensemble import GradientBoostingClassifier
        clf = GradientBoostingClassifier(random_state=42).fit(x_train, y_train)
        return clf.predict_proba(x_test)[:, 1], dict(zip(cols, clf.feature_importances_))
    except Exception:
        score = 1.0 - test["s_i"].fillna(1.0).astype(float).to_numpy()
        return score, {c: (1.0 if c == "s_i" else 0.0) for c in cols}


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
