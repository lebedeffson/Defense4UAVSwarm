from __future__ import annotations

import hashlib
import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, PoissonRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


STAGE1_NUMERIC = [
    "confidence_start",
    "absolute_support_start",
    "relative_confidence_start",
    "episode_start_rank",
    "bbox_area_norm_start",
    "log_bbox_area_norm_start",
    "aspect_ratio_start",
    "border_distance_norm_start",
    "simultaneous_episode_starts",
    "frame_candidate_count",
    "frame_median_support",
    "past_survival_rate",
    "past_quarantine_threshold",
]
STAGE1_CATEGORICAL = ["class_name_start", "tracker_profile"]
STAGE2_NUMERIC = STAGE1_NUMERIC + [
    "second_observation_gap_frames",
    "confidence_second",
    "support_second",
    "confidence_delta",
    "support_delta",
    "bbox_iou_1_2",
    "center_displacement_norm_1_2",
    "scale_ratio_1_2",
    "aspect_ratio_delta_1_2",
    "border_distance_delta_1_2",
    "local_new_start_density_second_frame",
]
STAGE2_CATEGORICAL = STAGE1_CATEGORICAL + ["class_consistent_1_2"]


@dataclass
class RiskRanker:
    stage: str
    numeric_features: list[str]
    categorical_features: list[str]
    classifier: Pipeline
    severity: Pipeline | None
    model_sha256: str

    def predict(self, data: pd.DataFrame, *, horizon: int = 10) -> pd.DataFrame:
        X = _ensure_columns(data, self.numeric_features + self.categorical_features)
        if len(X) == 0:
            out = data[["sequence_id", "tracklet_id", "episode_id"]].copy()
            out["p_false"] = []
            out["predicted_bounded_false_rows"] = []
            out["risk_score"] = []
            out["model_sha256"] = self.model_sha256
            return out
        p_false = self.classifier.predict_proba(X)[:, 1]
        if self.severity is None:
            predicted = np.zeros(len(X), dtype=float)
        else:
            predicted = np.asarray(self.severity.predict(X), dtype=float)
        predicted = np.clip(predicted, 0.0, float(horizon))
        out = data[["sequence_id", "tracklet_id", "episode_id"]].copy()
        out["p_false"] = p_false
        out["predicted_bounded_false_rows"] = predicted
        out["risk_score"] = p_false * np.maximum(0.0, predicted)
        out["model_sha256"] = self.model_sha256
        return out


def train_stage_ranker(data: pd.DataFrame, *, stage: int, seed: int = 2026) -> RiskRanker:
    numeric = STAGE1_NUMERIC if int(stage) == 1 else STAGE2_NUMERIC
    categorical = STAGE1_CATEGORICAL if int(stage) == 1 else STAGE2_CATEGORICAL
    target = "false_rows_first_1_frame" if int(stage) == 1 else "harm_target_h10"
    df = data.copy()
    if int(stage) == 2:
        df = df[df["second_frame_id"].notna()].copy()
    X = _ensure_columns(df, numeric + categorical)
    y_false = df["is_false_episode"].astype(bool).astype(int)
    classifier = Pipeline(
        [
            ("pre", _preprocessor(numeric, categorical)),
            (
                "model",
                LogisticRegression(
                    C=1.0,
                    class_weight="balanced",
                    solver="liblinear",
                    max_iter=2000,
                    random_state=int(seed),
                ),
            ),
        ]
    )
    if len(np.unique(y_false)) < 2:
        # Degenerate train fold; fit a stable model by adding one synthetic opposite row.
        X = pd.concat([X, X.iloc[[0]].copy()], ignore_index=True)
        y_false = pd.concat([y_false, pd.Series([1 - int(y_false.iloc[0])])], ignore_index=True)
    classifier.fit(X, y_false)
    false_df = df[df["is_false_episode"].astype(bool)].copy()
    severity: Pipeline | None
    if false_df.empty:
        severity = None
    else:
        severity = Pipeline(
            [
                ("pre", _preprocessor(numeric, categorical)),
                ("model", PoissonRegressor(alpha=1.0, max_iter=2000)),
            ]
        )
        severity.fit(_ensure_columns(false_df, numeric + categorical), false_df[target].astype(float).clip(lower=0.0))
    payload = pickle.dumps({"stage": int(stage), "classifier": classifier, "severity": severity, "features": [numeric, categorical]}, protocol=pickle.HIGHEST_PROTOCOL)
    return RiskRanker(
        stage=f"stage{int(stage)}",
        numeric_features=numeric,
        categorical_features=categorical,
        classifier=classifier,
        severity=severity,
        model_sha256=hashlib.sha256(payload).hexdigest(),
    )


def train_thresholds(train: pd.DataFrame, stage1_pred: pd.DataFrame, stage2_pred: pd.DataFrame, *, alpha1: float = 0.10, alpha2: float = 0.05) -> dict[str, float]:
    keys = ["sequence_id", "tracklet_id", "episode_id"]
    stage1 = train[keys + ["is_false_episode"]].merge(stage1_pred[keys + ["risk_score"]], on=keys, how="left")
    true1 = stage1[~stage1["is_false_episode"].astype(bool)]["risk_score"].fillna(0.0)
    stage2_train = train[train["second_frame_id"].notna()][keys + ["is_false_episode"]].copy()
    stage2 = stage2_train.merge(stage2_pred[keys + ["risk_score"]], on=keys, how="left")
    true2 = stage2[~stage2["is_false_episode"].astype(bool)]["risk_score"].fillna(0.0)
    return {
        "tau_stage1": float(true1.quantile(1.0 - float(alpha1))) if not true1.empty else float("inf"),
        "tau_stage2": float(true2.quantile(1.0 - float(alpha2))) if not true2.empty else float("inf"),
    }


def oof_predictions(episodes: pd.DataFrame, *, seed: int = 2026) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    stage1_rows: list[pd.DataFrame] = []
    stage2_rows: list[pd.DataFrame] = []
    cards: list[dict[str, Any]] = []
    for seq in sorted(episodes["sequence_id"].astype(str).unique()):
        train = episodes[~episodes["sequence_id"].astype(str).eq(seq)].copy()
        test = episodes[episodes["sequence_id"].astype(str).eq(seq)].copy()
        r1 = train_stage_ranker(train, stage=1, seed=seed)
        r2 = train_stage_ranker(train, stage=2, seed=seed)
        p1 = r1.predict(test, horizon=1)
        p2 = r2.predict(test[test["second_frame_id"].notna()].copy(), horizon=10)
        p1["excluded_sequence"] = seq
        p2["excluded_sequence"] = seq
        stage1_rows.append(p1)
        stage2_rows.append(p2)
        cards.append(
            {
                "excluded_sequence": seq,
                "train_sequences": sorted(train["sequence_id"].astype(str).unique()),
                "stage1_model_sha256": r1.model_sha256,
                "stage2_model_sha256": r2.model_sha256,
                "label_access": "source_sequence_labels_only",
            }
        )
    return (
        pd.concat(stage1_rows, ignore_index=True) if stage1_rows else pd.DataFrame(),
        pd.concat(stage2_rows, ignore_index=True) if stage2_rows else pd.DataFrame(),
        pd.DataFrame(cards),
    )


def save_model_card(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _preprocessor(numeric: list[str], categorical: list[str]) -> ColumnTransformer:
    return ColumnTransformer(
        [
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric),
            ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical),
        ]
    )


def _ensure_columns(data: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = data.copy()
    for col in columns:
        if col not in out:
            out[col] = np.nan
    return out[columns].copy()
