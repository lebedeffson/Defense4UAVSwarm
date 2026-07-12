from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Any

import numpy as np
import pandas as pd

from defense4uavswarm.q1_v5.evaluation_matching import MatchingConfig, evaluate_gate_result
from defense4uavswarm.q1_v5.initiation_gate import GateConfig, gate_result_from_trigger, tracker_baseline_gate_result


FEATURE_COLUMNS = [
    "confidence",
    "c_i",
    "k_i",
    "s_i",
    "Q_i",
    "confidence_new",
    "bbox_area",
    "bbox_aspect_ratio",
    "num_detections_in_frame",
    "temporal_age",
    "bbox_width",
    "bbox_height",
    "bbox_center_x",
    "bbox_center_y",
    "episode_observation_index",
    "observations_so_far",
    "misses_so_far",
    "frames_since_last_observation",
    "center_delta",
    "area_delta",
    "velocity_consistency",
    "quarantine_window_frame",
]


@dataclass(frozen=True)
class RFSpec:
    n_estimators: int = 80
    max_depth: int | None = 8
    min_samples_leaf: int = 4
    max_features: str | None = "sqrt"
    class_weight: str | None = "balanced"
    decision_threshold: float = 0.5

    def to_params(self) -> dict[str, Any]:
        return {
            "n_estimators": int(self.n_estimators),
            "max_depth": self.max_depth,
            "min_samples_leaf": int(self.min_samples_leaf),
            "max_features": self.max_features,
            "class_weight": self.class_weight,
            "decision_threshold": float(self.decision_threshold),
        }


def default_rf_grid() -> list[RFSpec]:
    rows = []
    for max_depth, min_leaf, threshold in product([6, 10], [4, 8], [0.35, 0.50, 0.65]):
        rows.append(RFSpec(max_depth=max_depth, min_samples_leaf=min_leaf, decision_threshold=threshold))
    return rows


def build_episode_feature_table(candidates: pd.DataFrame) -> pd.DataFrame:
    if candidates.empty:
        return pd.DataFrame(columns=["sequence_id", "tracklet_id", "episode_id", *FEATURE_COLUMNS])
    rows = []
    sort_cols = ["sequence_id", "tracklet_id", "episode_id", "frame_id"] + (["det_id"] if "det_id" in candidates else [])
    for (seq, tracklet, episode), group in candidates.sort_values(sort_cols).groupby(["sequence_id", "tracklet_id", "episode_id"], sort=False):
        first = group.iloc[0]
        width = max(0.0, float(first.x2) - float(first.x1))
        height = max(0.0, float(first.y2) - float(first.y1))
        area = float(first.get("bbox_area", width * height))
        row = {
            "sequence_id": str(seq),
            "tracklet_id": str(tracklet),
            "episode_id": int(episode),
            "first_frame_id": int(first.frame_id),
            "confidence": float(first.get("confidence", 0.0)),
            "c_i": float(first.get("c_i", first.get("confidence", 0.0))),
            "k_i": float(first.get("k_i", 0.0)),
            "s_i": float(first.get("s_i", 1.0)),
            "Q_i": float(first.get("Q_i", min(float(first.get("confidence", 0.0)), float(first.get("k_i", 0.0))))),
            "confidence_new": float(first.get("confidence_new", first.get("confidence", 0.0))),
            "bbox_area": area,
            "bbox_aspect_ratio": float(first.get("bbox_aspect_ratio", width / max(height, 1e-6))),
            "num_detections_in_frame": float(first.get("num_detections_in_frame", 0.0)),
            "temporal_age": float(first.get("temporal_age", 0.0)),
            "bbox_width": width,
            "bbox_height": height,
            "bbox_center_x": float(first.x1) + width / 2.0,
            "bbox_center_y": float(first.y1) + height / 2.0,
            "episode_observation_index": 0.0,
            "observations_so_far": 1.0,
            "misses_so_far": 0.0,
            "frames_since_last_observation": 0.0,
            "center_delta": 0.0,
            "area_delta": 0.0,
            "velocity_consistency": float(first.get("k_i", 0.0)),
            "quarantine_window_frame": 0.0,
        }
        rows.append(row)
    return pd.DataFrame(rows)


def add_episode_labels(
    candidates: pd.DataFrame,
    gt: pd.DataFrame,
    ignored: pd.DataFrame | None,
    frame_manifest: pd.DataFrame,
    matching_config: MatchingConfig,
    gate_config: GateConfig,
) -> pd.DataFrame:
    features = build_episode_feature_table(candidates)
    if features.empty:
        features["false_episode"] = pd.Series(dtype=int)
        features["true_episode"] = pd.Series(dtype=int)
        return features
    result = evaluate_gate_result(candidates, tracker_baseline_gate_result(candidates, gate_config), gt, ignored, frame_manifest, matching_config)
    matched = result.matched_detections.copy()
    labels = (
        matched.groupby(["sequence_id", "tracklet_id", "episode_id"], as_index=False)
        .agg(true_episode=("eval_is_tp", "max"))
    )
    labels["true_episode"] = labels["true_episode"].astype(bool).astype(int)
    out = features.merge(labels, on=["sequence_id", "tracklet_id", "episode_id"], how="left")
    out["true_episode"] = out["true_episode"].fillna(0).astype(int)
    out["false_episode"] = (1 - out["true_episode"]).astype(int)
    return out


def feature_matrix(episode_table: pd.DataFrame) -> np.ndarray:
    return episode_table[FEATURE_COLUMNS].replace([np.inf, -np.inf], 0.0).fillna(0.0).to_numpy(dtype=float)


def train_random_forest(episode_table: pd.DataFrame, spec: RFSpec, random_state: int = 2026) -> Any:
    from sklearn.ensemble import RandomForestClassifier

    y = episode_table["false_episode"].astype(int).to_numpy()
    if len(set(y.tolist())) < 2:
        raise ValueError("RF training needs both true and false episodes")
    model = RandomForestClassifier(
        n_estimators=int(spec.n_estimators),
        max_depth=spec.max_depth,
        min_samples_leaf=int(spec.min_samples_leaf),
        max_features=spec.max_features,
        class_weight=spec.class_weight,
        random_state=int(random_state),
        n_jobs=1,
    )
    model.fit(feature_matrix(episode_table), y)
    return model


def false_probability(model: Any, episode_table: pd.DataFrame) -> np.ndarray:
    proba = model.predict_proba(feature_matrix(episode_table))
    classes = list(getattr(model, "classes_", []))
    if 1 in classes:
        return proba[:, classes.index(1)]
    return np.zeros(len(episode_table), dtype=float)


def score_episode_predictions(episode_table: pd.DataFrame, fp_probability: np.ndarray, threshold: float) -> dict[str, float]:
    accepted = np.asarray(fp_probability, dtype=float) < float(threshold)
    true_mask = episode_table["true_episode"].astype(bool).to_numpy()
    false_mask = episode_table["false_episode"].astype(bool).to_numpy()
    true_retention = float((accepted & true_mask).sum() / max(1, true_mask.sum()))
    false_rejection = float(((~accepted) & false_mask).sum() / max(1, false_mask.sum()))
    false_acceptance = float((accepted & false_mask).sum() / max(1, false_mask.sum()))
    objective = true_retention * 2.0 + false_rejection - false_acceptance
    return {
        "true_retention": true_retention,
        "false_rejection": false_rejection,
        "false_acceptance": false_acceptance,
        "objective": objective,
    }


def select_rf_spec_nested(episode_table: pd.DataFrame, train_sequences: list[str], specs: list[RFSpec] | None = None, random_state: int = 2026) -> tuple[RFSpec, pd.DataFrame]:
    specs = specs or default_rf_grid()
    rows = []
    for spec_id, spec in enumerate(specs):
        fold_scores = []
        for inner_test in train_sequences:
            inner_train = [s for s in train_sequences if s != inner_test]
            train = episode_table[episode_table["sequence_id"].isin(inner_train)].copy()
            val = episode_table[episode_table["sequence_id"].eq(inner_test)].copy()
            if train.empty or val.empty or train["false_episode"].nunique() < 2:
                continue
            model = train_random_forest(train, spec, random_state=random_state + spec_id)
            fp = false_probability(model, val)
            score = score_episode_predictions(val, fp, spec.decision_threshold)
            score.update({"inner_test_sequence": inner_test})
            fold_scores.append(score)
        if not fold_scores:
            continue
        frame = pd.DataFrame(fold_scores)
        params = spec.to_params()
        rows.append(
            {
                "spec_id": spec_id,
                **params,
                "inner_mean_objective": float(frame["objective"].mean()),
                "inner_mean_true_retention": float(frame["true_retention"].mean()),
                "inner_min_true_retention": float(frame["true_retention"].min()),
                "inner_mean_false_rejection": float(frame["false_rejection"].mean()),
                "inner_mean_false_acceptance": float(frame["false_acceptance"].mean()),
            }
        )
    result = pd.DataFrame(rows)
    if result.empty:
        raise RuntimeError("No RF hyperparameter candidate could be selected")
    feasible = result[result["inner_min_true_retention"].ge(0.95)].copy()
    pool = feasible if not feasible.empty else result
    selected = pool.sort_values(["inner_mean_objective", "inner_mean_true_retention", "inner_mean_false_rejection"], ascending=[False, False, False]).iloc[0]
    return (
        RFSpec(
            n_estimators=int(selected["n_estimators"]),
            max_depth=None if pd.isna(selected["max_depth"]) else int(selected["max_depth"]),
            min_samples_leaf=int(selected["min_samples_leaf"]),
            max_features=None if pd.isna(selected["max_features"]) else str(selected["max_features"]),
            class_weight=None if pd.isna(selected["class_weight"]) else str(selected["class_weight"]),
            decision_threshold=float(selected["decision_threshold"]),
        ),
        result,
    )


def rf_unbounded_gate_result(candidates: pd.DataFrame, episode_table: pd.DataFrame, fp_probability: np.ndarray, spec: RFSpec, cfg: GateConfig | None = None):
    prob_by_key = {
        (str(r.sequence_id), str(r.tracklet_id), int(r.episode_id)): float(p)
        for r, p in zip(episode_table.itertuples(index=False), np.asarray(fp_probability, dtype=float))
    }
    probs = candidates.apply(lambda r: prob_by_key[(str(r.sequence_id), str(r.tracklet_id), int(r.episode_id))], axis=1)
    trigger = probs.astype(float) < float(spec.decision_threshold)
    return gate_result_from_trigger(candidates, trigger, cfg, method_id="rf_unbounded", parameters=spec.to_params())
