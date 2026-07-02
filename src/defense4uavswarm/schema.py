from __future__ import annotations

import pandas as pd


BASE_COLUMNS = [
    "scenario",
    "model_name",
    "fgsm_loss",
    "eps",
    "frame_id",
    "sequence_id",
    "gt_track_id",
    "pred_track_id",
    "class_id",
    "class_name",
    "x1",
    "y1",
    "x2",
    "y2",
    "confidence",
    "c_i",
    "k_i",
    "s_i",
    "x_i",
    "Q_i",
    "accepted",
    "t_norm",
    "tau",
    "image_path",
    "latency_ms",
    "xai_triggered",
]


def to_frame(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    for col in BASE_COLUMNS:
        if col not in df.columns:
            df[col] = None
    return df[BASE_COLUMNS]
