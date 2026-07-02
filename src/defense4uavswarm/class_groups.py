from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml


VISDRONE_ID_TO_NAME = {
    1: "pedestrian",
    2: "people",
    3: "bicycle",
    4: "car",
    5: "van",
    6: "truck",
    7: "tricycle",
    8: "awning-tricycle",
    9: "bus",
    10: "motor",
}


def load_class_groups(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def normalized_names(df: pd.DataFrame, aliases: dict[str, str]) -> pd.Series:
    if "class_name" in df.columns:
        names = df["class_name"].fillna("").astype(str).str.lower()
    else:
        names = df["class_id"].map(VISDRONE_ID_TO_NAME).fillna("").astype(str)
    return names.map(lambda x: aliases.get(x, x))


def filter_by_group(df: pd.DataFrame, cfg: dict, group: str) -> pd.DataFrame:
    groups_cfg = load_class_groups(cfg["class_groups_file"])
    names = set(groups_cfg["class_groups"][group])
    aliases = groups_cfg.get("aliases", {})
    mask = normalized_names(df, aliases).isin(names)
    return df[mask].copy()
