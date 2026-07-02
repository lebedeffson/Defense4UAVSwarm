from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dirs(cfg: dict[str, Any]) -> None:
    for key in ("results_dir", "figures_dir"):
        Path(cfg["outputs"][key]).mkdir(parents=True, exist_ok=True)
    Path(cfg["fgsm"]["cache_dir"]).mkdir(parents=True, exist_ok=True)
    Path(cfg["xai"]["examples_dir"]).mkdir(parents=True, exist_ok=True)


def require_packages(names: list[str]) -> None:
    missing = []
    for name in names:
        try:
            __import__(name)
        except ImportError:
            missing.append(name)
    if missing:
        joined = ", ".join(missing)
        raise RuntimeError(f"Missing packages: {joined}. Install: pip install -r requirements.txt")
