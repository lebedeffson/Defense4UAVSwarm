from __future__ import annotations

import json
import platform
import subprocess
from datetime import datetime
from pathlib import Path


def version_of(pkg: str) -> str | None:
    try:
        mod = __import__(pkg)
        return getattr(mod, "__version__", "unknown")
    except Exception:
        return None


def git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def write_metadata(path: str | Path, cfg: dict, extra: dict) -> None:
    payload = {
        "dataset": cfg["dataset"]["name"],
        "split": cfg["dataset"]["split"],
        "date": datetime.now().isoformat(timespec="seconds"),
        "python_version": platform.python_version(),
        "torch_version": version_of("torch"),
        "ultralytics_version": version_of("ultralytics"),
        "device": cfg.get("device", "auto"),
        "model_weights": cfg["model"]["weights"],
        "tracker": cfg["model"]["tracker"],
        "models": [m["name"] for m in cfg.get("models", [])],
        "weights": {m["name"]: m["weights"] for m in cfg.get("models", [])},
        "fgsm_loss": cfg["fgsm"]["loss"],
        "fgsm_eps": cfg["fgsm"]["eps_values"],
        "fgsm_is_gradient_based": True,
        "eps_values": cfg["fgsm"]["eps_values"],
        "s_feature_mode": cfg["filtering"]["s_feature_mode"],
        "xai": cfg["xai"],
        "metrics_note": "mAP field is AP@0.5 in the minimal first pipeline",
        "prediction_class_filter": cfg.get("evaluation", {}).get("prediction_class_filter"),
        "prediction_class_filter_mode": cfg.get("evaluation", {}).get("prediction_class_filter_mode"),
        "class_agnostic_eval": cfg.get("evaluation", {}).get("class_agnostic"),
        "tnorm_filter_mode": cfg.get("filtering", {}).get("tnorm_filter_mode"),
        "commit": git_commit(),
    }
    payload.update(extra)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
