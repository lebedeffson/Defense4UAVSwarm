from __future__ import annotations

import json
import platform
import subprocess
from datetime import datetime
from pathlib import Path

import yaml


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
    mapping_path = cfg.get("class_mapping_file")
    mapping_payload = {}
    if mapping_path and Path(mapping_path).exists():
        with open(mapping_path, "r", encoding="utf-8") as f:
            mapping_payload = yaml.safe_load(f) or {}
    payload = {
        "dataset": cfg["dataset"]["name"],
        "split": cfg["dataset"]["split"],
        "date": datetime.now().isoformat(timespec="seconds"),
        "python_version": platform.python_version(),
        "torch_version": version_of("torch"),
        "ultralytics_version": version_of("ultralytics"),
        "device": cfg.get("device", "auto"),
        "model_weights": cfg["model"]["weights"],
        "model_conf": cfg["model"].get("conf"),
        "tracker": cfg["model"]["tracker"],
        "models": [m["name"] for m in cfg.get("models", [])],
        "weights": {m["name"]: m["weights"] for m in cfg.get("models", [])},
        "fgsm_loss": cfg["fgsm"]["loss"],
        "fgsm_eps": cfg["fgsm"]["eps_values"],
        "fgsm_is_gradient_based": True,
        "fgsm_eps_values": cfg["fgsm"]["eps_values"],
        "fgsm_input_requires_grad": True,
        "fgsm_uses_sign_gradient": True,
        "fgsm_clipping": True,
        "eps_values": cfg["fgsm"]["eps_values"],
        "s_feature_mode": cfg["filtering"]["s_feature_mode"],
        "xai": cfg["xai"],
        "metrics_note": "mAP field is AP@0.5 in the minimal first pipeline",
        "prediction_class_filter": cfg.get("evaluation", {}).get("prediction_class_filter"),
        "prediction_class_filter_mode": cfg.get("evaluation", {}).get("prediction_class_filter_mode"),
        "class_agnostic_eval": cfg.get("evaluation", {}).get("class_agnostic"),
        "detector_pretraining": mapping_payload.get("detector_pretraining", "COCO"),
        "class_mapping": mapping_path,
        "used_coco_classes_for_visdrone": mapping_payload.get("used_coco_classes"),
        "unmapped_visdrone_classes": mapping_payload.get("unmapped_visdrone_classes"),
        "class_mapping_note": "Metrics are computed after explicit COCO-to-VisDrone class filtering/mapping metadata.",
        "tnorm_filter_mode": cfg.get("filtering", {}).get("tnorm_filter_mode"),
        "track_aware_filtering": any(
            mode in {"track_aware", "new_track_suppression"}
            for mode in [cfg.get("filtering", {}).get("tnorm_filter_mode")]
            + list(cfg.get("filtering", {}).get("filter_modes", []) or [])
        ),
        "tau_existing_grid": cfg.get("filtering", {}).get("tau_existing_grid"),
        "tau_new_grid": cfg.get("filtering", {}).get("tau_new_grid"),
        "confirm_age_grid": cfg.get("filtering", {}).get("confirm_age_grid"),
        "confirmed_conf_floor_grid": cfg.get("filtering", {}).get("confirmed_conf_floor_grid"),
        "commit": git_commit(),
    }
    payload.update(extra)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
