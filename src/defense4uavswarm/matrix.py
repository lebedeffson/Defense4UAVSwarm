from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pandas as pd
import yaml

from defense4uavswarm.class_groups import filter_by_group
from defense4uavswarm.config import ensure_dirs, require_packages
from defense4uavswarm.datasets.visdrone import VisDroneDataset
from defense4uavswarm.filtering.tnorms import apply_conf_threshold, apply_tnorm
from defense4uavswarm.metadata import write_metadata
from defense4uavswarm.metrics.asr import attack_success_rate
from defense4uavswarm.pipeline import build_threshold_selection, evaluate, load_existing, run_detection, save


def selected_models(cfg: dict, names: list[str]) -> list[dict]:
    all_models = {m["name"]: m for m in cfg.get("models", [])}
    if not all_models:
        all_models[cfg["model"].get("name", Path(cfg["model"]["weights"]).stem)] = cfg["model"]
    return [all_models[n] for n in names]


def task_dataset(cfg: dict, task: str) -> tuple[str, VisDroneDataset]:
    if task == "vid":
        return "VisDrone2019-VID-val", VisDroneDataset(cfg["dataset"]["root"], "val", subset_hint="VID")
    if task == "mot":
        return "VisDrone2019-MOT-val", VisDroneDataset(cfg["dataset"]["root"], "val", subset_hint="MOT")
    raise ValueError(f"Unknown task: {task}")


def matrix_row(base: dict, task: str, dataset_subset: str, num_sequences: int, num_frames: int, tau_conf: float | None) -> dict:
    return {
        "task": task,
        "dataset_subset": dataset_subset,
        "tau_conf": tau_conf,
        "fgsm_loss": base.get("fgsm_loss"),
        "num_sequences": num_sequences,
        "num_frames": num_frames,
        **base,
    }


def load_sequence_list(path: str | Path | None, task: str | None = None) -> list[str] | None:
    if path is None:
        return None
    with open(path, "r", encoding="utf-8") as f:
        payload = yaml.safe_load(f)
    if isinstance(payload, list):
        return [str(x) for x in payload]
    if isinstance(payload, dict):
        value = payload.get(task or "") or payload.get("sequences") or payload.get("vid")
        if value is None:
            return None
        return [str(x) for x in value]
    raise ValueError(f"Unsupported sequence list format: {path}")


def load_or_run_detection(path: Path, cfg: dict, ds: VisDroneDataset, scenario: str, eps: float, sequences: list[str], limit_sequences: int | None) -> pd.DataFrame:
    existing = load_existing(path)
    if existing is not None:
        return existing
    frame = run_detection(cfg, ds, scenario, eps, limit_sequences=None if sequences else limit_sequences, sequence_ids=sequences)
    save(frame, path)
    return frame


def load_threshold_selection(path: Path, task: str, model_name: str) -> tuple[float, dict[str, float], pd.DataFrame] | None:
    existing = load_existing(path)
    if existing is None or existing.empty:
        return None
    sub = existing[(existing["task"] == task) & (existing["model_name"] == model_name)] if {"task", "model_name"}.issubset(existing.columns) else existing
    selected = sub[sub["selected"] == True]
    conf = selected[selected["mode"] == "S_naive"]
    s2 = selected[selected["mode"] == "S2"]
    if conf.empty or s2.empty:
        return None
    tau_conf = float(conf.iloc[0]["tau"])
    tau_q = {str(r.t_norm): float(r.tau) for r in s2.itertuples()}
    return tau_conf, tau_q, sub.copy()


def run_experiment_matrix(
    cfg: dict,
    model_names: list[str],
    tasks: list[str],
    eps_values: list[float],
    scenarios: list[str],
    class_groups: list[str],
    limit_sequences: int | None,
    sequence_list: str | Path | None = None,
    fast_metrics: bool = False,
    build_ablation: bool = True,
) -> None:
    require_packages(["cv2", "pandas", "torch", "ultralytics", "tqdm"])
    ensure_dirs(cfg)
    results = Path(cfg["outputs"]["results_dir"])
    rows, threshold_rows = [], []
    ablation_frames = []
    models = selected_models(cfg, model_names)

    for task in tasks:
        dataset_subset, ds = task_dataset(cfg, task)
        selected_sequences = load_sequence_list(sequence_list, task)
        sequences = selected_sequences or (ds.sequence_ids()[:limit_sequences] if limit_sequences else ds.sequence_ids())
        gt = ds.all_annotations(sequences)
        num_frames = sum(1 for _ in ds.frames(sequences))
        for model in models:
            mc = deepcopy(cfg)
            mc["model"].update(model)
            model_name = model["name"]

            s0 = load_or_run_detection(results / f"{task}_{model_name}_s0_baseline.csv", mc, ds, "S0", 0.0, sequences, limit_sequences)
            loaded_thresholds = load_threshold_selection(results / "threshold_selection.csv", task, model_name)
            if loaded_thresholds:
                tau_conf, tau_q, threshold_df = loaded_thresholds
                threshold_df["fgsm_loss"] = mc["fgsm"]["loss"]
            else:
                tau_conf, tau_q, threshold_df = build_threshold_selection(gt, s0, mc)
                threshold_df.insert(0, "fgsm_loss", mc["fgsm"]["loss"])
                threshold_df.insert(0, "model_name", model_name)
                threshold_df.insert(0, "task", task)
            threshold_rows.append(threshold_df)

            for group in class_groups:
                if "s0" in scenarios:
                    rows.append(matrix_row(evaluate(gt, s0, "S0", 0.0, cfg=mc, class_group=group, include_map=not fast_metrics and group == "all", include_tracking=group == "all" and not fast_metrics), task, dataset_subset, len(sequences), num_frames, tau_conf))

            for eps in eps_values:
                s1 = load_or_run_detection(results / f"{task}_{model_name}_s1_fgsm_eps_{eps}.csv", mc, ds, "S1", eps, sequences, limit_sequences)
                naive = apply_conf_threshold(s1, tau_conf)
                save(naive, results / f"{task}_{model_name}_s_naive_eps_{eps}.csv")

                s2_by_norm = {name: apply_tnorm(s1, name, tau) for name, tau in tau_q.items()}
                for name, frame in s2_by_norm.items():
                    save(frame, results / f"{task}_{model_name}_s2_{name}_eps_{eps}.csv")

                for group in class_groups:
                    if "s1" in scenarios:
                        m = evaluate(gt, s1, "S1", eps, cfg=mc, class_group=group, include_map=not fast_metrics and group == "all", include_tracking=group == "all" and not fast_metrics)
                        m["ASR"] = None
                        rows.append(matrix_row(m, task, dataset_subset, len(sequences), num_frames, tau_conf))
                    if "s_naive" in scenarios:
                        m = evaluate(gt, naive, "S_naive", eps, "confidence", tau_conf, cfg=mc, class_group=group, include_map=not fast_metrics and group == "all", include_tracking=group == "all" and not fast_metrics)
                        m["ASR"] = None
                        rows.append(matrix_row(m, task, dataset_subset, len(sequences), num_frames, tau_conf))
                    if "s2" in scenarios:
                        for name, frame in s2_by_norm.items():
                            m = evaluate(gt, frame, "S2", eps, name, tau_q[name], cfg=mc, class_group=group, include_map=not fast_metrics and group == "all", include_tracking=group == "all" and not fast_metrics)
                            m["ASR"] = None
                            rows.append(matrix_row(m, task, dataset_subset, len(sequences), num_frames, tau_conf))

                if build_ablation:
                    ablation_frames.append(build_ablation_rows(gt, s0, s1, naive, s2_by_norm, mc, model_name, eps, class_groups))

    research = pd.DataFrame(rows)
    research["fgsm_loss"] = cfg["fgsm"]["loss"]
    save(research, results / "research_matrix.csv")
    save(research, results / "summary_metrics.csv")
    if threshold_rows:
        save(pd.concat(threshold_rows, ignore_index=True), results / "threshold_selection.csv")
    if ablation_frames:
        ablation = pd.concat(ablation_frames, ignore_index=True)
        save(add_ablation_deltas(ablation), results / "ablation_summary.csv")
    write_metadata(
        results / "metadata.json",
        cfg,
        {
            "models": [m["name"] for m in models],
            "weights": {m["name"]: m["weights"] for m in models},
            "matrix_tasks": tasks,
            "matrix_eps": eps_values,
            "matrix_class_groups": class_groups,
            "limit_sequences": limit_sequences,
            "sequence_list": str(sequence_list) if sequence_list else None,
            "selected_sequences": sequences,
        },
    )


def build_ablation_rows(gt, s0, s1, naive, s2_by_norm, cfg, model_name: str, eps: float, groups: list[str]) -> pd.DataFrame:
    rows = []
    variants = [("S1", "fgsm_unprotected", s1), ("S_naive", "confidence_only", naive)]
    ck = s1.copy()
    ck["Q_i"] = 0.5 * ck["c_i"].astype(float) + 0.5 * ck["k_i"].astype(float)
    ck["accepted"] = ck["Q_i"] >= 0.5
    variants.append(("S2_c", "confidence_kinematics_avg", ck))
    for name, frame in s2_by_norm.items():
        variants.append(("S2", f"confidence_kinematics_{name}", frame))
    for group in groups:
        for scenario, variant, frame in variants:
            m = evaluate(gt, frame, scenario, eps, cfg=cfg, class_group=group, include_map=False, include_tracking=False)
            m["ASR"] = None
            rows.append({"model_name": model_name, "fgsm_loss": cfg["fgsm"]["loss"], "eps": eps, "class_group": group, "filter_variant": variant, **m})
    return pd.DataFrame(rows)


def add_ablation_deltas(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for (model, eps, group), idx in out.groupby(["model_name", "eps", "class_group"]).groups.items():
        sub = out.loc[idx]
        base = sub[sub["filter_variant"] == "fgsm_unprotected"]
        if base.empty:
            continue
        b = base.iloc[0]
        out.loc[idx, "delta_F1_vs_S1"] = out.loc[idx, "F1"] - b["F1"]
        out.loc[idx, "delta_IDF1_vs_S1"] = out.loc[idx, "IDF1"] - b["IDF1"]
        out.loc[idx, "delta_ASR_vs_S1"] = out.loc[idx, "ASR"] - b["ASR"]
    return out
