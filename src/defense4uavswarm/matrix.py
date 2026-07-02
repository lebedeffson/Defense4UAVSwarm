from __future__ import annotations

from copy import deepcopy
from itertools import product
import json
from pathlib import Path

import pandas as pd
import yaml

from defense4uavswarm.class_groups import filter_by_group
from defense4uavswarm.config import ensure_dirs, require_packages
from defense4uavswarm.datasets.visdrone import VisDroneDataset, VisDroneDetDataset
from defense4uavswarm.filtering.tnorms import apply_conf_threshold, apply_tnorm
from defense4uavswarm.metadata import write_metadata
from defense4uavswarm.metrics.asr import attack_success_breakdown, attack_success_breakdown_det
from defense4uavswarm.pipeline import build_threshold_selection, evaluate, load_existing, run_detection, save
from defense4uavswarm.recovery import run_track_recovery_calibration, write_recovery_reports


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
    if task == "det":
        return "VisDrone2019-DET-val", VisDroneDetDataset(cfg["dataset"]["root"], "val")
    raise ValueError(f"Unknown task: {task}")


def matrix_row(base: dict, task: str, dataset_subset: str, num_sequences: int, num_frames: int, tau_conf: float | None) -> dict:
    return {
        "task": task,
        "dataset_subset": dataset_subset,
        "tau_conf": tau_conf,
        "fgsm_loss": base.get("fgsm_loss"),
        "num_sequences": num_sequences,
        "num_images": num_frames if task == "det" else None,
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


def load_split_sequences(path: str | Path | None, split: str | None) -> list[str] | None:
    if path is None or split is None:
        return None
    with open(path, "r", encoding="utf-8") as f:
        payload = yaml.safe_load(f) or {}
    value = payload.get(split)
    if value is None:
        raise ValueError(f"Split {split!r} not found in {path}")
    return [str(x) for x in value]


def load_selected_defense(path: str | Path | None) -> dict:
    if path is None:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_or_run_detection(path: Path, cfg: dict, ds: VisDroneDataset, scenario: str, eps: float, sequences: list[str], limit_sequences: int | None, enable_tracking: bool = True) -> pd.DataFrame:
    existing = load_existing(path)
    if existing is not None:
        return existing
    frame = run_detection(cfg, ds, scenario, eps, limit_sequences=None if sequences else limit_sequences, sequence_ids=sequences, enable_tracking=enable_tracking)
    save(frame, path)
    return frame


def load_threshold_selection(path: Path, task: str, model_name: str) -> tuple[float, dict[str, float], pd.DataFrame] | None:
    existing = load_existing(path)
    if existing is None or existing.empty:
        return None
    sub = existing[(existing["task"] == task) & (existing["model_name"] == model_name)] if {"task", "model_name"}.issubset(existing.columns) else existing
    selected = sub[sub["selected"] == True]
    conf = selected[selected["mode"] == "S_naive"]
    s2 = selected[selected["mode"].isin(["S2", "S2_conf"])]
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
    limit_images: int | None = None,
    fast_metrics: bool = False,
    build_ablation: bool = True,
    split_config: str | Path | None = None,
    split: str | None = None,
    k_variants: list[str] | None = None,
    filter_modes: list[str] | None = None,
    alpha_scales: list[float] | None = None,
    betas: list[float | None] | None = None,
    use_selected_defense: str | Path | None = None,
) -> None:
    require_packages(["cv2", "pandas", "torch", "ultralytics", "tqdm"])
    ensure_dirs(cfg)
    results = Path(cfg["outputs"]["results_dir"])
    rows, threshold_rows = [], []
    robust_rows, kinematic_rows = [], []
    selected_params: dict[str, dict] = {}
    ablation_frames = []
    models = selected_models(cfg, model_names)
    selected_defense = load_selected_defense(use_selected_defense)

    for task in tasks:
        dataset_subset, ds = task_dataset(cfg, task)
        is_det = task == "det"
        tracking_enabled = not is_det
        selected_sequences = load_split_sequences(split_config, split) or load_sequence_list(sequence_list, task)
        limit_items = limit_images if is_det and limit_images is not None else limit_sequences
        sequences = selected_sequences or (ds.sequence_ids()[:limit_items] if limit_items else ds.sequence_ids())
        gt = ds.all_annotations(sequences)
        num_frames = sum(1 for _ in ds.frames(sequences))
        for model in models:
            mc = deepcopy(cfg)
            mc["model"].update(model)
            if is_det:
                mc["filtering"]["tau_grid"] = mc["filtering"].get("det_tau_grid", mc["filtering"]["tau_grid"])
            model_name = model["name"]

            s0 = load_or_run_detection(results / f"{task}_{model_name}_s0_baseline.csv", mc, ds, "S0", 0.0, sequences, limit_items, enable_tracking=tracking_enabled)
            loaded_thresholds = load_threshold_selection(results / "threshold_selection.csv", task, model_name)
            if loaded_thresholds:
                tau_conf, tau_q, threshold_df = loaded_thresholds
                threshold_df["fgsm_loss"] = mc["fgsm"]["loss"]
            else:
                tau_conf, tau_q, threshold_df = build_threshold_selection(gt, s0, mc)
                if is_det:
                    threshold_df["mode"] = threshold_df["mode"].replace({"S2": "S2_conf"})
                threshold_df.insert(0, "fgsm_loss", mc["fgsm"]["loss"])
                threshold_df.insert(0, "model_name", model_name)
                threshold_df.insert(0, "task", task)
            threshold_rows.append(threshold_df)

            for group in class_groups:
                if "s0" in scenarios:
                    rows.append(matrix_row(evaluate(gt, s0, "S0", 0.0, cfg=mc, class_group=group, include_map=not fast_metrics and group == "all", include_tracking=tracking_enabled and group == "all" and not fast_metrics), task, dataset_subset, len(sequences), num_frames, tau_conf))

            for eps in eps_values:
                s1 = load_or_run_detection(results / f"{task}_{model_name}_s1_fgsm_eps_{eps}.csv", mc, ds, "S1", eps, sequences, limit_items, enable_tracking=tracking_enabled)
                naive = apply_conf_threshold(s1, tau_conf)
                save(naive, results / f"{task}_{model_name}_s_naive_eps_{eps}.csv")

                s2_by_norm = {}
                s2_specs_by_name = {}
                if "s2" in scenarios:
                    s2_specs = defense_specs_for_run(
                        cfg=mc,
                        model_name=model_name,
                        eps=eps,
                        tau_q=tau_q,
                        k_variants=k_variants,
                        filter_modes=filter_modes,
                        alpha_scales=alpha_scales,
                        betas=betas,
                        selected_defense=selected_defense,
                        calibration=split == "calibration" and task in {"vid", "mot"},
                    )
                    if split == "calibration" and task in {"vid", "mot"}:
                        cand_rows, chosen = select_robust_defense(gt, s0, s1, mc, model_name, eps, s2_specs, fast_metrics=fast_metrics)
                        robust_rows.extend(cand_rows)
                        kinematic_rows.extend(cand_rows)
                        selected_params.setdefault(model_name, {})[f"eps_{eps}"] = chosen
                    s2_specs_by_name = {spec["name"]: spec for spec in s2_specs}
                    s2_by_norm = {
                        spec["name"]: apply_tnorm(
                            s1,
                            spec["t_norm"],
                            spec["tau_Q"],
                            mode=spec["filter_mode"],
                            k_variant=spec["k_variant"],
                            alpha_scale=spec["alpha_scale"],
                            beta=spec["beta"],
                            preassociation=bool(mc["filtering"].get("preassociation_kinematics", False)) and not is_det,
                            gamma_assoc=float(spec.get("gamma_assoc", mc["filtering"].get("gamma_assoc", 0.2))),
                            k_new_track=float(mc["filtering"].get("k_new_track", 1.0)),
                            tau_soft_update=float(mc["filtering"].get("tau_soft_update", 0.05)),
                            max_missed_frames=int(mc["filtering"].get("max_missed_frames", 5)),
                            soft_conf_floor=float(spec.get("soft_conf_floor") or mc["filtering"].get("soft_conf_floor", 0.0)),
                            reject_patience=int(spec.get("reject_patience", mc["filtering"].get("reject_patience", 1))),
                            tau_existing=spec.get("tau_existing"),
                            tau_new=spec.get("tau_new"),
                            confirm_age=int(spec.get("confirm_age", mc["filtering"].get("confirm_age", 3))),
                            max_confirm_missed=int(mc["filtering"].get("max_confirm_missed", 1)),
                            confirmed_conf_floor=float(spec.get("confirmed_conf_floor") or mc["filtering"].get("confirmed_conf_floor", 0.03)),
                            risk_tau=float(spec.get("risk_tau", mc["filtering"].get("risk_tau_grid", [0.7])[0])),
                            new_conf_tau=float(spec.get("new_conf_tau", mc["filtering"].get("new_conf_tau_grid", [0.2])[0])),
                            risk_weights=tuple(mc["filtering"].get("risk_weights", [0.4, 0.3, 0.3])),
                            penalty_strength=float(spec.get("penalty_strength", mc["filtering"].get("penalty_strength_grid", [0.3])[0])),
                            min_penalty=float(spec.get("min_penalty", mc["filtering"].get("min_penalty_grid", [0.7])[0])),
                            max_reject_per_frame=int(spec.get("max_reject_per_frame", mc["filtering"].get("max_reject_per_frame_grid", [1])[0])),
                        )
                        for spec in s2_specs
                    }
                    for name, frame in s2_by_norm.items():
                        suffix = "s2_conf" if is_det else "s2"
                        save(frame, results / f"{task}_{model_name}_{suffix}_{name}_eps_{eps}.csv")

                recovered_scenarios: dict[str, pd.DataFrame] = {}
                if "s3_track_recovery" in scenarios and task in {"vid", "mot"}:
                    rec_cfg = mc.get("recovery", {})
                    s3, chosen, _ = run_track_recovery_calibration(
                        gt,
                        s0,
                        s1,
                        mc,
                        model_name,
                        eps,
                        results,
                        rec_cfg.get("modes", ["hold_last", "constant_velocity"]),
                        rec_cfg.get("horizons", [2, 3]),
                        rec_cfg.get("decays", [0.8, 0.9]),
                        rec_cfg.get("confirm_ages", [3]),
                        rec_cfg.get("max_recovered_tracks_per_frame", [10]),
                        scenario_name="S3_track_recovery",
                        safe=False,
                    )
                    recovered_scenarios["S3_track_recovery"] = s3

                if "s3_safe_recovery" in scenarios and task in {"vid", "mot"}:
                    rec_cfg = mc.get("recovery", {})
                    s3_safe, chosen, _ = run_track_recovery_calibration(
                        gt,
                        s0,
                        s1,
                        mc,
                        model_name,
                        eps,
                        results,
                        rec_cfg.get("modes", ["constant_velocity"]),
                        rec_cfg.get("horizons", [1, 2]),
                        rec_cfg.get("decays", [0.6, 0.7]),
                        rec_cfg.get("confirm_ages", [5, 8]),
                        rec_cfg.get("max_recovered_tracks_per_frame", [3, 5]),
                        scenario_name="S3_safe_recovery",
                        safe=True,
                        min_recent_confidences=rec_cfg.get("min_recent_confidences", [0.15, 0.20]),
                        min_mean_track_confidences=rec_cfg.get("min_mean_track_confidences", [0.15]),
                        recovered_conf_floors=rec_cfg.get("recovered_conf_floors", [0.05, 0.08]),
                        weak_detection_support_values=rec_cfg.get("weak_detection_support", [True]),
                        weak_confidence_mins=rec_cfg.get("weak_confidence_mins", [0.01, 0.03]),
                        weak_iou_mins=rec_cfg.get("weak_iou_mins", [0.30, 0.40]),
                        recovery_cooldowns=rec_cfg.get("recovery_cooldowns", [2]),
                    )
                    recovered_scenarios["S3_safe_recovery"] = s3_safe

                if recovered_scenarios:
                    write_recovery_reports(gt, s0, s1, list(recovered_scenarios.items()), mc, eps, results)

                for group in class_groups:
                    if "s1" in scenarios:
                        m = evaluate(gt, s1, "S1", eps, cfg=mc, class_group=group, include_map=not fast_metrics and group == "all", include_tracking=tracking_enabled and group == "all" and not fast_metrics)
                        m["ASR"] = None
                        if is_det and group == "all":
                            m.update(attack_success_breakdown_det(gt, s0, s1))
                        rows.append(matrix_row(m, task, dataset_subset, len(sequences), num_frames, tau_conf))
                    if "s_naive" in scenarios:
                        m = evaluate(gt, naive, "S_naive", eps, "confidence", tau_conf, cfg=mc, class_group=group, include_map=not fast_metrics and group == "all", include_tracking=tracking_enabled and group == "all" and not fast_metrics)
                        m["ASR"] = None
                        if is_det and group == "all":
                            m.update(attack_success_breakdown_det(gt, s0, naive))
                        rows.append(matrix_row(m, task, dataset_subset, len(sequences), num_frames, tau_conf))
                    if "s2" in scenarios:
                        for name, frame in s2_by_norm.items():
                            spec = s2_specs_by_name[name]
                            scenario_name = "S2_conf" if is_det else "S2"
                            m = evaluate(gt, frame, scenario_name, eps, spec["t_norm"], spec["tau_Q"], cfg=mc, class_group=group, include_map=not fast_metrics and group == "all", include_tracking=tracking_enabled and group == "all" and not fast_metrics)
                            m["ASR"] = None
                            if is_det and group == "all":
                                m.update(attack_success_breakdown_det(gt, s0, frame))
                            elif group == "all":
                                breakdown = attack_success_breakdown(gt, s0, frame)
                                m.update(breakdown)
                                m["ASR_track"] = breakdown["asr_total"]
                            rows.append(matrix_row(m, task, dataset_subset, len(sequences), num_frames, tau_conf))
                    if "s3_track_recovery" in scenarios and task in {"vid", "mot"}:
                        m = evaluate(gt, recovered_scenarios["S3_track_recovery"], "S3_track_recovery", eps, cfg=mc, class_group=group, include_map=not fast_metrics and group == "all", include_tracking=tracking_enabled and group == "all" and not fast_metrics)
                        rows.append(matrix_row(m, task, dataset_subset, len(sequences), num_frames, tau_conf))
                    if "s3_safe_recovery" in scenarios and task in {"vid", "mot"}:
                        m = evaluate(gt, recovered_scenarios["S3_safe_recovery"], "S3_safe_recovery", eps, cfg=mc, class_group=group, include_map=not fast_metrics and group == "all", include_tracking=tracking_enabled and group == "all" and not fast_metrics)
                        rows.append(matrix_row(m, task, dataset_subset, len(sequences), num_frames, tau_conf))

                if build_ablation:
                    ablation_frames.append(build_ablation_rows(gt, s0, s1, naive, s2_by_norm, mc, model_name, eps, class_groups, detection_only=is_det))

    research = pd.DataFrame(rows)
    research["fgsm_loss"] = cfg["fgsm"]["loss"]
    save(research, results / "research_matrix.csv")
    save(research, results / "summary_metrics.csv")
    if threshold_rows:
        save(pd.concat(threshold_rows, ignore_index=True), results / "threshold_selection.csv")
    if robust_rows:
        robust = pd.DataFrame(robust_rows)
        save(robust, results / "robust_threshold_selection.csv")
        save(robust, results / "kinematics_param_selection.csv")
        write_selected_defense(results / "selected_defense_params.yaml", selected_params)
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
            "task_det_note": "DET branch is detection-only; k_i=s_i=x_i=1; tracking metrics are disabled." if "det" in tasks else None,
            "kinematics_enabled": False if tasks == ["det"] else None,
            "tracking_enabled": False if tasks == ["det"] else None,
            "threshold_grid_s_naive": cfg.get("filtering", {}).get("confidence_grid"),
            "threshold_grid_s2": cfg.get("filtering", {}).get("tau_grid"),
            "threshold_grid_s2_conf_det": cfg.get("filtering", {}).get("det_tau_grid"),
            "limit_sequences": limit_sequences,
            "limit_images": limit_images,
            "sequence_list": str(sequence_list) if sequence_list else None,
            "vid_split_file": str(split_config) if split_config else None,
            "tuning_split": "calibration" if split == "calibration" else None,
            "evaluation_split": "holdout" if split == "holdout" else None,
            "split": split,
            "selected_sequences": sequences,
        },
    )
    if ("s3_track_recovery" in scenarios or "s3_safe_recovery" in scenarios) and (results / "selected_defense_params.yaml").exists():
        with open(results / "selected_defense_params.yaml", "r", encoding="utf-8") as f:
            selection = yaml.safe_load(f) or {}
        with open(results / "metadata.json", "r", encoding="utf-8") as f:
            metadata = json.load(f)
        metadata.update(
            {
                "stage": "v1.7" if "s3_safe_recovery" in scenarios else "v1.6",
                "scenario": "S3_safe_recovery" if "s3_safe_recovery" in scenarios else "S3_track_recovery",
                "selection_status": selection.get("selection_status"),
                "holdout_allowed": bool(selection.get("holdout_allowed", False)),
                "holdout_run": False,
            }
        )
        (results / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def defense_specs_for_run(
    cfg: dict,
    model_name: str,
    eps: float,
    tau_q: dict[str, float],
    k_variants: list[str] | None,
    filter_modes: list[str] | None,
    alpha_scales: list[float] | None,
    betas: list[float | None] | None,
    selected_defense: dict,
    calibration: bool,
) -> list[dict]:
    selected = selected_defense.get("selected", {}) if selected_defense else {}
    model_selected = selected.get(model_name, {})
    eps_selected = model_selected.get(f"eps_{eps}") or model_selected.get("model_level")
    if eps_selected and not calibration:
        return [normalize_defense_spec(eps_selected, tau_q)]
    if calibration:
        specs = []
        tau_existing_grid = cfg["filtering"].get("tau_existing_grid", cfg["filtering"].get("robust_tau_grid", cfg["filtering"]["tau_grid"]))
        tau_new_grid = cfg["filtering"].get("tau_new_grid", cfg["filtering"].get("robust_tau_grid", cfg["filtering"]["tau_grid"]))
        for tnorm, kv, alpha, gamma, mode, beta, tau_existing, tau_new, risk_tau, new_conf_tau, penalty_strength, min_penalty, max_reject, confirm_age, floor, patience in product(
            cfg["filtering"]["t_norms"],
            k_variants or cfg["filtering"].get("k_variants", ["center"]),
            alpha_scales or cfg["filtering"].get("alpha_scales", [1.0]),
            cfg["filtering"].get("gamma_assoc_grid", [cfg["filtering"].get("gamma_assoc", 0.2)]),
            filter_modes or [cfg["filtering"].get("tnorm_filter_mode", "hard_filter")],
            betas or [None],
            tau_existing_grid,
            tau_new_grid,
            cfg["filtering"].get("risk_tau_grid", [0.7]),
            cfg["filtering"].get("new_conf_tau_grid", [0.2]),
            cfg["filtering"].get("penalty_strength_grid", [0.3]),
            cfg["filtering"].get("min_penalty_grid", [0.7]),
            cfg["filtering"].get("max_reject_per_frame_grid", [1]),
            cfg["filtering"].get("confirm_age_grid", [cfg["filtering"].get("confirm_age", 3)]),
            cfg["filtering"].get("confirmed_conf_floor_grid", [cfg["filtering"].get("confirmed_conf_floor", 0.03)]),
            cfg["filtering"].get("reject_patience_grid", [cfg["filtering"].get("reject_patience", 1)]),
        ):
            if float(tau_new) < float(tau_existing):
                continue
            if mode != "suspicious_soft_penalty" and (float(penalty_strength) != float(cfg["filtering"].get("penalty_strength_grid", [0.3])[0]) or float(min_penalty) != float(cfg["filtering"].get("min_penalty_grid", [0.7])[0])):
                continue
            if mode != "top_risk_only_suppression" and int(max_reject) != int(cfg["filtering"].get("max_reject_per_frame_grid", [1])[0]):
                continue
            specs.append(
                {
                    "name": f"{tnorm}_{kv}_a{alpha:g}_g{gamma:g}_{mode}_b{('none' if beta is None else beta)}_te{tau_existing:g}_tn{tau_new:g}_ca{confirm_age}",
                    "t_norm": tnorm,
                    "tau_Q": float(tau_existing),
                    "tau_existing": float(tau_existing),
                    "tau_new": float(tau_new),
                    "risk_tau": float(risk_tau),
                    "new_conf_tau": float(new_conf_tau),
                    "penalty_strength": float(penalty_strength),
                    "min_penalty": float(min_penalty),
                    "max_reject_per_frame": int(max_reject),
                    "filter_mode": mode,
                    "k_variant": kv,
                    "alpha_scale": float(alpha),
                    "gamma_assoc": float(gamma),
                    "soft_conf_floor": None,
                    "confirmed_conf_floor": float(floor),
                    "confirm_age": int(confirm_age),
                    "reject_patience": int(patience),
                    "beta": beta,
                }
            )
        return specs
    return [
        {
            "name": name,
            "t_norm": name,
            "tau_Q": tau,
            "tau_existing": tau,
            "tau_new": tau,
            "filter_mode": cfg["filtering"].get("tnorm_filter_mode", "hard_filter"),
            "k_variant": cfg["filtering"].get("k_variant", "center"),
            "alpha_scale": float(cfg["filtering"].get("alpha_scale", 1.0)),
            "gamma_assoc": float(cfg["filtering"].get("gamma_assoc", 0.2)),
            "soft_conf_floor": cfg["filtering"].get("soft_conf_floor", 0.0),
            "confirmed_conf_floor": cfg["filtering"].get("confirmed_conf_floor", 0.03),
            "confirm_age": int(cfg["filtering"].get("confirm_age", 3)),
            "risk_tau": float(cfg["filtering"].get("risk_tau_grid", [0.7])[0]),
            "new_conf_tau": float(cfg["filtering"].get("new_conf_tau_grid", [0.2])[0]),
            "penalty_strength": float(cfg["filtering"].get("penalty_strength_grid", [0.3])[0]),
            "min_penalty": float(cfg["filtering"].get("min_penalty_grid", [0.7])[0]),
            "max_reject_per_frame": int(cfg["filtering"].get("max_reject_per_frame_grid", [1])[0]),
            "reject_patience": int(cfg["filtering"].get("reject_patience", 1)),
            "beta": None,
        }
        for name, tau in tau_q.items()
    ]


def normalize_defense_spec(spec: dict, tau_q: dict[str, float]) -> dict:
    t_norm = spec.get("t_norm") or next(iter(tau_q))
    tau = float(spec.get("tau_Q", tau_q.get(t_norm, next(iter(tau_q.values())))))
    return {
        "name": f"{t_norm}_{spec.get('k_variant', 'center')}_selected",
        "t_norm": t_norm,
        "tau_Q": tau,
        "tau_existing": float(spec.get("tau_existing", tau)),
        "tau_new": float(spec.get("tau_new", tau)),
        "filter_mode": spec.get("filter_mode", "hard_filter"),
        "k_variant": spec.get("k_variant", "center"),
        "alpha_scale": float(spec.get("alpha_scale", 1.0)),
        "gamma_assoc": float(spec.get("gamma_assoc", 0.2)),
        "soft_conf_floor": spec.get("soft_conf_floor"),
        "confirmed_conf_floor": spec.get("confirmed_conf_floor"),
        "confirm_age": int(spec.get("confirm_age", 3)),
        "risk_tau": float(spec.get("risk_tau", 0.7)),
        "new_conf_tau": float(spec.get("new_conf_tau", 0.2)),
        "penalty_strength": float(spec.get("penalty_strength", 0.3)),
        "min_penalty": float(spec.get("min_penalty", 0.7)),
        "max_reject_per_frame": int(spec.get("max_reject_per_frame", 1)),
        "reject_patience": int(spec.get("reject_patience", 1)),
        "beta": spec.get("beta"),
    }


def select_robust_defense(gt, s0, s1, cfg, model_name: str, eps: float, specs: list[dict], fast_metrics: bool = False) -> tuple[list[dict], dict]:
    clean_base = evaluate(gt, s0, "S0", 0.0, cfg=cfg, include_map=False, include_tracking=not fast_metrics)
    rows = []
    limit = float(cfg["filtering"].get("robust_clean_f1_drop_limit", 0.02))
    weights = cfg["filtering"].get("robust_score_weights", {})
    for spec in specs:
        clean = apply_tnorm(
            s0,
            spec["t_norm"],
            spec["tau_Q"],
            mode=spec["filter_mode"],
            k_variant=spec["k_variant"],
            alpha_scale=spec["alpha_scale"],
            beta=spec["beta"],
            preassociation=bool(cfg["filtering"].get("preassociation_kinematics", False)),
            gamma_assoc=float(cfg["filtering"].get("gamma_assoc", 0.2)),
            k_new_track=float(cfg["filtering"].get("k_new_track", 1.0)),
            tau_soft_update=float(cfg["filtering"].get("tau_soft_update", 0.05)),
            tau_existing=spec.get("tau_existing"),
            tau_new=spec.get("tau_new"),
            confirm_age=int(spec.get("confirm_age", cfg["filtering"].get("confirm_age", 3))),
            max_confirm_missed=int(cfg["filtering"].get("max_confirm_missed", 1)),
            confirmed_conf_floor=float(spec.get("confirmed_conf_floor") or cfg["filtering"].get("confirmed_conf_floor", 0.03)),
            reject_patience=int(spec.get("reject_patience", cfg["filtering"].get("reject_patience", 1))),
            risk_tau=float(spec.get("risk_tau", cfg["filtering"].get("risk_tau_grid", [0.7])[0])),
            new_conf_tau=float(spec.get("new_conf_tau", cfg["filtering"].get("new_conf_tau_grid", [0.2])[0])),
            risk_weights=tuple(cfg["filtering"].get("risk_weights", [0.4, 0.3, 0.3])),
            penalty_strength=float(spec.get("penalty_strength", cfg["filtering"].get("penalty_strength_grid", [0.3])[0])),
            min_penalty=float(spec.get("min_penalty", cfg["filtering"].get("min_penalty_grid", [0.7])[0])),
            max_reject_per_frame=int(spec.get("max_reject_per_frame", cfg["filtering"].get("max_reject_per_frame_grid", [1])[0])),
        )
        attack = apply_tnorm(
            s1,
            spec["t_norm"],
            spec["tau_Q"],
            mode=spec["filter_mode"],
            k_variant=spec["k_variant"],
            alpha_scale=spec["alpha_scale"],
            beta=spec["beta"],
            preassociation=bool(cfg["filtering"].get("preassociation_kinematics", False)),
            gamma_assoc=float(cfg["filtering"].get("gamma_assoc", 0.2)),
            k_new_track=float(cfg["filtering"].get("k_new_track", 1.0)),
            tau_soft_update=float(cfg["filtering"].get("tau_soft_update", 0.05)),
            tau_existing=spec.get("tau_existing"),
            tau_new=spec.get("tau_new"),
            confirm_age=int(spec.get("confirm_age", cfg["filtering"].get("confirm_age", 3))),
            max_confirm_missed=int(cfg["filtering"].get("max_confirm_missed", 1)),
            confirmed_conf_floor=float(spec.get("confirmed_conf_floor") or cfg["filtering"].get("confirmed_conf_floor", 0.03)),
            reject_patience=int(spec.get("reject_patience", cfg["filtering"].get("reject_patience", 1))),
            risk_tau=float(spec.get("risk_tau", cfg["filtering"].get("risk_tau_grid", [0.7])[0])),
            new_conf_tau=float(spec.get("new_conf_tau", cfg["filtering"].get("new_conf_tau_grid", [0.2])[0])),
            risk_weights=tuple(cfg["filtering"].get("risk_weights", [0.4, 0.3, 0.3])),
            penalty_strength=float(spec.get("penalty_strength", cfg["filtering"].get("penalty_strength_grid", [0.3])[0])),
            min_penalty=float(spec.get("min_penalty", cfg["filtering"].get("min_penalty_grid", [0.7])[0])),
            max_reject_per_frame=int(spec.get("max_reject_per_frame", cfg["filtering"].get("max_reject_per_frame_grid", [1])[0])),
        )
        cm = evaluate(gt, clean, "S2_clean", 0.0, spec["t_norm"], spec["tau_Q"], cfg=cfg, include_map=False, include_tracking=not fast_metrics)
        am = evaluate(gt, attack, "S2", eps, spec["t_norm"], spec["tau_Q"], cfg=cfg, include_map=False, include_tracking=not fast_metrics)
        breakdown = attack_success_breakdown(gt, s0, attack)
        asr = breakdown["asr_total"]
        idf1 = am.get("IDF1")
        clean_drop = clean_base["F1"] - cm["F1"]
        num_frames = max(1, len(gt[["sequence_id", "frame_id"]].drop_duplicates()))
        idf1_val = 0.0 if idf1 is None or pd.isna(idf1) else float(idf1)
        idsw_rate = (0.0 if am.get("IDSW") is None or pd.isna(am.get("IDSW")) else float(am["IDSW"])) / num_frames
        break_rate = (0.0 if am.get("track_breaks") is None or pd.isna(am.get("track_breaks")) else float(am["track_breaks"])) / num_frames
        score = (
            float(weights.get("F1_attack", 0.30)) * am["F1"]
            + float(weights.get("IDF1_attack", 0.25)) * idf1_val
            + float(weights.get("ASR_any", -0.15)) * asr
            + float(weights.get("IDSW_rate", -0.15)) * idsw_rate
            + float(weights.get("track_break_rate", -0.10)) * break_rate
            + float(weights.get("clean_F1_drop", -0.05)) * clean_drop
        )
        row = {
            "model_name": model_name,
            "eps": eps,
            "k_variant": spec["k_variant"],
            "alpha_scale": spec["alpha_scale"],
            "beta": spec["beta"],
            "filter_mode": spec["filter_mode"],
            "t_norm": spec["t_norm"],
            "tau_Q": spec["tau_Q"],
            "tau_existing": spec.get("tau_existing", spec["tau_Q"]),
            "tau_new": spec.get("tau_new", spec["tau_Q"]),
            "confirm_age": spec.get("confirm_age"),
            "confirmed_conf_floor": spec.get("confirmed_conf_floor"),
            "reject_patience": spec.get("reject_patience"),
            "risk_tau": spec.get("risk_tau"),
            "new_conf_tau": spec.get("new_conf_tau"),
            "penalty_strength": spec.get("penalty_strength"),
            "min_penalty": spec.get("min_penalty"),
            "max_reject_per_frame": spec.get("max_reject_per_frame"),
            "clean_F1": cm["F1"],
            "attack_F1": am["F1"],
            "attack_IDF1": idf1,
            "attack_ASR": asr,
            "clean_drop": clean_drop,
            "rejection_rate": am["rejection_rate"],
            "IDSW_rate": idsw_rate,
            "track_break_rate": break_rate,
            "robust_score": score,
            "passes_clean_constraint": clean_drop <= limit,
            "passes_rejection_constraint": am["rejection_rate"] >= float(cfg["filtering"].get("min_attack_rejection_rate", 0.01)),
            "selected": False,
        }
        rows.append(row)
    eligible = [r for r in rows if r["passes_clean_constraint"] and r["passes_rejection_constraint"]]
    pool = eligible or rows
    best = sorted(pool, key=lambda r: (-r["robust_score"], r.get("tau_new", r["tau_Q"]), r["alpha_scale"]))[0]
    for row in rows:
        row["selected"] = bool(eligible) and all(row[k] == best[k] for k in ["k_variant", "alpha_scale", "beta", "filter_mode", "t_norm", "tau_Q", "tau_existing", "tau_new", "reject_patience"])
        row["reason"] = "" if row["selected"] else ("no_candidate_with_nonzero_rejection" if not eligible else "")
    selected = {
        k: best[k]
        for k in [
            "k_variant",
            "alpha_scale",
            "beta",
            "filter_mode",
            "t_norm",
            "tau_Q",
            "tau_existing",
            "tau_new",
            "risk_tau",
            "new_conf_tau",
            "penalty_strength",
            "min_penalty",
            "max_reject_per_frame",
            "confirm_age",
            "confirmed_conf_floor",
            "reject_patience",
        ]
    }
    return rows, selected


def write_selected_defense(path: Path, selected_params: dict[str, dict]) -> None:
    payload = {"selection_scope": "model_eps_level", "selected": selected_params}
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False)


def build_ablation_rows(gt, s0, s1, naive, s2_by_norm, cfg, model_name: str, eps: float, groups: list[str], detection_only: bool = False) -> pd.DataFrame:
    rows = []
    variants = [("S1", "fgsm_unprotected", s1), ("S_naive", "confidence_only", naive)]
    if not detection_only:
        ck = s1.copy()
        ck["Q_i"] = 0.5 * ck["c_i"].astype(float) + 0.5 * ck["k_i"].astype(float)
        ck["accepted"] = ck["Q_i"] >= 0.5
        variants.append(("S2_c", "confidence_kinematics_avg", ck))
    for name, frame in s2_by_norm.items():
        scenario = "S2_conf" if detection_only else "S2"
        prefix = "confidence" if detection_only else "confidence_kinematics"
        variants.append((scenario, f"{prefix}_{name}", frame))
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
