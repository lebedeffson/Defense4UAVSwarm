#!/usr/bin/env python
from __future__ import annotations

import argparse

from defense4uavswarm.config import load_config
from defense4uavswarm.matrix import run_experiment_matrix


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--models", nargs="+", default=["yolov8n", "yolov8s"])
    p.add_argument("--tasks", nargs="+", default=["vid"])
    p.add_argument("--eps", nargs="+", type=float, default=[0.004, 0.008])
    p.add_argument("--scenarios", nargs="+", default=["s0", "s1", "s_naive", "s2"])
    p.add_argument("--class-groups", nargs="+", default=["all", "vru", "vehicles"])
    p.add_argument("--limit-sequences", type=int, default=3)
    p.add_argument("--limit-images", type=int, default=None)
    p.add_argument("--fgsm-loss", choices=["class_only", "proxy_conf_box_if_available"], default=None)
    p.add_argument("--sequence-list", default=None)
    p.add_argument("--split-config", default=None)
    p.add_argument("--split", choices=["calibration", "holdout"], default=None)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--fgsm-cache-dir", default=None)
    p.add_argument("--fast-metrics", action="store_true")
    p.add_argument("--skip-ablation", action="store_true")
    p.add_argument("--class-agnostic-eval", choices=["true", "false"], default=None)
    filter_choices = ["hard", "soft", "hard_filter", "soft_reweight", "delayed_hard_filter", "track_aware", "new_track_suppression"]
    p.add_argument("--filter-mode", choices=filter_choices, default=None)
    p.add_argument("--filter-modes", nargs="+", choices=filter_choices, default=None)
    p.add_argument("--k-variants", nargs="+", choices=["center", "iou", "combined", "gate", "acc", "robust_min"], default=None)
    p.add_argument("--alpha-scales", nargs="+", type=float, default=None)
    p.add_argument("--gamma-assoc", nargs="+", type=float, default=None)
    p.add_argument("--tau-q-grid", nargs="+", type=float, default=None)
    p.add_argument("--tau-existing-grid", nargs="+", type=float, default=None)
    p.add_argument("--tau-new-grid", nargs="+", type=float, default=None)
    p.add_argument("--confirm-age", nargs="+", type=int, default=None)
    p.add_argument("--soft-conf-floors", nargs="+", type=float, default=None)
    p.add_argument("--confirmed-conf-floors", nargs="+", type=float, default=None)
    p.add_argument("--reject-patience", nargs="+", type=int, default=None)
    p.add_argument("--betas", nargs="+", default=None, help="Temporal smoothing betas; use none to include no smoothing.")
    p.add_argument("--use-selected-defense", default=None)
    args = p.parse_args()
    cfg = load_config(args.config)
    if args.fgsm_loss:
        cfg["fgsm"]["loss"] = args.fgsm_loss
    if args.output_dir:
        cfg["outputs"]["results_dir"] = args.output_dir
    if args.fgsm_cache_dir:
        cfg["fgsm"]["cache_dir"] = args.fgsm_cache_dir
    if args.class_agnostic_eval is not None:
        cfg.setdefault("evaluation", {})["class_agnostic"] = args.class_agnostic_eval == "true"
    if args.filter_mode:
        cfg["filtering"]["tnorm_filter_mode"] = args.filter_mode
    if args.filter_modes:
        cfg["filtering"]["filter_modes"] = args.filter_modes
    if args.tau_q_grid:
        cfg["filtering"]["robust_tau_grid"] = args.tau_q_grid
    if args.tau_existing_grid:
        cfg["filtering"]["tau_existing_grid"] = args.tau_existing_grid
    if args.tau_new_grid:
        cfg["filtering"]["tau_new_grid"] = args.tau_new_grid
    if args.confirm_age:
        cfg["filtering"]["confirm_age_grid"] = args.confirm_age
    if args.gamma_assoc:
        cfg["filtering"]["gamma_assoc_grid"] = args.gamma_assoc
    if args.soft_conf_floors:
        cfg["filtering"]["soft_conf_floor_grid"] = args.soft_conf_floors
    if args.confirmed_conf_floors:
        cfg["filtering"]["confirmed_conf_floor_grid"] = args.confirmed_conf_floors
    if args.reject_patience:
        cfg["filtering"]["reject_patience_grid"] = args.reject_patience
    betas = None
    if args.betas is not None:
        betas = [None if str(x).lower() in {"none", "null", "off"} else float(x) for x in args.betas]
    run_experiment_matrix(
        cfg,
        model_names=args.models,
        tasks=args.tasks,
        eps_values=args.eps,
        scenarios=args.scenarios,
        class_groups=args.class_groups,
        limit_sequences=args.limit_sequences,
        limit_images=args.limit_images,
        sequence_list=args.sequence_list,
        fast_metrics=args.fast_metrics,
        build_ablation=not args.skip_ablation,
        split_config=args.split_config,
        split=args.split,
        k_variants=args.k_variants,
        filter_modes=args.filter_modes,
        alpha_scales=args.alpha_scales,
        betas=betas,
        use_selected_defense=args.use_selected_defense,
    )


if __name__ == "__main__":
    main()
