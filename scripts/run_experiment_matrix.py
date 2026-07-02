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
    p.add_argument("--fgsm-loss", choices=["class_only", "proxy_conf_box_if_available"], default=None)
    p.add_argument("--sequence-list", default=None)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--fast-metrics", action="store_true")
    p.add_argument("--skip-ablation", action="store_true")
    args = p.parse_args()
    cfg = load_config(args.config)
    if args.fgsm_loss:
        cfg["fgsm"]["loss"] = args.fgsm_loss
    if args.output_dir:
        cfg["outputs"]["results_dir"] = args.output_dir
    run_experiment_matrix(
        cfg,
        model_names=args.models,
        tasks=args.tasks,
        eps_values=args.eps,
        scenarios=args.scenarios,
        class_groups=args.class_groups,
        limit_sequences=args.limit_sequences,
        sequence_list=args.sequence_list,
        fast_metrics=args.fast_metrics,
        build_ablation=not args.skip_ablation,
    )


if __name__ == "__main__":
    main()
