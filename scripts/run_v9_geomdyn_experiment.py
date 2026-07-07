#!/usr/bin/env python
from __future__ import annotations

import argparse

from defense4uavswarm.v9_geomdyn import run_v9_experiment


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--gt-2d", required=True)
    p.add_argument("--gt-3d", default="")
    p.add_argument("--detections", required=True)
    p.add_argument("--scenarios", nargs="+", required=True)
    p.add_argument("--calibration-scenes", nargs="+", default=[])
    p.add_argument("--holdout-scenes", nargs="+", default=[])
    p.add_argument("--config", default="configs/v9_geomdyn_config.yaml")
    p.add_argument("--output-dir", required=True)
    args = p.parse_args()
    result = run_v9_experiment(args.manifest, args.gt_2d, args.detections, args.scenarios, args.holdout_scenes, args.output_dir, args.config)
    print(f"status=ok rows={len(result['summary'])} output={args.output_dir}/main_comparison_table.csv")


if __name__ == "__main__":
    main()
