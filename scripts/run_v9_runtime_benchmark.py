#!/usr/bin/env python
from __future__ import annotations

import argparse

from defense4uavswarm.v9_geomdyn import run_v9_runtime


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--detections", required=True)
    p.add_argument("--scenarios", nargs="+", required=True)
    p.add_argument("--selected-params", default="outputs/results/v9_geomdyn/main/v9_selected_params.yaml")
    p.add_argument("--output-dir", required=True)
    args = p.parse_args()
    run_v9_runtime("outputs/results/v9_geomdyn/main/main_comparison_table.csv", args.output_dir)
    print(f"status=ok output={args.output_dir}/runtime_summary.csv")


if __name__ == "__main__":
    main()
