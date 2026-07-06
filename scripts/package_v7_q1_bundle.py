#!/usr/bin/env python
from __future__ import annotations

import argparse
import subprocess
import zipfile
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="outputs/bundles/Defense4UAVSwarm_v7_q1_closure_bundle.zip")
    args = p.parse_args()
    repro = Path("outputs/results/v7_q1_closure/reproducibility")
    repro.mkdir(parents=True, exist_ok=True)
    (repro / "git_info.txt").write_text(run(["git", "log", "-1", "--oneline"]) + "\n" + run(["git", "status", "--short", "--branch"]), encoding="utf-8")
    (repro / "environment.txt").write_text(run(["/home/lebedeffson/Code/venv/bin/python", "--version"]), encoding="utf-8")
    bundle = Path(args.output)
    bundle.parent.mkdir(parents=True, exist_ok=True)
    roots = [
        Path("docs/DATASET_U2U.md"),
        Path("docs/METHOD_TEMPORAL.md"),
        Path("docs/LIMITATIONS_Q1.md"),
        Path("docs/REPRODUCE_U2U.md"),
        Path("configs/selected_s2_temporal.yaml"),
        Path("configs/u2u_dataset_config.yaml"),
        Path("configs/rf_baseline_config.yaml"),
        Path("outputs/results/v7_u2u"),
        Path("outputs/results/v7_q1_closure/reproducibility"),
    ]
    with zipfile.ZipFile(bundle, "w", zipfile.ZIP_DEFLATED) as zf:
        for root in roots:
            if root.is_file():
                zf.write(root, Path("Defense4UAVSwarm_v7_q1_closure_bundle") / root)
            elif root.is_dir():
                for file in sorted(root.rglob("*")):
                    if file.is_file():
                        zf.write(file, Path("Defense4UAVSwarm_v7_q1_closure_bundle") / file)
    print(f"bundle={bundle} size={bundle.stat().st_size}")


def run(cmd: list[str]) -> str:
    result = subprocess.run(cmd, check=False, capture_output=True, text=True)
    return (result.stdout + result.stderr).strip()


if __name__ == "__main__":
    main()
