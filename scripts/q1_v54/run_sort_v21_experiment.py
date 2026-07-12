#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON = sys.executable


def run(cmd: list[str]) -> None:
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


def git(args: list[str]) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True).strip()
    except Exception:
        return "unavailable"


def copy_if_exists(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.exists():
        shutil.copy2(src, dst)


def write_statistics(loso_dir: Path, out: Path) -> None:
    outer = pd.read_csv(loso_dir / "outer_test_by_sequence.csv")
    required = {"tracker_baseline", "selective_trust_quarantine"}
    seen = set(outer["method"].astype(str))
    missing = required - seen
    complete = {}
    for method in sorted(required):
        complete[method] = int(outer[outer["method"].eq(method)]["outer_fold"].nunique())
    stats = {
        "tracker": "SORT",
        "folds_required": 7,
        "folds_by_method": complete,
        "sort_baseline_complete": complete.get("tracker_baseline", 0) == 7,
        "sort_selective_v21_complete": complete.get("selective_trust_quarantine", 0) == 7,
        "missing_required_methods": sorted(missing),
        "git_commit": git(["rev-parse", "HEAD"]),
    }
    out.write_text(json.dumps(stats, indent=2, sort_keys=True), encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--mot-root", default="outputs/results/q1_final_plus/mot_inputs/yolov8s")
    p.add_argument("--dataset-root", default="data/visdrone/VisDrone2019-VID-val")
    p.add_argument("--config", default="configs/q1_v54/sort.yaml")
    p.add_argument("--output-dir", default="outputs/q1_practical_closure_v7/sort")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--with-curves", action="store_true", help="Also run full corrected operating curves. The first SORT closure only requires LOSO folds.")
    args = p.parse_args()

    out = REPO_ROOT / args.output_dir
    if out.exists() and args.overwrite:
        shutil.rmtree(out)
    if out.exists() and any(out.iterdir()) and not args.overwrite:
        raise SystemExit(f"Output exists; use --overwrite: {out}")
    out.mkdir(parents=True, exist_ok=True)
    tracks = out / "tracker_outputs" / "sort"
    loso = out / "loso"

    run(
        [
            PYTHON,
            "scripts/run_external_sort.py",
            "--mot-root",
            args.mot_root,
            "--dataset-root",
            args.dataset_root,
            "--output-dir",
            tracks.as_posix(),
        ]
    )
    if args.with_curves:
        curves = out / "curves"
        run(
            [
                PYTHON,
                "scripts/q1_v54/run_corrected_operating_curves.py",
                "--config",
                args.config,
                "--output-dir",
                curves.as_posix(),
                "--overwrite",
            ]
        )
    run(
        [
            PYTHON,
            "scripts/q1_v54/run_nested_loso.py",
            "--config",
            args.config,
            "--output-dir",
            loso.as_posix(),
            "--overwrite",
        ]
    )

    copy_if_exists(loso / "outer_test_by_sequence.csv", out / "fold_results.csv")
    copy_if_exists(loso / "outer_test_summary.csv", out / "aggregate_results.csv")
    copy_if_exists(loso / "inner_selection_results.csv", out / "selected_parameters.csv")
    copy_if_exists(loso / "episode_terminal_audit.csv", out / "terminal_state_audit.csv")
    copy_if_exists(tracks / "sort_run_summary.csv", out / "runtime.csv")
    write_statistics(loso, out / "statistics.json")
    manifest = {
        "status": "success",
        "tracker": "SORT",
        "protocol": "q1_selective_quarantine_v21",
        "git_commit": git(["rev-parse", "HEAD"]),
        "git_branch": git(["branch", "--show-current"]),
        "mot_root": args.mot_root,
        "dataset_root": args.dataset_root,
        "config": args.config,
        "outputs": {
            "tracker_outputs": (tracks).as_posix(),
            "fold_results": (out / "fold_results.csv").as_posix(),
            "aggregate_results": (out / "aggregate_results.csv").as_posix(),
            "selected_parameters": (out / "selected_parameters.csv").as_posix(),
            "terminal_state_audit": (out / "terminal_state_audit.csv").as_posix(),
            "statistics": (out / "statistics.json").as_posix(),
            "runtime": (out / "runtime.csv").as_posix(),
        },
    }
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(f"status=ok output={out}")


if __name__ == "__main__":
    main()
