#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import subprocess
import zipfile
from datetime import datetime
from pathlib import Path

import yaml


ALLOWED_SUFFIXES = {".csv", ".json", ".yaml", ".yml", ".md", ".txt", ".png"}


def sh(cmd: list[str]) -> str:
    try:
        return subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT).strip()
    except Exception as exc:
        return f"{' '.join(cmd)} unavailable: {exc}"


def write_run_files(results: Path, commands: Path | None) -> None:
    results.mkdir(parents=True, exist_ok=True)
    commit = sh(["git", "rev-parse", "HEAD"])
    manifest = {
        "project": "Defense4UAVSwarm",
        "stage": "v1.3",
        "commit": commit,
        "dataset": "VisDrone2019-VID-val",
        "split": "calibration",
        "models": ["yolov8n"],
        "eps": [0.008],
        "scenarios": ["s0", "s1", "s_naive", "s2"],
        "filter_modes": [
            "risk_gated_new_suppression",
            "low_conf_new_suppression",
            "suspicious_label_only",
            "track_aware",
            "new_track_suppression",
        ],
        "holdout_run": False,
        "holdout_reason": "no_safe_non_noop_candidate",
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    (results / "run_manifest.yaml").write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
    metadata = {
        "project": "Defense4UAVSwarm",
        "stage": "v1.3",
        "commit": commit,
        "debug_bundle": True,
        "holdout_run": False,
        "holdout_reason": "no_safe_non_noop_candidate",
        "primary_outputs": [
            "summary_metrics.csv",
            "summary_metrics_tracking_selected.csv",
            "selected_defense_params.yaml",
            "rejection_truth_audit.csv",
            "rejection_truth_summary.csv",
            "oracle_new_fp_filter_summary.csv",
            "status_transition_summary.csv",
        ],
    }
    (results / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    env = [sh(["/home/lebedeffson/Code/venv/bin/python", "-V"]), sh(["/home/lebedeffson/Code/venv/bin/pip", "freeze"]), sh(["nvidia-smi"])]
    (results / "environment.txt").write_text("\n\n".join(env) + "\n", encoding="utf-8")
    git_info = "\n".join([sh(["git", "rev-parse", "HEAD"]), sh(["git", "status", "--short"]), sh(["git", "diff", "--stat"])])
    (results / "git_info.txt").write_text(git_info + "\n", encoding="utf-8")
    if commands and commands.exists():
        (results / "command_history.txt").write_text(commands.read_text(encoding="utf-8"), encoding="utf-8")
    elif not (results / "command_history.txt").exists():
        (results / "command_history.txt").write_text(
            """[DET-val]
see README.md DET commands

[VID-v1.0]
see outputs/results/vid_calibration_v10 metadata

[VID-v1.1]
see outputs/results/vid_calibration_v11 metadata

[VID-v1.2]
python scripts/select_vid_defense.py --config configs/default.yaml --split-config configs/vid_split.yaml --split calibration --results outputs/results/vid_calibration_v12 --model yolov8n --eps 0.008

[VID-v1.3]
python scripts/select_vid_defense.py --config configs/default.yaml --split-config configs/vid_split.yaml --split calibration --results outputs/results/vid_calibration_v13 --model yolov8n --eps 0.008
""",
            encoding="utf-8",
        )
    readme = f"""# Defense4UAVSwarm v1.3 Debug Bundle

1. Commit: `{commit}`.
2. v1.0/v1.1/v1.2 commands are summarized in `command_history.txt`.
3. Dataset root: `data/visdrone`.
4. Calibration/holdout split: `configs/vid_split.yaml`.
5. Holdout was not run because selection requires a safe non-noop candidate.
6. Main v1.2/v1.3 conclusion: global or unsafe filtering can reduce FP but risks tracking; current safe candidate is not enough for holdout.
7. Start with `rejection_truth_summary.csv`, `status_transition_summary.csv`, `oracle_new_fp_filter_summary.csv`, `summary_metrics_tracking_selected.csv`.
"""
    (results / "README_DEBUG.md").write_text(readme, encoding="utf-8")


def add_path(zf: zipfile.ZipFile, path: Path, seen: set[str]) -> None:
    if not path.exists():
        return
    if path.is_file():
        if path.suffix.lower() in ALLOWED_SUFFIXES:
            arcname = path.as_posix()
            if arcname not in seen:
                zf.write(path, arcname)
                seen.add(arcname)
        return
    for root, _, files in os.walk(path):
        for file in files:
            p = Path(root) / file
            if p.suffix.lower() in ALLOWED_SUFFIXES:
                arcname = p.as_posix()
                if arcname not in seen:
                    zf.write(p, arcname)
                    seen.add(arcname)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="outputs/bundles/Defense4UAVSwarm_v13_debug_bundle.zip")
    p.add_argument("--include-results", nargs="+", default=[])
    p.add_argument("--include-configs", nargs="+", default=["configs"])
    p.add_argument("--include-docs", nargs="+", default=["README.md", "docs/research_framework.md"])
    p.add_argument("--commands", default=None)
    args = p.parse_args()
    v13 = Path("outputs/results/vid_calibration_v13")
    write_run_files(v13, Path(args.commands) if args.commands else None)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        seen: set[str] = set()
        for path in [
            v13 / "README_DEBUG.md",
            v13 / "run_manifest.yaml",
            v13 / "metadata.json",
            v13 / "environment.txt",
            v13 / "git_info.txt",
            v13 / "command_history.txt",
        ]:
            add_path(zf, path, seen)
        for item in args.include_results + args.include_configs + args.include_docs:
            add_path(zf, Path(item), seen)
    print(f"{out} {out.stat().st_size / (1024 * 1024):.2f} MB")


if __name__ == "__main__":
    main()
