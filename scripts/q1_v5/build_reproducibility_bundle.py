#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import subprocess
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", default="reproducibility/q1_v5")
    p.add_argument("--bundle", default="outputs/bundles/Defense4UAVSwarm_q1_v5_methodology_bundle.zip")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if args.dry_run:
        print(f"dry_run=ok output={out}")
        return
    commit = git(["rev-parse", "HEAD"])
    manifest = {
        "git_commit": commit,
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": "VisDrone2019-VID-val",
        "trackers": ["ByteTrack-like", "OC-SORT frozen v9"],
        "detectors": ["YOLOv8s", "YOLOv8n"],
        "primary_metric": "false_new_tracks_per_100_frames",
        "noninferiority_margins": [0.005, 0.010, 0.015],
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (out / "README.md").write_text(readme(), encoding="utf-8")
    commands = """#!/usr/bin/env bash
set -euo pipefail
/home/lebedeffson/Code/venv/bin/python scripts/q1_v5/freeze_legacy_baseline.py --output-dir outputs/results/q1_v5/legacy_freeze
/home/lebedeffson/Code/venv/bin/python scripts/q1_v5/run_operating_curves.py --config configs/q1_v5/visdrone_bytetrack.yaml --output-dir outputs/results/q1_v5/operating_curves/bytetrack --overwrite
/home/lebedeffson/Code/venv/bin/python scripts/q1_v5/run_matched_point_analysis.py --operating-points outputs/results/q1_v5/operating_curves/bytetrack/operating_points_raw.csv --trust-method legacy_geometry_dynamic_no_multiagent --trust-parameter 0 --output-dir outputs/results/q1_v5/matched_points/bytetrack
/home/lebedeffson/Code/venv/bin/python scripts/q1_v5/run_statistical_analysis.py --by-sequence outputs/results/q1_v5/operating_curves/bytetrack/operating_points_by_sequence.csv --baseline-method tracker_baseline --method legacy_geometry_dynamic_no_multiagent --primary-metric false_new_tracks_per_100_frames --output-dir outputs/results/q1_v5/statistics/bytetrack
"""
    (out / "commands.sh").write_text(commands, encoding="utf-8")
    (out / "commands.sh").chmod(0o755)
    write_mapping(out / "table_figure_mapping.csv")
    write_checksums(out / "input_checksums.csv", [Path("outputs/results/q1_final_corrected/yolov8s_main/feature_audit.csv"), Path("outputs/results/q1_final_plus/tracker_comparison_yolov8s_v9/tracker_comparison_summary.csv")])
    write_checksums(out / "output_checksums.csv", [p for p in Path("outputs/results/q1_v5").rglob("*") if p.is_file()])
    build_zip(Path(args.bundle), out)
    print(f"status=ok bundle={args.bundle}")


def git(args: list[str]) -> str:
    try:
        return subprocess.check_output(["git", *args], text=True).strip()
    except Exception:
        return "unavailable"


def readme() -> str:
    return """# Defense4UAVSwarm q1_v5 reproducibility package

No new datasets are required. The package uses existing VisDrone annotations,
saved YOLO detections/features, and frozen v9 tracker-comparison summaries.

Run `bash reproducibility/q1_v5/commands.sh` from the repository root.

Important scope notes:
- VisDrone is single-camera UAV validation, not real multi-UAV validation.
- q1_v5 operating curves are candidate-space analyses from `feature_audit.csv`.
- frozen v9 tracker-comparison numbers are kept separate in `legacy_freeze`.
"""


def write_mapping(path: Path) -> None:
    rows = [
        ["Table A", "scripts/q1_v5/freeze_legacy_baseline.py", "outputs/results/q1_v5/legacy_freeze/legacy_baseline_freeze.csv", "n/a", "legacy tracker freeze"],
        ["Table B", "scripts/q1_v5/run_matched_point_analysis.py", "outputs/results/q1_v5/matched_points/bytetrack/matched_operating_points.csv", "configs/q1_v5/visdrone_bytetrack.yaml", "matched operating points"],
        ["Figure 1", "scripts/q1_v5/run_operating_curves.py", "outputs/results/q1_v5/operating_curves/bytetrack/operating_points_raw.csv", "configs/q1_v5/visdrone_bytetrack.yaml", "fig_f1_vs_false_new_bytetrack.png"],
        ["Stats", "scripts/q1_v5/run_statistical_analysis.py", "outputs/results/q1_v5/statistics/bytetrack/statistical_results.csv", "configs/q1_v5/visdrone_bytetrack.yaml", "statistical_summary.md"],
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["paper_item", "source_script", "source_csv", "config", "output_file"])
        w.writerows(rows)


def write_checksums(path: Path, files: list[Path]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["path", "sha256", "bytes"])
        for file in files:
            if file.exists() and file.is_file():
                w.writerow([file.as_posix(), sha256(file), file.stat().st_size])


def build_zip(bundle: Path, repro_dir: Path) -> None:
    bundle.parent.mkdir(parents=True, exist_ok=True)
    root = bundle.stem
    include = [repro_dir, Path("configs/q1_v5"), Path("scripts/q1_v5"), Path("src/defense4uavswarm/q1_v5"), Path("tests/q1_v5"), Path("outputs/results/q1_v5")]
    with zipfile.ZipFile(bundle, "w", zipfile.ZIP_DEFLATED) as z:
        for base in include:
            if not base.exists():
                continue
            for file in sorted(p for p in base.rglob("*") if p.is_file()):
                z.write(file, f"{root}/{file.as_posix()}")


if __name__ == "__main__":
    main()
