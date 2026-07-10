#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path


REQUIRED = [
    "README.md",
    "manifest.json",
    "visdrone_eval_frames.csv",
    "corrected_operating_points.csv",
    "closure_checks_c01_c45.csv",
    "closure_checks_report.md",
    "evaluation_matching.py",
    "initiation_gate.py",
    "frame_manifest.py",
    "run_corrected_operating_curves.py",
    "verify_article_grade_closure.py",
]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--bundle", default="outputs/bundles/Defense4UAVSwarm_q1_v542_corrected_partial_bundle.zip")
    args = p.parse_args()
    path = Path(args.bundle)
    if not path.exists():
        raise SystemExit(f"Bundle not found: {path}")
    with zipfile.ZipFile(path) as z:
        bad = z.testzip()
        names = z.namelist()
    rows = [("zip_integrity", bad is None, str(bad)), ("no_cache", not any("__pycache__" in n or n.endswith(".pyc") for n in names), "no pyc/cache")]
    for needle in REQUIRED:
        rows.append((f"contains:{needle}", any(needle in n for n in names), needle))
    for name, ok, detail in rows:
        print(f"{name}: {'PASS' if ok else 'FAIL'} {detail}")
    failed = [r for r in rows if not r[1]]
    if failed:
        sys.exit(1)
    print(f"status=PASS bundle={path} files={len(names)}")


if __name__ == "__main__":
    main()
