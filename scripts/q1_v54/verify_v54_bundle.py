#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path


REQUIRED = [
    "README_Q1_V54_CORRECTED.md",
    "manifest.json",
    "corrected_operating_points.csv",
    "corrected_operating_claim_safe.md",
    "closure_checks_c01_c27.csv",
    "closure_checks_report.md",
    "evaluation_matching.py",
    "initiation_gate.py",
    "run_corrected_operating_curves.py",
    "verify_closure_checks.py",
]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--bundle", default="outputs/bundles/Defense4UAVSwarm_q1_v54_corrected_practice_bundle.zip")
    args = p.parse_args()
    path = Path(args.bundle)
    if not path.exists():
        raise SystemExit(f"Bundle not found: {path}")
    with zipfile.ZipFile(path) as z:
        bad = z.testzip()
        names = z.namelist()
        rows = []
        rows.append(("zip_integrity", bad is None, str(bad)))
        rows.append(("no_pyc", not any(n.endswith(".pyc") or "__pycache__" in n for n in names), "cache files absent"))
        for needle in REQUIRED:
            rows.append((f"contains:{needle}", any(needle in n for n in names), needle))
    failed = [r for r in rows if not r[1]]
    for name, ok, detail in rows:
        print(f"{name}: {'PASS' if ok else 'FAIL'} {detail}")
    if failed:
        sys.exit(1)
    print(f"status=PASS bundle={path} files={len(names)}")


if __name__ == "__main__":
    main()
