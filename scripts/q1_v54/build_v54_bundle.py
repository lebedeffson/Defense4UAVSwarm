#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import subprocess
import zipfile
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="outputs/bundles/Defense4UAVSwarm_q1_v54_corrected_practice_bundle.zip")
    p.add_argument("--results-root", default="outputs/results/q1_v54")
    args = p.parse_args()
    bundle = Path(args.output)
    bundle.parent.mkdir(parents=True, exist_ok=True)
    root = bundle.stem
    include = [
        Path("configs/q1_v54"),
        Path("scripts/q1_v54"),
        Path("src/defense4uavswarm/q1_v5"),
        Path("tests/q1_v5"),
        Path(args.results_root),
    ]
    with zipfile.ZipFile(bundle, "w", zipfile.ZIP_DEFLATED) as z:
        add_text(z, f"{root}/README_Q1_V54_CORRECTED.md", readme())
        add_text(z, f"{root}/manifest.json", json.dumps(manifest(), indent=2))
        seen: set[str] = set()
        for base in include:
            if not base.exists():
                continue
            for file in sorted(p for p in base.rglob("*") if p.is_file()):
                if "__pycache__" in file.parts or file.suffix in {".pyc", ".pyo"}:
                    continue
                arc = f"{root}/{file.as_posix()}"
                if arc in seen:
                    continue
                seen.add(arc)
                z.write(file, arc)
    print(f"status=ok bundle={bundle}")


def readme() -> str:
    return """# Defense4UAVSwarm q1_v54 corrected practice bundle

This bundle contains the v5.4.1 corrected evaluation path.

Critical scope notes:
- Metrics in `outputs/results/q1_v54` are recomputed after each acceptance mask.
- Frozen q1_v5/v9 results are regression audits only and must not be mixed with corrected metrics.
- VisDrone remains single-camera real-detector validation, not real multi-UAV validation.
- Occupancy is `observed_false_track_occupancy_frames` unless a full tracker lifecycle table is available.
"""


def manifest() -> dict[str, str]:
    return {
        "git_commit": git(["rev-parse", "HEAD"]),
        "branch": git(["branch", "--show-current"]),
        "metric_source": "recomputed_after_acceptance",
        "output_root": "outputs/results/q1_v54",
    }


def add_text(z: zipfile.ZipFile, name: str, text: str) -> None:
    z.writestr(name, text)


def git(args: list[str]) -> str:
    try:
        return subprocess.check_output(["git", *args], text=True).strip()
    except Exception:
        return "unavailable"


if __name__ == "__main__":
    main()
