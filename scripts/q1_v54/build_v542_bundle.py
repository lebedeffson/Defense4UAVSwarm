#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import subprocess
import zipfile
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="outputs/bundles/Defense4UAVSwarm_q1_v542_corrected_partial_bundle.zip")
    p.add_argument("--results-root", default="outputs/results/q1_v542")
    args = p.parse_args()
    bundle = Path(args.output)
    bundle.parent.mkdir(parents=True, exist_ok=True)
    root = bundle.stem
    include = [
        Path("pyproject.toml"),
        Path("requirements.txt"),
        Path("configs/q1_v54"),
        Path("scripts/q1_v54"),
        Path("src/defense4uavswarm/q1_v5"),
        Path("src/defense4uavswarm/q1_visdrone.py"),
        Path("src/defense4uavswarm/v8_sim.py"),
        Path("src/defense4uavswarm/datasets/visdrone.py"),
        Path("tests/q1_v5"),
        Path("data_manifests/visdrone_eval_frames.csv"),
        Path(args.results_root),
    ]
    with zipfile.ZipFile(bundle, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr(f"{root}/README.md", readme())
        z.writestr(f"{root}/manifest.json", json.dumps(manifest(args.results_root), indent=2))
        seen: set[str] = set()
        for base in include:
            if not base.exists():
                continue
            files = [base] if base.is_file() else sorted(p for p in base.rglob("*") if p.is_file())
            for file in files:
                if "__pycache__" in file.parts or file.suffix in {".pyc", ".pyo"}:
                    continue
                arc = f"{root}/{file.as_posix()}"
                if arc in seen:
                    continue
                seen.add(arc)
                z.write(file, arc)
    print(f"status=ok bundle={bundle}")


def readme() -> str:
    return """# Defense4UAVSwarm q1_v542 corrected partial bundle

This bundle contains the corrected v5.4.2 evaluator slice:
- physical VisDrone frame manifest;
- max-cardinality IoU matcher after acceptance masks;
- ignore-region suppression only for unmatched detections;
- terminal initiation gates;
- corrected ByteTrack and OC-SORT operating curves;
- C01-C45 closure report.

The closure report is authoritative. If it says DO_NOT_TRANSFER_TO_ARTICLE,
these results are not article-grade final numbers.
"""


def manifest(results_root: str) -> dict[str, str]:
    return {
        "git_commit": git(["rev-parse", "HEAD"]),
        "branch": git(["branch", "--show-current"]),
        "protocol_id": "q1_v542_corrected_partial",
        "matcher_id": "q1_v542_max_cardinality_iou_v1",
        "results_root": results_root,
    }


def git(args: list[str]) -> str:
    try:
        return subprocess.check_output(["git", *args], text=True).strip()
    except Exception:
        return "unavailable"


if __name__ == "__main__":
    main()
