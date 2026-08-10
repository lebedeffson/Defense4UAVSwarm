#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import subprocess
import zipfile
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="outputs/bundles/Defense4UAVSwarm_q1_selective_v2_article_slim_bundle.zip")
    p.add_argument("--results-root", default="outputs/results/q1_selective_quarantine_v2")
    args = p.parse_args()

    bundle = Path(args.output)
    bundle.parent.mkdir(parents=True, exist_ok=True)
    root = bundle.stem
    include = [
        Path("pyproject.toml"),
        Path("requirements.txt"),
        Path("configs/q1_v54"),
        Path("scripts/q1_v54"),
        Path("src/defense4uavswarm/__init__.py"),
        Path("src/defense4uavswarm/q1_v5"),
        Path("src/defense4uavswarm/q1_visdrone.py"),
        Path("src/defense4uavswarm/v8_sim.py"),
        Path("src/defense4uavswarm/datasets/__init__.py"),
        Path("src/defense4uavswarm/datasets/visdrone.py"),
        Path("tests/q1_v5"),
        Path("data_manifests/visdrone_eval_frames.csv"),
        Path(args.results_root),
    ]

    with zipfile.ZipFile(bundle, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr(f"{root}/README.md", readme(args.results_root))
        z.writestr(f"{root}/manifest.json", json.dumps(manifest(args.results_root), indent=2))
        seen: set[str] = set()
        for base in include:
            if not base.exists():
                continue
            files = [base] if base.is_file() else sorted(p for p in base.rglob("*") if p.is_file())
            for file in files:
                if "__pycache__" in file.parts or file.suffix in {".pyc", ".pyo"}:
                    continue
                if should_skip(file):
                    continue
                arc = f"{root}/{file.as_posix()}"
                if arc in seen:
                    continue
                seen.add(arc)
                z.write(file, arc)
    print(f"status=ok bundle={bundle}")


def should_skip(path: Path) -> bool:
    heavy_suffixes = {".pkl", ".pt", ".pth", ".onnx", ".engine"}
    if path.suffix.lower() in heavy_suffixes:
        return True
    if path.name.endswith("_visdrone.json") and "detections" in path.parts:
        return True
    return False


def readme(results_root: str) -> str:
    return f"""# Defense4UAVSwarm selective quarantine v2 article bundle

This slim bundle contains the final article-ready selective quarantine v2 slice:
- corrected evaluator modules and tests;
- q1_v54 scripts and configs;
- regenerated selective v2 curves, LOSO, statistics, budget audit, and article-ready tables;
- no raw images, detector JSON dumps, external tracker repositories, or model weights.

Primary numbers must come from:
- `{results_root}/article_ready/article_numbers_selective_v2.json`
- `{results_root}/article_ready/article_patch_selective_v2.md`

The primary F1 is backfilled-map F1. Online current-frame metrics are reported separately.
Legacy RF, label-scarcity, runtime, and v542 article-number tables are not part of this selective v2 claim unless explicitly marked legacy protocol.
"""


def manifest(results_root: str) -> dict[str, str]:
    return {
        "git_commit": git(["rev-parse", "HEAD"]),
        "branch": git(["branch", "--show-current"]),
        "protocol_id": "q1_selective_quarantine_v2_corrected",
        "results_root": results_root,
    }


def git(args: list[str]) -> str:
    try:
        return subprocess.check_output(["git", *args], text=True).strip()
    except Exception:
        return "unavailable"


if __name__ == "__main__":
    main()
