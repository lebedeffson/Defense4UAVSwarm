#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import subprocess
import zipfile
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="outputs/bundles/Defense4UAVSwarm_q1_selective_v21_article_final_bundle.zip")
    p.add_argument("--results-root", default="outputs/results/q1_selective_quarantine_v21")
    args = p.parse_args()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    root = out.stem
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
        Path(args.results_root),
    ]
    manifest = {"git_commit": git(["rev-parse", "HEAD"]), "branch": git(["branch", "--show-current"]), "protocol_id": "q1_selective_quarantine_v21_final", "working_tree_dirty": bool(git(["status", "--short"]))}
    seen: set[str] = set()
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr(f"{root}/README.md", "# Defense4UAVSwarm q1 selective v2.1 final article bundle\n\nPrimary numbers are in `outputs/results/q1_selective_quarantine_v21/article_ready/article_numbers_selective_v21.json`.\n")
        z.writestr(f"{root}/bundle_manifest.json", json.dumps(manifest, indent=2))
        for base in include:
            if not base.exists():
                continue
            files = [base] if base.is_file() else sorted(p for p in base.rglob("*") if p.is_file())
            for file in files:
                if skip(file):
                    continue
                arc = f"{root}/{file.as_posix()}"
                if arc in seen:
                    continue
                seen.add(arc)
                z.write(file, arc)
    print(f"status=ok bundle={out}")


def skip(path: Path) -> bool:
    if "__pycache__" in path.parts or path.suffix in {".pyc", ".pyo"}:
        return True
    if path.suffix.lower() in {".pkl", ".pt", ".pth", ".onnx", ".engine", ".zip"}:
        return True
    if path.name.endswith("_visdrone.json") and "detections" in path.parts:
        return True
    return False


def git(args: list[str]) -> str:
    try:
        return subprocess.check_output(["git", *args], text=True).strip()
    except Exception:
        return "unavailable"


if __name__ == "__main__":
    main()
