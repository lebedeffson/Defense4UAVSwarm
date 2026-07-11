#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from scripts.q1_v55.common import read_config, results_root


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/q1_v55/bytetrack_risk_prioritized.yaml")
    p.add_argument("--output", default="outputs/bundles/Defense4UAVSwarm_q1_selective_v22_risk_prioritized_bundle.zip")
    args = p.parse_args()
    cfg = read_config(args.config)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    root_name = out.stem
    include = [
        Path("pyproject.toml"),
        Path("requirements.txt"),
        Path("configs/q1_v55"),
        Path("scripts/q1_v55"),
        Path("src/defense4uavswarm/__init__.py"),
        Path("src/defense4uavswarm/q1_v5"),
        Path("src/defense4uavswarm/q1_visdrone.py"),
        Path("src/defense4uavswarm/v8_sim.py"),
        Path("src/defense4uavswarm/datasets"),
        Path("tests/q1_v5"),
        results_root(cfg),
    ]
    manifest = {"protocol_id": cfg.get("protocol_id"), "git_commit": git(["rev-parse", "HEAD"]), "branch": git(["branch", "--show-current"]), "working_tree_dirty": bool(git(["status", "--short"]))}
    seen: set[str] = set()
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr(f"{root_name}/README.md", "# Defense4UAVSwarm q1 selective v2.2-RP bundle\n\nPrimary numbers: `outputs/results/q1_selective_quarantine_v22_risk_prioritized/article_ready/article_numbers_selective_v22.json`.\n")
        z.writestr(f"{root_name}/bundle_manifest.json", json.dumps(manifest, indent=2, sort_keys=True))
        for base in include:
            if not base.exists():
                continue
            files = [base] if base.is_file() else sorted(p for p in base.rglob("*") if p.is_file())
            for file in files:
                if skip(file):
                    continue
                arc = f"{root_name}/{file.as_posix()}"
                if arc in seen:
                    continue
                seen.add(arc)
                z.write(file, arc)
    with zipfile.ZipFile(out) as z:
        bad = z.testzip()
    print(f"status=ok bundle={out} testzip={bad}")


def skip(path: Path) -> bool:
    if "__pycache__" in path.parts or path.suffix in {".pyc", ".pyo", ".zip", ".pkl", ".pt", ".pth", ".onnx"}:
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
