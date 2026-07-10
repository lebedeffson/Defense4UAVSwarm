#!/usr/bin/env python
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("bundle", nargs="?", default="outputs/bundles/Defense4UAVSwarm_q1_v542_corrected_partial_bundle.zip")
    p.add_argument("--output-dir", default="outputs/results/q1_v542/isolated_verify")
    args = p.parse_args()
    bundle = Path(args.bundle)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if not bundle.exists():
        raise SystemExit(f"Bundle not found: {bundle}")
    with zipfile.ZipFile(bundle) as z:
        bad = z.testzip()
        names = z.namelist()
        if bad is not None:
            raise SystemExit(f"Bad zip member: {bad}")
        if any("__pycache__" in n or n.endswith(".pyc") for n in names):
            raise SystemExit("Bundle contains bytecode/cache files")
        if any(Path(n).is_absolute() for n in names):
            raise SystemExit("Bundle contains absolute paths")
    with tempfile.TemporaryDirectory(prefix="q1_v542_bundle_") as tmp:
        tmp_path = Path(tmp)
        with zipfile.ZipFile(bundle) as z:
            z.extractall(tmp_path)
        roots = [p for p in tmp_path.iterdir() if p.is_dir()]
        if len(roots) != 1:
            raise SystemExit(f"Expected one bundle root, got {roots}")
        root = roots[0]
        venv = root / ".venv"
        run([sys.executable, "-m", "venv", str(venv)], cwd=root)
        py = venv / "bin/python"
        run([str(py), "-m", "pip", "install", "--upgrade", "pip"], cwd=root)
        run([str(py), "-m", "pip", "install", "-e", ".", "pytest", "pandas", "numpy", "scipy", "PyYAML", "opencv-python"], cwd=root)
        pytest = run([str(py), "-m", "pytest", "-q", "tests/q1_v5"], cwd=root, capture=True)
        (out / "pytest_report.txt").write_text(pytest, encoding="utf-8")
        smoke = run(
            [
                str(py),
                "scripts/q1_v54/run_corrected_operating_curves.py",
                "--config",
                "configs/q1_v54/bytetrack.yaml",
                "--output-dir",
                "outputs/results/q1_v542/smoke",
                "--dry-run",
            ],
            cwd=root,
            capture=True,
        )
        (out / "smoke_run.txt").write_text(smoke, encoding="utf-8")
    (out / "bundle_smoke_pass.txt").write_text(f"PASS {bundle}\n", encoding="utf-8")
    print(f"status=PASS bundle={bundle} output={out}")


def run(cmd: list[str], cwd: Path, capture: bool = False) -> str:
    result = subprocess.run(cmd, cwd=cwd, text=True, stdout=subprocess.PIPE if capture else None, stderr=subprocess.STDOUT if capture else None)
    if result.returncode != 0:
        if capture and result.stdout:
            print(result.stdout)
        raise SystemExit(f"Command failed ({result.returncode}): {' '.join(cmd)}")
    return result.stdout or ""


if __name__ == "__main__":
    main()
