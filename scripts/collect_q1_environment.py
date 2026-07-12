#!/usr/bin/env python
from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
THREAD_ENV_KEYS = [
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
]


def main() -> None:
    out = Path("outputs/q1_practical_closure_v7/environment")
    out.mkdir(parents=True, exist_ok=True)
    payload = {
        "cpu_model": cpu_model(),
        "physical_cores": physical_cores(),
        "logical_cores": os.cpu_count(),
        "ram": ram(),
        "os": platform.platform(),
        "python": sys.version,
        "packages": {
            "numpy": package_version("numpy"),
            "scipy": package_version("scipy"),
            "pandas": package_version("pandas"),
            "scikit-learn": package_version("sklearn"),
        },
        "git_commit": git(["rev-parse", "HEAD"]),
        "git_branch": git(["branch", "--show-current"]),
        "working_tree_status": git(["status", "--short"]),
        "thread_environment_variables": {key: os.environ.get(key) for key in THREAD_ENV_KEYS},
    }
    (out / "environment.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(f"status=ok output={out / 'environment.json'}")


def package_version(module: str) -> str:
    try:
        mod = __import__(module)
        return str(getattr(mod, "__version__", "unknown"))
    except Exception as exc:
        return f"unavailable: {type(exc).__name__}"


def cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return platform.processor() or platform.machine()


def physical_cores() -> int | str:
    try:
        pairs = set()
        physical_id = None
        core_id = None
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("physical id"):
                physical_id = line.split(":", 1)[1].strip()
            elif line.startswith("core id"):
                core_id = line.split(":", 1)[1].strip()
            elif not line.strip():
                if physical_id is not None and core_id is not None:
                    pairs.add((physical_id, core_id))
                physical_id = None
                core_id = None
        if pairs:
            return len(pairs)
    except Exception:
        pass
    logical = os.cpu_count()
    return max(1, logical // 2) if logical else "unavailable"


def ram() -> str:
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("MemTotal:"):
                return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return "unavailable"


def git(args: list[str]) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True).strip()
    except Exception:
        return "unavailable"


if __name__ == "__main__":
    main()
