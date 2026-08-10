#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
CLOSURE_COMMIT = "06d2870b248665d7483ce25bbd3b8f61538deef8"
CLOSURE_TAG = "q1-practical-closure-v7"

REQUIRED_FILES = [
    "README.md",
    "CITATION.cff",
    "LICENSE",
    "configs/q1_v54/bytetrack.yaml",
    "configs/q1_v54/ocsort.yaml",
    "configs/q1_v54/sort.yaml",
    "src/defense4uavswarm/q1_v5/selective_quarantine.py",
    "src/defense4uavswarm/q1_v5/rf_v21.py",
    "src/defense4uavswarm/trackers/bytetrack_adapter.py",
    "src/defense4uavswarm/trackers/ocsort_adapter.py",
    "src/defense4uavswarm/trackers/sort_adapter.py",
    "scripts/q1_v54/run_sort_v21_experiment.py",
    "scripts/q1_v54/run_rf_v21_experiment.py",
    "scripts/q1_v54/build_unified_comparison_v7.py",
    "scripts/q1_v54/build_false_track_lifecycle_v7.py",
    "scripts/q1_v54/build_prefix_bound_audit_v7.py",
    "scripts/q1_v54/benchmark_practical_runtime_v7.py",
]

README_REQUIRED = [
    "Interpretable control of track initiation in UAV video tracking.",
    "14,967",
    "392",
    "46,368 to 45,943",
    "10,031 to 9,855",
    "29,385",
    "0 violations",
    "124112200072-2",
    "Reduction not statistically confirmed",
]

AUTHORS = [
    "Yuri V. Trofimov",
    "Alexey N. Averkin",
    "Alexey V. Shevchenko",
    "Egor M. Kuznetsov",
    "Alexander D. Lebedev",
]

PUBLIC_DOCUMENTATION = ["README.md", "CITATION.cff"]
LOCAL_HOME_PREFIX = "/" + "home/"


def run_command(args: list[str], cwd: Path) -> tuple[bool, str]:
    result = subprocess.run(args, cwd=cwd, text=True, capture_output=True, check=False)
    output = (result.stdout + result.stderr).strip()
    return result.returncode == 0, output


def verify_repository(root: Path = REPO_ROOT, run_tests: bool = False) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    def record(name: str, passed: bool, detail: str = "") -> None:
        checks.append({"check": name, "pass": bool(passed), "detail": detail})

    missing = [relative for relative in REQUIRED_FILES if not (root / relative).is_file()]
    record("required_tracked_files", not missing, ", ".join(missing))

    readme = (root / "README.md").read_text(encoding="utf-8") if (root / "README.md").is_file() else ""
    citation = (root / "CITATION.cff").read_text(encoding="utf-8") if (root / "CITATION.cff").is_file() else ""
    missing_readme = [value for value in README_REQUIRED if value not in readme]
    record("scientific_status_and_metadata", not missing_readme, ", ".join(missing_readme))
    record("mermaid_diagrams", readme.count("```mermaid") == 3, str(readme.count("```mermaid")))
    record("article_authors_readme", all(author in readme for author in AUTHORS))
    record(
        "article_authors_citation",
        all(part in citation for part in ["Trofimov", "Averkin", "Shevchenko", "Kuznetsov", "Lebedev"]),
    )
    record("mit_license", "license: MIT" in citation and (root / "LICENSE").is_file())

    public_text = "\n".join(
        (root / relative).read_text(encoding="utf-8")
        for relative in PUBLIC_DOCUMENTATION
        if (root / relative).is_file()
    )
    record("public_surface_hygiene", LOCAL_HOME_PREFIX not in public_text)
    verifier_text = (root / "scripts/verify_publication_state.py").read_text(encoding="utf-8")
    record("verifier_has_no_local_path", LOCAL_HOME_PREFIX not in verifier_text)

    tag_ok, tag_output = run_command(["git", "rev-parse", f"{CLOSURE_TAG}^{{}}"], root)
    record("closure_tag", tag_ok and tag_output.splitlines()[0] == CLOSURE_COMMIT, tag_output)
    ancestor_ok, ancestor_output = run_command(
        ["git", "merge-base", "--is-ancestor", CLOSURE_COMMIT, "HEAD"], root
    )
    record("closure_reachable", ancestor_ok, ancestor_output)

    if run_tests:
        pytest_ok, pytest_output = run_command([sys.executable, "-m", "pytest", "-q"], root)
        record("pytest", pytest_ok, pytest_output)
        compile_ok, compile_output = run_command(
            [sys.executable, "-m", "compileall", "-q", "-f", "src", "scripts", "tests"], root
        )
        record("compileall", compile_ok, compile_output)
        diff_ok, diff_output = run_command(["git", "diff", "--check"], root)
        record("git_diff_check", diff_ok, diff_output)

    return {
        "status": "PASS" if all(item["pass"] for item in checks) else "FAIL",
        "repository_root": str(root),
        "checks": checks,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify the tracked publication state without recomputation.")
    parser.add_argument("--repository", type=Path, default=REPO_ROOT)
    parser.add_argument("--run-tests", action="store_true")
    args = parser.parse_args()
    report = verify_repository(args.repository.resolve(), run_tests=args.run_tests)
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["status"] != "PASS":
        sys.exit(1)


if __name__ == "__main__":
    main()
