#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import zipfile
from pathlib import Path
from typing import Any

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
ROOT = Path("outputs/q1_practical_closure_v7")
DEFAULT_OUTPUT = Path("outputs/bundles/Defense4UAVSwarm_q1_practical_closure_v7_evidence_bundle.zip")


REQUIRED_FILES = [
    "sort/fold_results.csv",
    "sort/aggregate_results.csv",
    "sort/selected_parameters.csv",
    "sort/terminal_state_audit.csv",
    "sort/statistics.json",
    "sort/runtime.csv",
    "sort/run_manifest.json",
    "rf_v21/fold_results.csv",
    "rf_v21/aggregate_results.csv",
    "rf_v21/selected_hyperparameters.csv",
    "rf_v21/selected_budget_parameters.csv",
    "rf_v21/selected_thresholds.csv",
    "rf_v21/feature_schema.json",
    "rf_v21/feature_importance.csv",
    "rf_v21/leakage_audit.json",
    "rf_v21/statistics.json",
    "rf_v21/run_manifest.json",
    "unified_comparison/comparison_by_tracker.csv",
    "unified_comparison/comparison_bytetrack.csv",
    "unified_comparison/comparison_statistics.json",
    "unified_comparison/table_article_ru.csv",
    "unified_comparison/table_article_en.csv",
    "lifecycle/false_episode_lifecycle.csv",
    "lifecycle/false_episode_lifecycle_summary.csv",
    "lifecycle/false_episode_duration_histogram_ru.png",
    "lifecycle/false_episode_survival_curve_ru.png",
    "lifecycle/false_episode_rows_distribution_ru.png",
    "lifecycle/false_episode_duration_vs_rows_ru.png",
    "lifecycle/run_manifest.json",
    "bound_audit/prefix_bound_audit.csv",
    "bound_audit/sequence_bound_summary.csv",
    "bound_audit/bound_violations.csv",
    "bound_audit/bound_tightness_statistics.json",
    "bound_audit/fig_bound_actual_vs_limit_ru.png",
    "environment/environment.json",
    "environment/runtime.csv",
    "environment/runtime_manifest.json",
]


SOURCE_FILES = [
    "src/defense4uavswarm/trackers/sort_adapter.py",
    "src/defense4uavswarm/q1_v5/rf_v21.py",
    "scripts/run_external_sort.py",
    "scripts/q1_v54/run_sort_v21_experiment.py",
    "scripts/q1_v54/run_rf_v21_experiment.py",
    "scripts/q1_v54/build_unified_comparison_v7.py",
    "scripts/q1_v54/build_false_track_lifecycle_v7.py",
    "scripts/q1_v54/build_prefix_bound_audit_v7.py",
    "scripts/q1_v54/benchmark_practical_runtime_v7.py",
    "scripts/collect_q1_environment.py",
]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output", default=str(DEFAULT_OUTPUT))
    p.add_argument("--root", default=str(ROOT))
    args = p.parse_args()
    root = Path(args.root)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    checks = run_checks(root)
    if not all(item["pass"] for item in checks):
        failed = [item for item in checks if not item["pass"]]
        raise SystemExit(json.dumps({"status": "failed", "failed_checks": failed}, indent=2))
    files = [root / rel for rel in REQUIRED_FILES]
    files += sorted((root / "rf_v21" / "predictions").glob("*.csv"))
    files += [Path(rel) for rel in SOURCE_FILES if Path(rel).exists()]
    manifest = bundle_manifest(root, checks, files)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr("Defense4UAVSwarm_q1_practical_closure_v7/manifest.json", json.dumps(manifest, indent=2, sort_keys=True))
        z.writestr("Defense4UAVSwarm_q1_practical_closure_v7/README.md", readme(manifest))
        for path in files:
            arc = "Defense4UAVSwarm_q1_practical_closure_v7/" + str(path)
            z.write(path, arc)
    print(f"status=ok output={output} files={len(files) + 2} sha256={sha256(output)}")


def run_checks(root: Path) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    for rel in REQUIRED_FILES:
        checks.append(check((root / rel).exists(), f"required file exists: {rel}"))

    sort_stats = read_json(root / "sort/statistics.json")
    checks.append(check(sort_stats.get("sort_baseline_complete") is True, "SORT baseline folds complete: 7/7"))
    checks.append(check(sort_stats.get("sort_selective_v21_complete") is True, "SORT selective v2.1 folds complete: 7/7"))
    terminal = pd.read_csv(root / "sort/terminal_state_audit.csv")
    checks.append(check(int(terminal["pending_end_count"].fillna(0).sum()) == 0, "terminal pending states: 0"))

    rf_stats = read_json(root / "rf_v21/statistics.json")
    checks.append(check(rf_stats.get("rf_unbounded_complete") is True, "RF unbounded folds complete: 7/7"))
    checks.append(check(rf_stats.get("rf_budgeted_complete") is True, "RF budgeted folds complete: 7/7"))
    leak = read_json(root / "rf_v21/leakage_audit.json")
    fold_leak = any(bool(row.get("future_features_detected")) or bool(row.get("test_leakage_detected")) for row in leak.get("folds", []))
    checks.append(check(leak.get("future_features_detected") is False and not fold_leak, "future feature leakage: 0"))
    checks.append(check(leak.get("test_leakage_detected") is False and not fold_leak, "test-fold leakage: 0"))

    bound = read_json(root / "bound_audit/bound_tightness_statistics.json")
    checks.append(check(int(bound.get("budget_balance_violations", -1)) == 0, "budget equality violations: 0"))
    checks.append(check(int(bound.get("episode_bound_violations", -1)) == 0, "episode D-bound violations: 0"))
    checks.append(check(int(bound.get("budget_bound_violations", -1)) == 0, "budget D-bound violations: 0"))

    runtime = pd.read_csv(root / "environment/runtime.csv")
    checks.append(check(int(runtime["repetitions"].min()) >= 30, "runtime repetitions: >=30"))
    unified = read_json(root / "unified_comparison/comparison_statistics.json")
    checks.append(check(unified.get("bytetrack_complete_methods") is True, "article tables generated: yes"))
    figures = [
        root / "lifecycle/false_episode_duration_histogram_ru.png",
        root / "lifecycle/false_episode_survival_curve_ru.png",
        root / "lifecycle/false_episode_rows_distribution_ru.png",
        root / "lifecycle/false_episode_duration_vs_rows_ru.png",
        root / "bound_audit/fig_bound_actual_vs_limit_ru.png",
    ]
    checks.append(check(all(path.exists() and path.stat().st_size > 0 for path in figures), "article figures generated: yes"))
    return checks


def bundle_manifest(root: Path, checks: list[dict[str, Any]], files: list[Path]) -> dict[str, Any]:
    return {
        "status": "success",
        "bundle_id": "Defense4UAVSwarm_q1_practical_closure_v7",
        "git_commit": git(["rev-parse", "HEAD"]),
        "git_branch": git(["branch", "--show-current"]),
        "working_tree_status": git(["status", "--short"]),
        "root": str(root),
        "checks": checks,
        "files": [{"path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size} for path in files],
    }


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def check(ok: bool, label: str) -> dict[str, Any]:
    return {"check": label, "pass": bool(ok)}


def readme(manifest: dict[str, Any]) -> str:
    lines = [
        "# Defense4UAVSwarm Q1 Practical Closure v7 Evidence Bundle",
        "",
        f"Git commit: `{manifest['git_commit']}`",
        f"Git branch: `{manifest['git_branch']}`",
        "",
        "This archive contains generated evidence artifacts for SORT, RF v2.1, unified comparison, false-track lifecycle, D-bound audit, environment, and runtime.",
        "",
        "Checks:",
    ]
    for item in manifest["checks"]:
        status = "PASS" if item["pass"] else "FAIL"
        lines.append(f"- {status}: {item['check']}")
    return "\n".join(lines) + "\n"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git(args: list[str]) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True).strip()
    except Exception:
        return "unavailable"


if __name__ == "__main__":
    main()
