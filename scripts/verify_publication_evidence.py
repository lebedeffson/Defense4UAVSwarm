#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any


EXPECTED_SHA256 = "0224da22e4b52bd6ab401c2cb301a5f83b9df80bb8814772d39f32d411a12b6a"
DEFAULT_BUNDLE = Path("outputs/bundles/Defense4UAVSwarm_q1_practical_closure_v7_evidence_bundle.zip")
ARCHIVE_ROOT = PurePosixPath("Defense4UAVSwarm_q1_practical_closure_v7")
MANIFEST_MEMBER = str(ARCHIVE_ROOT / "manifest.json")
KEY_MEMBERS = {
    str(ARCHIVE_ROOT / "outputs/q1_practical_closure_v7/sort/fold_results.csv"),
    str(ARCHIVE_ROOT / "outputs/q1_practical_closure_v7/rf_v21/fold_results.csv"),
    str(ARCHIVE_ROOT / "outputs/q1_practical_closure_v7/rf_v21/leakage_audit.json"),
    str(ARCHIVE_ROOT / "outputs/q1_practical_closure_v7/unified_comparison/comparison_by_tracker.csv"),
    str(ARCHIVE_ROOT / "outputs/q1_practical_closure_v7/lifecycle/false_episode_lifecycle.csv"),
    str(ARCHIVE_ROOT / "outputs/q1_practical_closure_v7/bound_audit/prefix_bound_audit.csv"),
    str(ARCHIVE_ROOT / "outputs/q1_practical_closure_v7/environment/runtime.csv"),
}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_bundle(path: Path, expected_sha256: str = EXPECTED_SHA256) -> dict[str, Any]:
    report: dict[str, Any] = {
        "bundle": str(path),
        "expected_sha256": expected_sha256,
        "checks": [],
    }

    def record(name: str, passed: bool, detail: str = "") -> None:
        report["checks"].append({"check": name, "pass": bool(passed), "detail": detail})

    if not path.is_file():
        record("bundle_exists", False, "original archive is not available at this path")
        report["status"] = "FAIL"
        return report

    actual_sha = sha256_file(path)
    report["actual_sha256"] = actual_sha
    record("sha256_matches", actual_sha == expected_sha256, actual_sha)

    try:
        with zipfile.ZipFile(path) as archive:
            corrupt = archive.testzip()
            record("zip_integrity", corrupt is None, corrupt or "all members readable")
            names = set(archive.namelist())
            record("manifest_present", MANIFEST_MEMBER in names, MANIFEST_MEMBER)
            missing_members = sorted(KEY_MEMBERS - names)
            record("key_artifacts_present", not missing_members, ", ".join(missing_members))
            if MANIFEST_MEMBER in names:
                manifest = json.loads(archive.read(MANIFEST_MEMBER))
                checks = manifest.get("checks", [])
                report["manifest_check_count"] = len(checks)
                report["manifest_failed_checks"] = [
                    item.get("check", "unnamed") for item in checks if item.get("pass") is not True
                ]
                record("manifest_status_success", manifest.get("status") == "success")
                record("manifest_checks_50", len(checks) == 50, str(len(checks)))
                record("manifest_checks_pass", all(item.get("pass") is True for item in checks))
                hash_errors = verify_manifest_file_hashes(archive, manifest)
                record("manifest_file_hashes", not hash_errors, ", ".join(hash_errors))
    except (OSError, zipfile.BadZipFile, json.JSONDecodeError) as exc:
        record("archive_parse", False, str(exc))

    report["status"] = "PASS" if all(item["pass"] for item in report["checks"]) else "FAIL"
    return report


def verify_manifest_file_hashes(archive: zipfile.ZipFile, manifest: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    names = set(archive.namelist())
    for item in manifest.get("files", []):
        relative = PurePosixPath(str(item.get("path", "")))
        member = str(ARCHIVE_ROOT / relative)
        if member not in names:
            failures.append(f"missing:{relative}")
            continue
        actual = sha256_bytes(archive.read(member))
        if actual != item.get("sha256"):
            failures.append(f"sha256:{relative}")
    return failures


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify the frozen Q1 publication evidence without recomputation.")
    parser.add_argument("bundle", nargs="?", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--expected-sha256", default=EXPECTED_SHA256)
    args = parser.parse_args()
    report = verify_bundle(args.bundle, args.expected_sha256)
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["status"] != "PASS":
        sys.exit(1)


if __name__ == "__main__":
    main()
