#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import hashlib
import sys
from pathlib import Path


ALLOWED_STATUSES = {
    "closed_by_experiment",
    "closed_by_analysis",
    "closed_by_documentation",
    "blocked_missing_local_asset",
    "not_applicable",
}
NUMERICAL_HINTS = ("dataset", "sort", "deepsort", "strongsort", "random forest", "rf", "f1", "runtime", "comparison")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def has_checked_paths(path: Path) -> bool:
    text = path.read_text(encoding="utf-8", errors="ignore")
    return "checked_paths" in text and "blocking_reason" in text


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--matrix", required=True)
    args = p.parse_args()
    matrix = Path(args.matrix)
    rows = read_rows(matrix)
    errors: list[str] = []
    for row in rows:
        issue_id = row.get("issue_id", "")
        status = row.get("status", "")
        evidence_path = Path(row.get("evidence_path", ""))
        expected_sha = row.get("evidence_sha256", "")
        reviewer_issue = row.get("reviewer_issue", "").lower()
        if status not in ALLOWED_STATUSES:
            errors.append(f"{issue_id}: unsupported status {status}")
        if not evidence_path.exists():
            errors.append(f"{issue_id}: missing evidence path {evidence_path}")
            continue
        if evidence_path.is_file() and expected_sha != sha256(evidence_path):
            errors.append(f"{issue_id}: sha256 mismatch for {evidence_path}")
        if status == "closed_by_documentation" and any(token in reviewer_issue for token in NUMERICAL_HINTS):
            errors.append(f"{issue_id}: numerical issue cannot be closed by documentation")
        if status == "closed_by_experiment" and evidence_path.suffix.lower() != ".csv":
            errors.append(f"{issue_id}: closed_by_experiment requires result CSV evidence")
        if status == "blocked_missing_local_asset" and not has_checked_paths(evidence_path):
            errors.append(f"{issue_id}: blocker evidence must include checked_paths and blocking_reason")
    if errors:
        for error in errors:
            print(f"ERROR {error}")
        sys.exit(1)
    print(f"status=PASS matrix={matrix} rows={len(rows)}")


if __name__ == "__main__":
    main()
