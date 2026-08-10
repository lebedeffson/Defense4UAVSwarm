from __future__ import annotations

import csv
import hashlib
import subprocess
import sys
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_matrix(path: Path, evidence: Path, status: str = "closed_by_analysis") -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "issue_id",
                "reviewer_issue",
                "status",
                "protocol_scope",
                "evidence_path",
                "evidence_sha256",
                "allowed_wording",
                "forbidden_wording",
                "blocking_reason",
                "article_location",
                "reviewer_response_location",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "issue_id": "X01",
                "reviewer_issue": "documentation audit",
                "status": status,
                "protocol_scope": "test",
                "evidence_path": evidence.as_posix(),
                "evidence_sha256": sha256(evidence),
                "allowed_wording": "ok",
                "forbidden_wording": "bad",
                "blocking_reason": "",
                "article_location": "test",
                "reviewer_response_location": "test",
            }
        )


def test_check_review_closure_evidence_passes(tmp_path: Path) -> None:
    evidence = tmp_path / "evidence.txt"
    evidence.write_text("checked", encoding="utf-8")
    matrix = tmp_path / "matrix.csv"
    write_matrix(matrix, evidence)
    result = subprocess.run(
        [sys.executable, "scripts/check_review_closure_evidence.py", "--matrix", matrix.as_posix()],
        check=False,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr + result.stdout


def test_check_review_closure_evidence_rejects_sha_mismatch(tmp_path: Path) -> None:
    evidence = tmp_path / "evidence.txt"
    evidence.write_text("checked", encoding="utf-8")
    matrix = tmp_path / "matrix.csv"
    write_matrix(matrix, evidence)
    evidence.write_text("changed", encoding="utf-8")
    result = subprocess.run(
        [sys.executable, "scripts/check_review_closure_evidence.py", "--matrix", matrix.as_posix()],
        check=False,
        text=True,
        capture_output=True,
    )
    assert result.returncode != 0
    assert "sha256 mismatch" in result.stdout
