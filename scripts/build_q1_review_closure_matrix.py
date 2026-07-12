#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path
from typing import Any


ALLOWED_STATUSES = {
    "closed_by_experiment",
    "closed_by_analysis",
    "closed_by_documentation",
    "blocked_missing_local_asset",
    "not_applicable",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def first_existing(paths: list[Path]) -> Path:
    for path in paths:
        if path.exists():
            return path
    return paths[0]


def evidence(path: Path) -> tuple[str, str]:
    return path.as_posix(), sha256(path) if path.exists() and path.is_file() else ""


def row(
    issue_id: str,
    reviewer_issue: str,
    status: str,
    protocol_scope: str,
    evidence_path: Path,
    allowed_wording: str,
    forbidden_wording: str,
    blocking_reason: str,
    article_location: str,
    reviewer_response_location: str,
) -> dict[str, Any]:
    if status not in ALLOWED_STATUSES:
        raise ValueError(f"Unsupported status for {issue_id}: {status}")
    path_text, digest = evidence(evidence_path)
    return {
        "issue_id": issue_id,
        "reviewer_issue": reviewer_issue,
        "status": status,
        "protocol_scope": protocol_scope,
        "evidence_path": path_text,
        "evidence_sha256": digest,
        "allowed_wording": allowed_wording,
        "forbidden_wording": forbidden_wording,
        "blocking_reason": blocking_reason,
        "article_location": article_location,
        "reviewer_response_location": reviewer_response_location,
    }


def build_rows(audit_dir: Path) -> list[dict[str, Any]]:
    blockers = audit_dir / "blockers"
    dataset_inventory = audit_dir / "dataset_inventory.csv"
    tracker_inventory = audit_dir / "tracker_inventory.csv"
    learned_inventory = audit_dir / "learned_baseline_inventory.csv"
    env_modules = audit_dir / "environment_modules.csv"
    primary_stats = first_existing(
        [
            Path("outputs/results/q1_selective_quarantine_v21/article_ready/table_selective_v21_statistics.csv"),
            Path("outputs/results/q1_v542/article_ready/table_statistics.csv"),
        ]
    )
    primary_runtime = first_existing(
        [
            Path("outputs/results/q1_selective_quarantine_v21/article_ready/table_selective_v21_runtime.csv"),
            Path("outputs/results/q1_v542/article_ready/table_runtime.csv"),
        ]
    )
    primary_params = first_existing(
        [
            Path("outputs/results/q1_selective_quarantine_v21/article_ready/article_numbers_selective_v21.json"),
            Path("outputs/results/q1_v542/article_ready/article_numbers.json"),
        ]
    )
    strongsort_blocker = first_existing(
        [
            blockers / "strongsort_blocker.json",
            Path("outputs/results/q1_final_plus/external_tracks/yolov8s/strongsort/strongsort_unavailable.md"),
        ]
    )
    rows = [
        row(
            "D01_SECOND_DATASET",
            "Second dataset or cross-domain validation is required for Q1 generalization.",
            "blocked_missing_local_asset",
            "Selective Trust Quarantine v2.1",
            blockers / "u2udata_blocker.json",
            "Confirmatory validation is on VisDrone only; other local adapters are not external validation.",
            "validated on U2UData; validated on V2U4Real; cross-domain generalization is proven",
            "U2UData local assets are not claim-eligible for unified temporal H1/H2; V2U4Real/OPV2V are missing locally.",
            "Limitations / Experimental protocol",
            "Reviewer response: dataset scope",
        ),
        row(
            "D02_VISDRONE_FAMILY",
            "VisDrone VID/MOT/DET must not be counted as multiple independent datasets.",
            "closed_by_analysis",
            "Local asset audit",
            dataset_inventory,
            "VisDrone VID is primary; MOT/DET are same-family or detection-only assets.",
            "validated on three independent VisDrone datasets",
            "",
            "Experimental protocol",
            "Reviewer response: dataset inventory",
        ),
        row(
            "T01_SORT",
            "SORT comparison is requested.",
            "blocked_missing_local_asset",
            "Unified v2.1 tracker comparison",
            blockers / "sort_blocker.json",
            "SORT is not reported because no current adapter or unified result exists.",
            "SORT was evaluated; SORT confirms generality",
            "No SORT adapter/result exists without counting OC-SORT or StrongSORT files.",
            "Limitations / Future work",
            "Reviewer response: tracker baselines",
        ),
        row(
            "T02_DEEPSORT",
            "DeepSORT comparison is requested.",
            "blocked_missing_local_asset",
            "Unified v2.1 tracker comparison",
            blockers / "deepsort_blocker.json",
            "DeepSORT is not reported because no ReID-capable local implementation/result exists.",
            "DeepSORT without ReID was evaluated as DeepSORT",
            "No current DeepSORT adapter or result exists; no no-ReID proxy is allowed.",
            "Limitations / Future work",
            "Reviewer response: tracker baselines",
        ),
        row(
            "T03_STRONGSORT",
            "StrongSORT comparison is requested.",
            "blocked_missing_local_asset",
            "Unified v2.1 tracker comparison",
            strongsort_blocker,
            "StrongSORT is blocked by missing ReID resources; no external weights were downloaded.",
            "StrongSORT was validated; StrongSORT supports the claim",
            "Only unavailable/blocker evidence exists, not unified numeric metrics.",
            "Limitations / Future work",
            "Reviewer response: tracker baselines",
        ),
        row(
            "RF01_LEARNED_FILTER",
            "Random Forest or learned filter should be compared using calibration labels.",
            "blocked_missing_local_asset",
            "Unified v2.1 learned baseline",
            first_existing([blockers / "rf_terminal_gate_v54_blocker.json", learned_inventory]),
            "Legacy RF exists but is not part of v2.1 confirmatory tables until rerun without leakage.",
            "RF was beaten in the v2.1 protocol; RF proves superiority",
            "Only legacy or non-unified RF evidence is present in the local audit.",
            "Limitations / Baselines",
            "Reviewer response: learned filters",
        ),
        row(
            "S01_H1_H2",
            "F1 preservation must gate contamination superiority.",
            "closed_by_experiment",
            "Selective Trust Quarantine v2.1",
            primary_stats,
            "H1 F1 non-inferiority is tested before H2 contamination reduction.",
            "H2 is claimed when H1 fails; F1 improvement is the main claim",
            "",
            "Results / Statistics",
            "Reviewer response: hypotheses",
        ),
        row(
            "P01_PARAMETERS",
            "Parameter grids and selected values must be visible.",
            "closed_by_analysis",
            "Selective Trust Quarantine v2.1",
            primary_params,
            "Parameters are reported from frozen config/article-number outputs.",
            "parameters were manually chosen after test results",
            "",
            "Experimental protocol / parameter table",
            "Reviewer response: parameters",
        ),
        row(
            "R01_RUNTIME_ENVIRONMENT",
            "Runtime must state hardware/software scope and exclude detector/tracker time.",
            "closed_by_experiment",
            "Selective Trust Quarantine v2.1",
            primary_runtime,
            "Runtime is gate-only and environment-scoped.",
            "5 ms proves UAV deployment readiness",
            "",
            "Runtime section",
            "Reviewer response: runtime",
        ),
        row(
            "A01_AUDIT_ENVIRONMENT",
            "Local audit must record environment module availability.",
            "closed_by_analysis",
            "Q1 practical closure v7 audit",
            env_modules,
            "Environment availability is audited locally without downloading resources.",
            "missing packages were installed during audit",
            "",
            "Reproducibility",
            "Reviewer response: reproducibility",
        ),
    ]
    return rows


def write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Q1 Review Closure Matrix",
        "",
        "| issue_id | status | evidence | allowed wording |",
        "|---|---|---|---|",
    ]
    for item in rows:
        lines.append(
            f"| {item['issue_id']} | {item['status']} | `{item['evidence_path']}` | {item['allowed_wording']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--audit-dir", default="outputs/q1_practical_closure_v7/audit")
    p.add_argument("--output", default="outputs/q1_practical_closure_v7/review_closure_matrix.csv")
    args = p.parse_args()
    audit_dir = Path(args.audit_dir)
    output = Path(args.output)
    rows = build_rows(audit_dir)
    write_csv(output, rows)
    write_markdown(output.with_suffix(".md"), rows)
    print(f"status=ok output={output} rows={len(rows)}")


if __name__ == "__main__":
    main()
