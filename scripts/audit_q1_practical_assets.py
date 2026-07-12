#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = "outputs/q1_practical_closure_v7/audit"


@dataclass(frozen=True)
class AssetProbe:
    name: str
    paths: tuple[str, ...]
    adapter_paths: tuple[str, ...] = ()
    config_paths: tuple[str, ...] = ()
    result_globs: tuple[str, ...] = ()
    notes: str = ""


DATASETS = [
    AssetProbe(
        "VisDrone2019-VID-val",
        ("data/visdrone/VisDrone2019-VID-val",),
        adapter_paths=("src/defense4uavswarm/datasets/visdrone.py",),
        config_paths=("configs/q1_v54/bytetrack.yaml", "configs/q1_v54/ocsort.yaml"),
        result_globs=("outputs/results/q1_selective_quarantine_v21/**", "outputs/results/q1_v542/**"),
        notes="Primary local single-camera UAV tracking validation dataset.",
    ),
    AssetProbe(
        "VisDrone2019-MOT-val",
        ("data/visdrone/VisDrone2019-MOT-val",),
        adapter_paths=("src/defense4uavswarm/datasets/visdrone.py",),
        config_paths=("configs/visdrone_split.yaml",),
        result_globs=("outputs/results/**/*mot*.csv", "outputs/results/**/*MOT*.csv"),
        notes="Same VisDrone family; not an independent cross-domain dataset for Q1 claims.",
    ),
    AssetProbe(
        "VisDrone2019-DET-val",
        ("data/visdrone/VisDrone2019-DET-val",),
        adapter_paths=("src/defense4uavswarm/datasets/visdrone.py",),
        result_globs=("outputs/results/det_*/**",),
        notes="Detection-only branch; not sufficient for temporal H1/H2 tracking claims.",
    ),
    AssetProbe(
        "U2UData",
        ("data/U2UData", "data/manifests/u2u_v7", "data/manifests/u2u_v7_min300"),
        adapter_paths=("src/defense4uavswarm/datasets/u2u_adapter.py",),
        config_paths=("configs/u2u_dataset_config.yaml", "docs/DATASET_U2U.md", "docs/REPRODUCE_U2U.md"),
        result_globs=("outputs/results/v7_u2u/**",),
        notes="Local files may support structure/manifest validation; require temporal IDs and GT/detections for H1/H2.",
    ),
    AssetProbe(
        "V2U4Real",
        ("data/V2U4Real", "data/v2u4real"),
        adapter_paths=("src/defense4uavswarm/datasets/v2u4real_adapter.py",),
        result_globs=("outputs/results/**/*v2u4real*", "outputs/results/**/*V2U4Real*"),
        notes="Adapter exists, but local data/results must be present before any validation claim.",
    ),
    AssetProbe(
        "OPV2V",
        ("data/OPV2V", "data/opv2v"),
        adapter_paths=("src/defense4uavswarm/datasets/opv2v_adapter.py",),
        result_globs=("outputs/results/**/*opv2v*", "outputs/results/**/*OPV2V*"),
        notes="Different cooperative driving/BEV domain; do not use as direct 2D UAV tracking evidence without a matched protocol.",
    ),
    AssetProbe(
        "custom_uav_swarm_v8",
        ("data/custom_uav_swarm_v8",),
        result_globs=("outputs/results/v8_custom_swarm/**",),
        notes="Synthetic/custom local dataset; useful for diagnostics, not independent real-world validation.",
    ),
]

TRACKERS = [
    AssetProbe(
        "ByteTrack-like",
        (),
        adapter_paths=("src/defense4uavswarm/trackers/bytetrack_adapter.py",),
        result_globs=("outputs/results/q1_selective_quarantine_v21/**/*bytetrack*", "outputs/results/q1_v542/**/*bytetrack*"),
        notes="Primary confirmed tracker baseline in v2.1.",
    ),
    AssetProbe(
        "OC-SORT",
        (),
        adapter_paths=("src/defense4uavswarm/trackers/ocsort_adapter.py", "scripts/run_external_ocsort.py"),
        result_globs=("outputs/results/q1_selective_quarantine_v21/**/*ocsort*", "outputs/results/q1_v542/**/*ocsort*"),
        notes="Confirmed but H2 is not claim-positive in v2.1.",
    ),
    AssetProbe(
        "SORT",
        (),
        adapter_paths=("src/defense4uavswarm/trackers/sort_adapter.py", "scripts/run_external_sort.py"),
        result_globs=("outputs/results/**/sort/*.csv", "outputs/results/**/sort/*.json", "outputs/results/**/sort/*.md"),
        notes="Required by reviewer closure if implemented; OC-SORT and StrongSORT evidence must not be counted as SORT.",
    ),
    AssetProbe(
        "DeepSORT",
        (),
        adapter_paths=("src/defense4uavswarm/trackers/deepsort_adapter.py", "scripts/run_external_deepsort.py"),
        result_globs=("outputs/results/**/*deepsort*", "outputs/results/**/*DeepSORT*"),
        notes="Requires real ReID-capable implementation or explicit blocker; no no-ReID proxy claim.",
    ),
    AssetProbe(
        "StrongSORT",
        (),
        adapter_paths=("src/defense4uavswarm/trackers/strongsort_adapter.py", "scripts/run_external_strongsort.py"),
        result_globs=("outputs/results/**/*strongsort*", "outputs/results/**/*StrongSORT*"),
        notes="Requires BoxMOT and ReID resources; blocker is valid if weights are absent and no download is allowed.",
    ),
]

LEARNED_BASELINES = [
    AssetProbe(
        "rf_learned_gate_legacy",
        (),
        adapter_paths=("src/defense4uavswarm/q1_visdrone.py", "scripts/run_q1_real_detector_trust_experiment.py"),
        config_paths=("configs/rf_baseline_config.yaml",),
        result_globs=("outputs/results/**/*rf*.csv", "outputs/results/**/*rf*.json", "outputs/results/**/*RF*.csv"),
        notes="Legacy RF support exists; must be rerun in unified v2.1 protocol before article comparison.",
    ),
    AssetProbe(
        "rf_terminal_gate_v54",
        (),
        adapter_paths=("src/defense4uavswarm/q1_v5/initiation_gate.py",),
        result_globs=("outputs/results/q1_v542/**/*rf*", "outputs/results/q1_selective_quarantine_v21/**/*rf*"),
        notes="Gate API exists; leakage-safe unbounded/budgeted RF tables still need evidence.",
    ),
    AssetProbe(
        "risk_prioritized_ranker_v22",
        (),
        adapter_paths=("src/defense4uavswarm/q1_v5/risk_ranker.py", "scripts/q1_v55/train_risk_rankers_oof.py"),
        result_globs=("outputs/results/q1_selective_quarantine_v22_risk_prioritized/**",),
        notes="Gated negative package; do not use as positive article result.",
    ),
]


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except Exception:
        return path.as_posix()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git(args: list[str]) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return ""


def existing_paths(paths: tuple[str, ...]) -> list[Path]:
    return [REPO_ROOT / p for p in paths if (REPO_ROOT / p).exists()]


def glob_existing(patterns: tuple[str, ...], limit: int = 50) -> list[Path]:
    hits: list[Path] = []
    for pattern in patterns:
        pattern_path = Path(pattern)
        if pattern_path.is_absolute():
            if any(ch in pattern for ch in "*?[]"):
                hits.extend(p for p in pattern_path.parent.glob(pattern_path.name) if p.is_file())
            elif pattern_path.is_file():
                hits.append(pattern_path)
        else:
            hits.extend(p for p in REPO_ROOT.glob(pattern) if p.is_file())
        if len(hits) >= limit:
            break
    return sorted(set(hits))[:limit]


def count_files(path: Path, limit: int = 200_000) -> int:
    if path.is_file():
        return 1
    if not path.exists():
        return 0
    count = 0
    for _ in path.rglob("*"):
        count += 1
        if count >= limit:
            return count
    return count


def summarize_path(path: Path) -> dict[str, Any]:
    return {
        "path": rel(path),
        "exists": path.exists(),
        "is_dir": path.is_dir(),
        "file_count": count_files(path) if path.exists() else 0,
    }


def status_for_dataset(name: str, roots: list[Path], result_hits: list[Path]) -> str:
    if name == "VisDrone2019-VID-val":
        return "verified_primary_local" if roots and result_hits else "present_needs_result_check"
    if name in {"VisDrone2019-MOT-val", "VisDrone2019-DET-val"}:
        return "same_family_or_detection_only_not_independent"
    if not roots:
        return "blocked_missing_local_asset"
    if result_hits:
        return "present_with_legacy_results_needs_unified_h1h2_audit"
    return "present_needs_temporal_gt_detection_audit"


def claim_scope_for_dataset(name: str, status: str) -> str:
    if name == "VisDrone2019-VID-val" and status == "verified_primary_local":
        return "claim_eligible_primary_single_dataset"
    if name.startswith("VisDrone2019-"):
        return "not_independent_cross_domain_evidence"
    if status.startswith("blocked"):
        return "blocker_only"
    return "not_claim_eligible_until_unified_temporal_h1h2_results_exist"


def probe_assets(kind: str, probes: list[AssetProbe]) -> list[dict[str, Any]]:
    rows = []
    for probe in probes:
        roots = existing_paths(probe.paths)
        adapters = existing_paths(probe.adapter_paths)
        configs = existing_paths(probe.config_paths)
        results = glob_existing(probe.result_globs)
        if kind == "dataset":
            status = status_for_dataset(probe.name, roots, results)
            claim_scope = claim_scope_for_dataset(probe.name, status)
        else:
            blocker_only_results = results and all(
                any(token in p.name.lower() for token in ["unavailable", "blocker", "run_summary"])
                for p in results
            )
            if adapters and blocker_only_results:
                status = "implemented_with_blocker_evidence"
            elif adapters and results:
                status = "implemented_with_result_evidence"
            elif adapters:
                status = "implemented_needs_unified_result_or_blocker"
            elif results:
                status = "results_found_without_current_adapter"
            else:
                status = "blocked_missing_local_implementation"
            claim_scope = (
                "blocker_only"
                if status.startswith("blocked") or status == "implemented_with_blocker_evidence"
                else "not_claim_eligible_until_unified_v21_evidence"
            )
        rows.append(
            {
                "kind": kind,
                "name": probe.name,
                "status": status,
                "claim_scope": claim_scope,
                "paths_checked": ";".join(probe.paths),
                "existing_paths": ";".join(rel(p) for p in roots),
                "adapter_paths": ";".join(rel(p) for p in adapters),
                "config_or_doc_paths": ";".join(rel(p) for p in configs),
                "result_evidence_count": len(results),
                "result_evidence_sample": ";".join(rel(p) for p in results[:10]),
                "notes": probe.notes,
            }
        )
    return rows


def module_available(module: str) -> tuple[bool, str]:
    try:
        spec = importlib.util.find_spec(module)
    except Exception as exc:
        return False, str(exc)
    return spec is not None, "" if spec is not None else "module_not_found"


def environment_rows() -> list[dict[str, Any]]:
    packages = ["numpy", "pandas", "scipy", "sklearn", "torch", "boxmot", "ultralytics"]
    rows = []
    for pkg in packages:
        available, reason = module_available(pkg)
        rows.append({"package": pkg, "available": available, "reason": reason})
    return rows


def local_dataset_adapter_reports() -> list[dict[str, Any]]:
    sys.path.insert(0, str(REPO_ROOT / "src"))
    rows: list[dict[str, Any]] = []
    try:
        from defense4uavswarm.datasets.opv2v_adapter import OPV2VAdapter
        from defense4uavswarm.datasets.u2u_adapter import U2UDataAdapter
        from defense4uavswarm.datasets.v2u4real_adapter import V2U4RealAdapter
    except Exception as exc:
        return [{"dataset": "adapter_import", "status": "blocked_import_failed", "reason": str(exc)}]
    for cls, root in [
        (U2UDataAdapter, REPO_ROOT / "data/U2UData"),
        (V2U4RealAdapter, REPO_ROOT / "data/V2U4Real"),
        (OPV2VAdapter, REPO_ROOT / "data/OPV2V"),
    ]:
        try:
            report = cls(root).report()
        except Exception as exc:
            report = {"dataset": cls.dataset_name, "dataset_root": rel(root), "status": "blocked_report_failed", "reason": str(exc)}
        rows.append(flatten_dict(report))
    return rows


def flatten_dict(data: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in data.items():
        if isinstance(value, (list, tuple, set)):
            out[key] = ";".join(str(v) for v in value)
        elif isinstance(value, dict):
            out[key] = json.dumps(value, sort_keys=True)
        else:
            out[key] = value
    return out


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")


def write_blockers(out: Path, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    blockers = []
    for row in rows:
        status = str(row.get("status", ""))
        claim_scope = str(row.get("claim_scope", ""))
        if status.startswith("blocked") or "not_independent" in str(row.get("claim_scope", "")) or "needs_" in status:
            blocker = {
                "asset": row["name"],
                "kind": row["kind"],
                "status": status,
                "checked_paths": row.get("paths_checked", ""),
                "existing_paths": row.get("existing_paths", ""),
                "blocking_reason": row.get("notes", ""),
                "claim_policy": row.get("claim_scope", "blocker_only"),
            }
            blockers.append(blocker)
            safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in row["name"].lower())
            write_json(out / "blockers" / f"{safe}_blocker.json", blocker)
        elif claim_scope == "blocker_only":
            blocker = {
                "asset": row["name"],
                "kind": row["kind"],
                "status": status,
                "checked_paths": row.get("paths_checked", ""),
                "existing_paths": row.get("existing_paths", ""),
                "blocking_reason": row.get("notes", ""),
                "claim_policy": claim_scope,
            }
            blockers.append(blocker)
            safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in row["name"].lower())
            write_json(out / "blockers" / f"{safe}_blocker.json", blocker)
    return blockers


def collect_hash_rows(paths: list[Path]) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(set(paths)):
        if path.exists() and path.is_file():
            rows.append({"path": rel(path), "sha256": sha256(path), "bytes": path.stat().st_size})
    return rows


def write_summary(out: Path, payload: dict[str, Any]) -> None:
    datasets = payload["datasets"]
    trackers = payload["trackers"]
    learned = payload["learned_baselines"]
    blockers = payload["blockers"]
    lines = [
        "# Q1 Practical Closure v7 Local Asset Audit",
        "",
        f"- git_commit: `{payload['git']['commit']}`",
        f"- git_branch: `{payload['git']['branch']}`",
        f"- dirty: `{payload['git']['dirty']}`",
        f"- datasets audited: {len(datasets)}",
        f"- trackers audited: {len(trackers)}",
        f"- learned baselines audited: {len(learned)}",
        f"- blockers written: {len(blockers)}",
        "",
        "## Claim Boundary",
        "",
        "- VisDrone2019-VID-val is the only primary local dataset with current article-grade evidence.",
        "- VisDrone MOT/DET are not independent cross-domain evidence for Q1 generalization claims.",
        "- U2UData/V2U4Real/OPV2V require local temporal GT/detection evidence before any validation wording.",
        "- SORT/DeepSORT/StrongSORT require unified v2.1 results or explicit blockers before article tables.",
        "- Legacy v18/q1_final_plus and v2.2-RP results must not be merged into Selective Trust Quarantine v2.1 confirmatory tables.",
        "",
        "## Next Required Work",
        "",
        "1. Build evidence-driven `review_closure_matrix.csv` from these audited paths.",
        "2. Re-run or block RF/SORT/DeepSORT/StrongSORT under the unified v2.1 evaluator.",
        "3. Create `results_lock.json` only after claim-eligible results and blockers are complete.",
        "4. Generate article tables and figures only from the lock.",
    ]
    (out / "audit_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser(description="Inventory local Q1 practical-closure assets without downloads.")
    p.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    args = p.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    dataset_rows = probe_assets("dataset", DATASETS)
    tracker_rows = probe_assets("tracker", TRACKERS)
    learned_rows = probe_assets("learned_baseline", LEARNED_BASELINES)
    adapter_reports = local_dataset_adapter_reports()
    env_rows = environment_rows()
    blockers = write_blockers(out, dataset_rows + tracker_rows + learned_rows)

    evidence_paths: list[Path] = []
    for row in dataset_rows + tracker_rows + learned_rows:
        for key in ["adapter_paths", "config_or_doc_paths", "result_evidence_sample"]:
            for item in str(row.get(key, "")).split(";"):
                if item:
                    evidence_paths.append(REPO_ROOT / item)
    hash_rows = collect_hash_rows(evidence_paths)

    payload = {
        "git": {
            "commit": git(["rev-parse", "HEAD"]),
            "branch": git(["branch", "--show-current"]),
            "dirty": bool(git(["status", "--short"])),
        },
        "datasets": dataset_rows,
        "dataset_adapter_reports": adapter_reports,
        "trackers": tracker_rows,
        "learned_baselines": learned_rows,
        "environment_modules": env_rows,
        "blockers": blockers,
        "hashes": hash_rows,
    }
    write_csv(out / "dataset_inventory.csv", dataset_rows)
    write_csv(out / "dataset_adapter_reports.csv", adapter_reports)
    write_csv(out / "tracker_inventory.csv", tracker_rows)
    write_csv(out / "learned_baseline_inventory.csv", learned_rows)
    write_csv(out / "environment_modules.csv", env_rows)
    write_csv(out / "evidence_hashes.csv", hash_rows)
    write_json(out / "asset_inventory.json", payload)
    write_summary(out, payload)
    print(f"status=ok output={out / 'asset_inventory.json'} blockers={len(blockers)}")


if __name__ == "__main__":
    main()
