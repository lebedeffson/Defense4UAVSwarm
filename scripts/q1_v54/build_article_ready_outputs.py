#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from pathlib import Path

import pandas as pd


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--results-root", default="outputs/results/q1_v542")
    p.add_argument("--output-dir", default="outputs/results/q1_v542/article_ready")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    root = Path(args.results_root)
    out = Path(args.output_dir)
    if out.exists() and any(out.iterdir()) and not args.overwrite and not args.dry_run:
        raise SystemExit(f"Output exists; use --overwrite: {out}")
    out.mkdir(parents=True, exist_ok=True)
    if args.dry_run:
        print(f"dry_run=ok output={out}")
        return
    bytetrack = pd.read_csv(root / "curves/bytetrack/corrected_operating_points.csv")
    ocsort = pd.read_csv(root / "curves/ocsort/corrected_operating_points.csv")
    loso = pd.read_csv(root / "loso/outer_test_summary.csv")
    stats = pd.read_csv(root / "statistics/statistical_results.csv")
    matched = pd.read_csv(root / "matched_points/matched_point_audit.csv")
    bytetrack.to_csv(out / "table_main_bytetrack.csv", index=False)
    ocsort.to_csv(out / "table_main_ocsort.csv", index=False)
    loso.to_csv(out / "table_outer_loso.csv", index=False)
    stats.to_csv(out / "table_statistics.csv", index=False)
    matched.to_csv(out / "table_matched_points.csv", index=False)
    pd.DataFrame([{"status": "excluded_not_recomputed", "reason": "v5.4.2a uses corrected LOSO as main protocol; old label-scarcity numbers are not article-ready"}]).to_csv(out / "table_label_scarcity.csv", index=False)
    pd.DataFrame([{"status": "excluded_not_recomputed", "reason": "runtime not recomputed under v5.4.2a online/offline separation"}]).to_csv(out / "table_runtime.csv", index=False)
    primary = stats[(stats["method"].eq("legacy_trust_terminalized")) & (stats["hypothesis"].eq("H1_F1_noninferiority"))]
    h1_pass_any = bool(primary["status"].eq("PASS").any())
    article = {
        "protocol_id": "q1_v542_article_grade_final",
        "git_commit": git(["rev-parse", "HEAD"]),
        "branch": git(["branch", "--show-current"]),
        "matcher_id": "q1_v542_max_cardinality_iou_v1",
        "primary_method": "legacy_trust_terminalized",
        "bytetrack": clean_json(summarize_tracker(bytetrack)),
        "ocsort": clean_json(summarize_tracker(ocsort)),
        "outer_loso": clean_json(loso.to_dict(orient="records")),
        "statistics": clean_json(stats.to_dict(orient="records")),
        "trustguard_decision": "excluded",
        "label_scarcity_status": "excluded_not_recomputed",
        "runtime_status": "excluded_not_recomputed",
        "article_ready": bool(h1_pass_any),
    }
    schema = {
        "type": "object",
        "required": ["protocol_id", "git_commit", "matcher_id", "primary_method", "article_ready"],
    }
    (out / "article_numbers.json").write_text(json.dumps(article, indent=2, allow_nan=False), encoding="utf-8")
    (out / "article_numbers.schema.json").write_text(json.dumps(schema, indent=2), encoding="utf-8")
    (out / "claims_supported.md").write_text(write_supported(stats), encoding="utf-8")
    (out / "claims_not_supported.md").write_text(write_not_supported(stats), encoding="utf-8")
    (out / "limitations.md").write_text(write_limitations(), encoding="utf-8")
    (out / "article_patch.md").write_text(write_patch(out / "article_numbers.json"), encoding="utf-8")
    (out / "run_metadata.json").write_text(json.dumps({"status": "success", "git_commit": git(["rev-parse", "HEAD"]), "article_ready": bool(h1_pass_any)}, indent=2), encoding="utf-8")
    print(f"status=ok output={out} article_ready={h1_pass_any} article_numbers_sha256={sha256(out/'article_numbers.json')}")


def summarize_tracker(df: pd.DataFrame) -> dict:
    rows = {}
    for method in ["tracker_baseline", "legacy_trust_terminalized", "m_of_n_confirmation", "bayesian_fixed_terminal"]:
        hit = df[df["method"].eq(method)]
        if not hit.empty:
            rows[method] = hit.iloc[0].to_dict()
    return rows


def clean_json(value):
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [clean_json(v) for v in value]
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if pd.isna(value) if not isinstance(value, (dict, list, tuple)) else False:
        return None
    return value


def write_supported(stats: pd.DataFrame) -> str:
    lines = ["# Supported Claims", ""]
    lines.append("- Corrected evaluator recomputes matching after each acceptance mask.")
    lines.append("- ByteTrack and OC-SORT are evaluated with the same corrected matcher.")
    lines.append("- Contamination reductions can be reported as descriptive/exploratory when H1 fails.")
    h1 = stats[(stats["hypothesis"].eq("H1_F1_noninferiority")) & (stats["status"].eq("PASS"))]
    if not h1.empty:
        lines.append("- At least one method passes the predefined F1 non-inferiority test.")
    return "\n".join(lines) + "\n"


def write_not_supported(stats: pd.DataFrame) -> str:
    h1_fail = stats[(stats["hypothesis"].eq("H1_F1_noninferiority")) & (stats["status"].ne("PASS"))]
    lines = ["# Claims Not Supported", ""]
    if not h1_fail.empty:
        lines.append("- Confirmatory H2 contamination-superiority claims are not supported for methods that fail H1.")
    lines.append("- Old label-scarcity and runtime tables are not article-ready under v5.4.2a.")
    lines.append("- TrustGuard is excluded from main results in this closure pass.")
    return "\n".join(lines) + "\n"


def write_limitations() -> str:
    return """# Limitations

- VisDrone remains single-camera real-detector validation, not real multi-UAV validation.
- Occupancy is observed detection-row proxy because lifecycle prediction rows are unavailable.
- Label-scarcity and runtime sections are excluded from main results unless recomputed under the corrected protocol.
"""


def write_patch(article_numbers: Path) -> str:
    return f"""# Article Patch Instructions

Use only `{article_numbers}` as the numeric source.

Do not transfer v5.4.2a numbers into confirmatory claims unless `article_ready=true`
and closure reports `ARTICLE_READY_PASS`.

If H1 fails for a method, describe contamination reductions as exploratory or
descriptive, not confirmatory superiority.
"""


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git(args: list[str]) -> str:
    try:
        return subprocess.check_output(["git", *args], text=True).strip()
    except Exception:
        return "unavailable"


if __name__ == "__main__":
    main()
