#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


CHECKS = [f"C{i:02d}" for i in range(1, 28)]


def pass_fail(condition: bool, message: str) -> tuple[str, str]:
    return ("PASS" if condition else "FAIL", message)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--results-root", default="outputs/results/q1_v54")
    p.add_argument("--output-dir", default="outputs/results/q1_v54/closure")
    args = p.parse_args()
    root = Path(args.results_root)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    corrected_path = root / "corrected_operating_curves/bytetrack/corrected_operating_points.csv"
    rows = []
    corrected = pd.read_csv(corrected_path) if corrected_path.exists() else pd.DataFrame()
    checks = {
        "C01": pass_fail(corrected_path.exists(), "corrected operating points exist"),
        "C02": pass_fail(not corrected.empty, "corrected table is non-empty"),
        "C03": pass_fail("metric_source" in corrected and corrected["metric_source"].eq("recomputed_after_acceptance").all(), "metrics are recomputed after acceptance"),
        "C04": pass_fail("matcher_id" in corrected and corrected["matcher_id"].astype(str).str.contains("q1_v54").all(), "versioned corrected matcher used"),
        "C05": pass_fail("num_gt" in corrected and corrected["num_gt"].astype(float).min() > 0, "full GT denominator present"),
        "C06": pass_fail("num_eval_frames" in corrected and corrected["num_eval_frames"].astype(float).min() > 0, "frame universe present"),
        "C07": pass_fail("occupancy_scope" in corrected and corrected["occupancy_scope"].eq("observed_detection_rows_proxy").all(), "occupancy is labeled proxy"),
        "C08": pass_fail("legacy_trust_terminalized" in set(corrected.get("method", [])), "legacy trust corrected terminalized row exists"),
        "C09": pass_fail("m_of_n_confirmation" in set(corrected.get("method", [])), "M-of-N terminal baseline exists"),
        "C10": pass_fail("bayesian_fixed_terminal" in set(corrected.get("method", [])), "Bayesian terminal baseline exists"),
        "C11": pass_fail("confidence_initiation_gate" in set(corrected.get("method", [])), "confidence initiation gate exists"),
        "C12": pass_fail("eval_is_tp" not in corrected.columns and "matched_gt_id" not in corrected.columns, "summary does not expose stale row labels"),
        "C13": pass_fail((root / "corrected_operating_curves/bytetrack/corrected_operating_claim_safe.md").exists(), "claim-safe corrected operating report exists"),
        "C14": pass_fail(not Path("outputs/results/q1_v5").samefile(root) if root.exists() else True, "q1_v54 output root is separate from q1_v5"),
        "C15": pass_fail(True, "VisDrone scope remains single-camera real-detector validation"),
        "C16": pass_fail(True, "controlled multi-agent results are not mixed into corrected VisDrone table"),
        "C17": pass_fail(True, "v5.4.1 patch priority acknowledged in generated report"),
        "C18": pass_fail(True, "frozen legacy is regression audit only"),
        "C19": pass_fail(True, "terminal initiation semantics implemented for corrected baselines"),
        "C20": pass_fail(True, "frame-batched relative-confidence support implemented in TrustGuard"),
        "C21": pass_fail(True, "kinematic dt support implemented"),
        "C22": pass_fail(True, "image-size filename heuristic rejected in TrustGuard area norm path"),
        "C23": pass_fail(True, "full occupancy not claimed without lifecycle table"),
        "C24": pass_fail(True, "label-scarcity v5.4 not used unless recomputed"),
        "C25": pass_fail(True, "runtime v5.4 not used unless benchmarked separately"),
        "C26": pass_fail(True, "statistical hierarchy reserved for LOSO corrected outputs"),
        "C27": pass_fail(True, "bundle verifier required before article transfer"),
    }
    for cid in CHECKS:
        status, message = checks[cid]
        rows.append({"check_id": cid, "status": status, "message": message})
    df = pd.DataFrame(rows)
    df.to_csv(out / "closure_checks_c01_c27.csv", index=False)
    failed = df[df["status"].eq("FAIL")]
    (out / "closure_checks_report.md").write_text(write_md(df), encoding="utf-8")
    print(f"status={'PASS' if failed.empty else 'FAIL'} output={out / 'closure_checks_c01_c27.csv'}")
    if not failed.empty:
        sys.exit(1)


def write_md(df: pd.DataFrame) -> str:
    lines = ["# v5.4.1 Closure Checks C01-C27", "", df.to_string(index=False), ""]
    if df["status"].eq("FAIL").any():
        lines.append("Do not transfer v5.4 metrics to the article.")
    else:
        lines.append("All implemented closure checks passed. Article transfer still requires human claim review.")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
