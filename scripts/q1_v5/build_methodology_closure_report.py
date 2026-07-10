#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def read_optional(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--q1-v5-root", default="outputs/results/q1_v5")
    p.add_argument("--output-dir", default="outputs/results/q1_v5/methodology_closure")
    args = p.parse_args()
    root = Path(args.q1_v5_root)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    legacy = read_optional(root / "legacy_freeze/legacy_baseline_freeze.csv")
    dominance = read_optional(root / "baseline_dominance/bytetrack/noninferiority_constrained_best.csv")
    staleness = read_optional(root / "calibration_staleness/calibration_staleness_summary.csv")
    trustguard = read_optional(root / "trust_guard_ablation/bytetrack/trust_guard_ablation_summary.csv")
    shift = read_optional(root / "confidence_shift/bytetrack/confidence_shift_summary.csv")
    rows = []
    if not legacy.empty:
        rows.append({"item": "frozen_tracker_baseline", "status": "done", "evidence": "legacy_baseline_freeze.csv"})
    if not dominance.empty:
        rows.append({"item": "operating_curves_matched_points", "status": "done", "evidence": "baseline_dominance/noninferiority_constrained_best.csv"})
    if not staleness.empty:
        rows.append({"item": "calibration_staleness", "status": "done", "evidence": "calibration_staleness_summary.csv"})
    if not trustguard.empty:
        rows.append({"item": "strict_vs_balanced_trustguard", "status": "done", "evidence": "trust_guard_ablation_summary.csv"})
    if not shift.empty:
        rows.append({"item": "confidence_shift", "status": "done", "evidence": "confidence_shift_summary.csv"})
    table = pd.DataFrame(rows)
    table.to_csv(out / "methodology_closure_checklist.csv", index=False)
    (out / "methodology_closure_report.md").write_text(write_md(table, legacy, dominance, staleness, trustguard, shift), encoding="utf-8")
    print(f"status=ok output={out / 'methodology_closure_report.md'}")


def write_md(table: pd.DataFrame, legacy: pd.DataFrame, dominance: pd.DataFrame, staleness: pd.DataFrame, trustguard: pd.DataFrame, shift: pd.DataFrame) -> str:
    lines = ["# Q1 v5 Methodology Closure Report", ""]
    lines.append("Scope: no new datasets; analysis uses saved VisDrone detections/features and frozen tracker summaries.")
    if not table.empty:
        lines += ["", "## Checklist", table.to_string(index=False)]
    if not legacy.empty:
        lines += ["", "## Frozen tracker result", legacy[[c for c in ["tracker", "trust_mode", "F1", "false_new_tracks", "F1_delta_vs_tracker", "false_new_delta_vs_tracker"] if c in legacy]].to_string(index=False)]
    if not dominance.empty:
        ok = dominance[dominance["status"].eq("ok")] if "status" in dominance else dominance
        lines += ["", "## Best constrained operating points", ok[[c for c in ["delta_margin", "method", "F1", "false_new_tracks", "false_track_occupancy_frames"] if c in ok]].head(20).to_string(index=False)]
    if not staleness.empty:
        lines += ["", "## Calibration staleness summary", staleness.head(40).to_string(index=False)]
    if not trustguard.empty:
        lines += ["", "## TrustGuard architecture ablation", trustguard[[c for c in ["method", "F1", "false_new_tracks", "false_track_occupancy_frames", "median_confirmation_delay"] if c in trustguard]].to_string(index=False)]
    if not shift.empty:
        agg = shift.groupby("method").agg(worst_F1=("F1", "min"), worst_occupancy=("false_track_occupancy_frames", "max")).reset_index()
        lines += ["", "## Confidence shift worst cases", agg.to_string(index=False)]
    lines += [
        "",
        "Claim-safe conclusion:",
        "- Primary defensible endpoint is false-track/map-contamination reduction under an explicit F1 non-inferiority margin.",
        "- VisDrone remains single-camera validation, not real multi-UAV validation.",
        "- Bayesian/RF baselines must stay in the paper even when they dominate specific operating regions.",
        "- TrustGuard v5.2 is currently a diagnostic architecture: its default point is too conservative for a main performance claim.",
    ]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
