#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

import pandas as pd

from defense4uavswarm.q1_v5.pareto import dominance_matrix, pareto_front


def closest(group: pd.DataFrame, column: str, target: float) -> pd.Series:
    idx = (group[column].astype(float) - float(target)).abs().sort_values().index[0]
    return group.loc[idx]


def curve_best_at_f1(points: pd.DataFrame, min_f1: float) -> pd.DataFrame:
    rows = []
    for method, group in points.groupby("method", sort=True):
        feasible = group[group["F1"].astype(float) >= float(min_f1)]
        if feasible.empty:
            rows.append({"method": method, "status": "infeasible", "min_f1": min_f1})
            continue
        best = feasible.sort_values(["false_new_tracks", "false_track_occupancy_frames", "F1"], ascending=[True, True, False]).iloc[0].to_dict()
        best.update({"status": "ok", "min_f1": min_f1})
        rows.append(best)
    return pd.DataFrame(rows)


def matched_against(points: pd.DataFrame, reference_method: str) -> pd.DataFrame:
    ref = points[points["method"].eq(reference_method)].copy()
    if ref.empty:
        return pd.DataFrame()
    ref_point = ref.sort_values(["false_new_tracks", "F1"], ascending=[True, False]).iloc[0]
    rows: list[dict[str, Any]] = []
    for method, group in points.groupby("method", sort=True):
        if method == reference_method:
            continue
        same_false = closest(group, "false_new_tracks", float(ref_point["false_new_tracks"])).to_dict()
        same_false.update(
            {
                "reference_method": reference_method,
                "reference_F1": float(ref_point["F1"]),
                "reference_false_new_tracks": float(ref_point["false_new_tracks"]),
                "comparison_type": "matched_false_new",
                "delta_F1_vs_reference": float(same_false["F1"]) - float(ref_point["F1"]),
            }
        )
        rows.append(same_false)
        same_f1 = closest(group, "F1", float(ref_point["F1"])).to_dict()
        same_f1.update(
            {
                "reference_method": reference_method,
                "reference_F1": float(ref_point["F1"]),
                "reference_false_new_tracks": float(ref_point["false_new_tracks"]),
                "comparison_type": "matched_F1",
                "delta_false_new_vs_reference": float(same_f1["false_new_tracks"]) - float(ref_point["false_new_tracks"]),
            }
        )
        rows.append(same_f1)
    return pd.DataFrame(rows)


def write_claim(out: Path, points: pd.DataFrame, dominance: pd.DataFrame, constrained: pd.DataFrame, matched: pd.DataFrame) -> None:
    lines = ["# Baseline Dominance Audit", ""]
    trust = points[points["method"].str.contains("trust|legacy_geometry", regex=True)]
    bayes = points[points["method"].eq("bayesian_fixed")]
    if not trust.empty and not bayes.empty:
        bayes_dom = dominance[(dominance["method_a"].eq("bayesian_fixed")) & (dominance["method_b"].isin(trust["method"].unique())) & dominance["a_dominates_b"].astype(bool)]
        trust_dom = dominance[(dominance["method_b"].eq("bayesian_fixed")) & (dominance["method_a"].isin(trust["method"].unique())) & dominance["a_dominates_b"].astype(bool)]
        lines.append(f"Bayesian fixed dominates at least one trust curve: {'yes' if len(bayes_dom) else 'no'}")
        lines.append(f"Any trust curve dominates Bayesian fixed: {'yes' if len(trust_dom) else 'no'}")
    if not constrained.empty:
        base = constrained[constrained["status"].eq("ok")]
        if not base.empty:
            best = base.sort_values(["min_f1", "false_new_tracks"], ascending=[False, True]).groupby("min_f1", sort=False).head(1)
            lines += ["", "## Best methods under F1 constraints", "", best[["min_f1", "method", "F1", "false_new_tracks", "false_track_occupancy_frames"]].to_string(index=False)]
    if not matched.empty:
        lines += ["", "## Matched point warning", ""]
        lines.append("Matched-point rows identify exploitable trade-offs. They do not prove universal superiority unless the full curve or constrained endpoint supports it.")
    lines += [
        "",
        "Claim-safe rule: if Bayesian or a confidence threshold dominates the trust curve, use the trust result only as an interpretable/safety-rule demonstration or matched-operating-point trade-off.",
    ]
    (out / "baseline_dominance_claim_safe.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_metadata(out: Path, args: argparse.Namespace) -> None:
    meta = {
        "status": "success",
        "command": " ".join(["scripts/q1_v5/run_baseline_dominance_audit.py", "--operating-points", args.operating_points, "--output-dir", args.output_dir]),
        "git_commit": git(["rev-parse", "HEAD"]),
        "input": args.operating_points,
    }
    (out / "run_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def git(args: list[str]) -> str:
    try:
        return subprocess.check_output(["git", *args], text=True).strip()
    except Exception:
        return "unavailable"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--operating-points", default="outputs/results/q1_v5/operating_curves/bytetrack/operating_points_raw.csv")
    p.add_argument("--reference-method", default="legacy_geometry_dynamic_no_multiagent")
    p.add_argument("--f1-margins", nargs="+", type=float, default=[0.005, 0.010, 0.015])
    p.add_argument("--output-dir", default="outputs/results/q1_v5/baseline_dominance/bytetrack")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if args.dry_run:
        print(f"dry_run=ok output={out}")
        return
    points = pd.read_csv(args.operating_points)
    points["is_pareto_f1_false_new_occupancy"] = pareto_front(points, ["F1"], ["false_new_tracks", "false_track_occupancy_frames"])
    dominance = dominance_matrix(points)
    baseline = points[points["method"].eq("tracker_baseline")]
    constrained_frames = []
    if not baseline.empty:
        base_f1 = float(baseline.iloc[0]["F1"])
        for margin in args.f1_margins:
            constrained_frames.append(curve_best_at_f1(points, base_f1 - float(margin)).assign(delta_margin=margin, baseline_F1=base_f1))
    constrained = pd.concat(constrained_frames, ignore_index=True) if constrained_frames else pd.DataFrame()
    matched = matched_against(points, args.reference_method)
    points.to_csv(out / "operating_points_with_pareto_flags.csv", index=False)
    dominance.to_csv(out / "dominance_matrix.csv", index=False)
    constrained.to_csv(out / "noninferiority_constrained_best.csv", index=False)
    matched.to_csv(out / "matched_reference_audit.csv", index=False)
    write_claim(out, points, dominance, constrained, matched)
    write_metadata(out, args)
    print(f"status=ok output={out} points={len(points)}")


if __name__ == "__main__":
    main()
