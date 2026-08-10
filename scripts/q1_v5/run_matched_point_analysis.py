#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def closest(group: pd.DataFrame, column: str, target: float) -> pd.Series:
    idx = (group[column].astype(float) - float(target)).abs().sort_values().index[0]
    return group.loc[idx]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--operating-points", required=True)
    p.add_argument("--trust-method", default="trust_balanced")
    p.add_argument("--trust-parameter", type=float, default=None)
    p.add_argument("--delta-margins", nargs="+", type=float, default=[0.005, 0.010, 0.015])
    p.add_argument("--output-dir", required=True)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if args.dry_run:
        print(f"dry_run=ok output={out}")
        return
    df = pd.read_csv(args.operating_points)
    trust = df[df["method"].eq(args.trust_method)].copy()
    if args.trust_parameter is not None:
        trust = trust[(trust["parameter_value"].astype(float) - args.trust_parameter).abs() < 1e-9]
    if trust.empty:
        raise SystemExit(f"No trust point found for {args.trust_method}")
    trust_point = trust.sort_values(["false_new_tracks", "F1"], ascending=[True, False]).iloc[0]
    rows = []
    for method, group in df.groupby("method", sort=True):
        if method == args.trust_method:
            continue
        same_false = closest(group, "false_new_tracks", float(trust_point["false_new_tracks"]))
        r = same_false.to_dict()
        r.update({"comparison_type": "same_false_new", "target_false_new": trust_point["false_new_tracks"], "target_F1": trust_point["F1"], "delta_F1_vs_trust": float(r["F1"]) - float(trust_point["F1"])})
        rows.append(r)
        same_f1 = closest(group, "F1", float(trust_point["F1"]))
        r = same_f1.to_dict()
        r.update({"comparison_type": "same_F1", "target_false_new": trust_point["false_new_tracks"], "target_F1": trust_point["F1"], "delta_false_new_vs_trust": float(r["false_new_tracks"]) - float(trust_point["false_new_tracks"])})
        rows.append(r)
    base = df[df["method"].eq("tracker_baseline")].iloc[0]
    for margin in args.delta_margins:
        min_f1 = float(base["F1"]) - float(margin)
        for method, group in df.groupby("method", sort=True):
            feasible = group[group["F1"].astype(float) >= min_f1]
            if feasible.empty:
                continue
            best = feasible.sort_values(["false_new_tracks", "recall"], ascending=[True, False]).iloc[0]
            r = best.to_dict()
            r.update({"comparison_type": "noninferiority_constrained", "delta_margin": margin, "baseline_F1": float(base["F1"]), "min_allowed_F1": min_f1})
            rows.append(r)
    result = pd.DataFrame(rows)
    result.to_csv(out / "matched_operating_points.csv", index=False)
    write_md(out / "matched_operating_points.md", trust_point, result)
    plot(result, out / "fig_matched_points.png")
    print(f"status=ok output={out / 'matched_operating_points.csv'} rows={len(result)}")


def write_md(path: Path, trust: pd.Series, result: pd.DataFrame) -> None:
    lines = ["# Matched Operating Point Analysis", ""]
    lines.append(f"Reference trust point: `{trust['method']}` parameter={trust['parameter_value']} F1={trust['F1']:.6f}, false_new={trust['false_new_tracks']:.0f}.")
    lines.append("")
    same = result[result["comparison_type"].eq("same_false_new")]
    if not same.empty:
        lines += ["## Same false-new comparison", "", same[["method", "parameter", "parameter_value", "false_new_tracks", "F1", "recall", "delta_F1_vs_trust"]].to_string(index=False), ""]
    ni = result[result["comparison_type"].eq("noninferiority_constrained")]
    if not ni.empty:
        lines += ["## Non-inferiority constrained optimum", "", ni[["method", "delta_margin", "parameter", "parameter_value", "F1", "false_new_tracks", "false_track_occupancy_frames"]].to_string(index=False), ""]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot(result: pd.DataFrame, output: Path) -> None:
    same = result[result["comparison_type"].eq("same_false_new")]
    if same.empty:
        return
    plt.figure(figsize=(7, 4), dpi=180)
    plt.bar(same["method"], same["F1"], color="#4b77be")
    plt.xticks(rotation=35, ha="right")
    plt.ylabel("F1 at matched false-new count")
    plt.tight_layout()
    plt.savefig(output)
    plt.close()


if __name__ == "__main__":
    main()
