#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--curves", nargs="+", default=["outputs/results/q1_v542/curves/bytetrack/corrected_operating_points.csv", "outputs/results/q1_v542/curves/ocsort/corrected_operating_points.csv"])
    p.add_argument("--output-dir", default="outputs/results/q1_v542/matched_points")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    out = Path(args.output_dir)
    if out.exists() and any(out.iterdir()) and not args.overwrite and not args.dry_run:
        raise SystemExit(f"Output exists; use --overwrite: {out}")
    out.mkdir(parents=True, exist_ok=True)
    if args.dry_run:
        print(f"dry_run=ok output={out}")
        return
    rows = []
    pareto_rows = []
    for curve_path in args.curves:
        df = pd.read_csv(curve_path)
        tracker = "ocsort" if "ocsort" in str(curve_path) else "bytetrack"
        base = df[df["method"].eq("tracker_baseline")].iloc[0]
        for row in df.itertuples(index=False):
            if row.method == "tracker_baseline":
                continue
            target = float(base.false_new_tracks)
            candidate = float(row.false_new_tracks)
            gap = abs(candidate - target)
            rel = gap / max(1.0, target)
            rows.append(
                {
                    "tracker": tracker,
                    "reference_method": "tracker_baseline",
                    "candidate_method": row.method,
                    "matching_target": "false_new_tracks",
                    "reference_value": target,
                    "candidate_value": candidate,
                    "absolute_match_gap": gap,
                    "relative_match_gap": rel,
                    "is_match_feasible": rel <= 0.05,
                }
            )
        for idx, row in df.iterrows():
            dominated = ((df["F1"] >= row["F1"]) & (df["observed_false_track_occupancy_rows"] <= row["observed_false_track_occupancy_rows"]) & ((df["F1"] > row["F1"]) | (df["observed_false_track_occupancy_rows"] < row["observed_false_track_occupancy_rows"]))).any()
            pareto_rows.append({"tracker": tracker, "method": row["method"], "parameter_json": row.get("parameter_json", ""), "F1": row["F1"], "observed_false_track_occupancy_rows": row["observed_false_track_occupancy_rows"], "is_pareto": not bool(dominated)})
    pd.DataFrame(rows).to_csv(out / "matched_point_audit.csv", index=False)
    pd.DataFrame(pareto_rows).to_csv(out / "pareto_points.csv", index=False)
    print(f"status=ok output={out}")


if __name__ == "__main__":
    main()
