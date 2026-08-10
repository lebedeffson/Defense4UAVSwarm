#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--tracker-summary", default="outputs/results/q1_final_plus/tracker_comparison_yolov8s_v9/tracker_comparison_summary.csv")
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
    df = pd.read_csv(args.tracker_summary)
    rows = []
    wanted = [
        ("bytetrack", "none"),
        ("bytetrack", "geometry_dynamic_no_multiagent"),
        ("ocsort", "none"),
        ("ocsort", "geometry_dynamic_no_multiagent"),
    ]
    for tracker, mode in wanted:
        hit = df[(df["tracker"].astype(str).eq(tracker)) & (df["trust_mode"].astype(str).eq(mode)) & (df["available"].astype(str).str.lower().eq("true"))]
        if hit.empty:
            rows.append({"tracker": tracker, "trust_mode": mode, "available": False})
        else:
            r = hit.iloc[0].to_dict()
            r["available"] = True
            rows.append(r)
    frozen = pd.DataFrame(rows)
    frozen.to_csv(out / "legacy_baseline_freeze.csv", index=False)
    (out / "legacy_baseline_freeze.md").write_text(write_md(frozen), encoding="utf-8")
    print(f"status=ok output={out / 'legacy_baseline_freeze.csv'}")


def write_md(df: pd.DataFrame) -> str:
    lines = ["# Legacy Baseline Freeze", "", "These values are copied from the v9 tracker-comparison protocol and are kept separate from q1_v5 operating-curve candidate-space metrics.", ""]
    cols = ["tracker", "trust_mode", "F1", "false_new_tracks", "F1_delta_vs_tracker", "false_new_delta_vs_tracker"]
    present = [c for c in cols if c in df]
    lines.append(df[present].to_string(index=False))
    lines.append("")
    lines.append("Do not mix these frozen tracker-comparison false-new counts with feature-audit operating-curve false-new counts without stating the protocol difference.")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
