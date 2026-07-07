#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--q1-root", default="outputs/results")
    p.add_argument("--output", required=True)
    p.add_argument("--csv-output", required=True)
    args = p.parse_args()
    root = Path(args.q1_root)
    rows = []
    md = ["# Q1 Upgrade Summary", ""]
    for detector in ["yolov8n", "yolov8s"]:
        main_path = root / "q1_real_detector" / f"{detector}_main" / "main_comparison_table.csv"
        label_path = root / "q1_label_scarcity" / detector / "label_budget_mean_std.csv"
        corr_path = root / "q1_corruption_robustness" / detector / "corruption_summary.csv"
        if main_path.exists():
            main = pd.read_csv(main_path)
            md += [f"## {detector} Main", "", "```text", main.to_string(index=False), "```", ""]
            rows.extend(summarize_main(detector, main))
        if label_path.exists():
            label = pd.read_csv(label_path)
            md += [f"## {detector} Label Scarcity", "", summarize_label_md(label), ""]
            rows.extend(summarize_label(detector, label))
        if corr_path.exists():
            corr = pd.read_csv(corr_path)
            md += [f"## {detector} Corruption Robustness", "", summarize_corruption_md(corr), ""]
            rows.extend(summarize_corruption(detector, corr))
    claim = claim_safe(rows)
    md += ["## Claim-Safe Interpretation", "", claim, ""]
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(md), encoding="utf-8")
    csv_out = Path(args.csv_output)
    csv_out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(csv_out, index=False)
    print(f"status=ok output={out}")


def summarize_main(detector: str, frame: pd.DataFrame) -> list[dict]:
    rows = []
    for _, row in frame.iterrows():
        rows.append({"detector": detector, "section": "main", "condition": "full", "method": row["method"], "F1": row.get("F1"), "false_new_tracks": row.get("false_new_tracks"), "available": row.get("available", True)})
    return rows


def summarize_label(detector: str, frame: pd.DataFrame) -> list[dict]:
    rows = []
    for _, row in frame.iterrows():
        rows.append({"detector": detector, "section": "label_scarcity", "condition": row["label_budget"], "method": row["method"], "F1": row.get("F1_mean"), "false_new_tracks": row.get("false_new_tracks_mean"), "available": not pd.isna(row.get("F1_mean"))})
    return rows


def summarize_corruption(detector: str, frame: pd.DataFrame) -> list[dict]:
    rows = []
    for _, row in frame.iterrows():
        rows.append({"detector": detector, "section": "corruption", "condition": row["corruption"], "method": row["method"], "F1": row.get("F1"), "false_new_tracks": row.get("false_new_tracks"), "available": row.get("available", True)})
    return rows


def summarize_label_md(frame: pd.DataFrame) -> str:
    cols = [c for c in ["label_budget", "method", "F1_mean", "F1_std", "false_new_tracks_mean", "false_new_tracks_std"] if c in frame]
    return "```text\n" + frame[cols].to_string(index=False) + "\n```"


def summarize_corruption_md(frame: pd.DataFrame) -> str:
    cols = [c for c in ["corruption", "method", "F1", "false_new_tracks"] if c in frame]
    return "```text\n" + frame[cols].to_string(index=False) + "\n```"


def claim_safe(rows: list[dict]) -> str:
    df = pd.DataFrame(rows)
    if df.empty:
        return "No full Q1 results were found. Do not make Q1 claims."
    lines = []
    for detector in sorted(df["detector"].unique()):
        sub = df[(df["detector"].eq(detector)) & (df["section"].eq("label_scarcity"))]
        zero = sub[sub["condition"].astype(str).isin(["0.0", "0.00", "0"])]
        if not zero.empty:
            rf = zero[zero["method"].eq("rf_learned_gate")]
            lines.append(f"{detector}: RF at 0% labels is {'unavailable' if rf.empty or rf['F1'].isna().all() else 'available - check protocol bug'}.")
            avail = zero[zero["method"].ne("rf_learned_gate")].dropna(subset=["F1"])
            if not avail.empty:
                best = avail.sort_values("F1", ascending=False).iloc[0]
                naive = avail[avail["method"].eq("s_naive")]
                lines.append(f"{detector}: best 0% no-label method by F1 is {best['method']} (F1={best['F1']:.6f}); naive F1={float(naive.iloc[0]['F1']):.6f}" if not naive.empty else f"{detector}: best 0% method is {best['method']}.")
    lines.append("VisDrone results are real-detector single-UAV validation, not real multi-UAV validation.")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
