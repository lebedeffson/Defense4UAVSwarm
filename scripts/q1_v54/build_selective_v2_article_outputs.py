#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

import pandas as pd


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--results-root", default="outputs/results/q1_selective_quarantine_v2")
    p.add_argument("--output-dir", default="outputs/results/q1_selective_quarantine_v2/article_ready")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()
    root = Path(args.results_root)
    out = Path(args.output_dir)
    if out.exists() and any(out.iterdir()) and not args.overwrite:
        raise SystemExit(f"Output exists; use --overwrite: {out}")
    out.mkdir(parents=True, exist_ok=True)

    loso = read_csv(root / "loso/outer_test_summary.csv")
    by_sequence = read_csv(root / "loso/outer_test_by_sequence.csv")
    stats = read_csv(root / "statistics/statistical_results.csv")
    budget = read_csv(root / "loso/quarantine_budget_by_sequence.csv")
    payload: dict[str, Any] = {
        "protocol_id": "q1_selective_quarantine_v2_corrected",
        "git_commit": git(["rev-parse", "HEAD"]),
        "branch": git(["branch", "--show-current"]),
        "method": "selective_trust_quarantine_v2",
        "online_metrics": aggregate_method_metrics(by_sequence, prefix="online_current_frame_"),
        "backfilled_map_metrics": aggregate_method_metrics(by_sequence, prefix=""),
        "outer_loso": records(loso),
        "outer_loso_by_sequence": records(by_sequence),
        "normalized_statistics": records(stats),
        "budget_audit": records(budget),
        "claim_status": claim_status(stats),
    }
    (out / "article_numbers_selective_v2.json").write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    (out / "table_selective_v2_outer_loso.csv").write_text(loso.to_csv(index=False), encoding="utf-8")
    (out / "table_selective_v2_outer_loso_by_sequence.csv").write_text(by_sequence.to_csv(index=False), encoding="utf-8")
    (out / "table_selective_v2_statistics.csv").write_text(stats.to_csv(index=False), encoding="utf-8")
    (out / "table_selective_v2_budget_audit.csv").write_text(budget.to_csv(index=False), encoding="utf-8")
    (out / "article_patch_selective_v2.md").write_text(article_patch(payload, by_sequence), encoding="utf-8")
    (out / "run_metadata.json").write_text(json.dumps({"status": "success", "git_commit": payload["git_commit"], "branch": payload["branch"]}, indent=2), encoding="utf-8")
    print(f"status=ok output={out / 'article_numbers_selective_v2.json'}")


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def records(df: pd.DataFrame) -> list[dict[str, Any]]:
    if df.empty:
        return []
    clean = df.astype(object).where(pd.notna(df), None)
    return clean.to_dict(orient="records")


def extract_method_metrics(loso: pd.DataFrame, prefix: str) -> list[dict[str, Any]]:
    if loso.empty:
        return []
    rows = []
    for row in loso[loso["method"].isin(["tracker_baseline", "selective_trust_quarantine"])].itertuples(index=False):
        item = {"tracker": row.tracker, "method": row.method}
        for metric in ["TP", "FP", "FN", "precision", "recall", "F1", "observed_false_track_occupancy_per_100_frames", "observed_false_track_occupancy_rows", "false_new_tracks_per_100_frames"]:
            col = prefix + metric
            if col in loso.columns:
                item[metric] = getattr(row, col)
            elif not prefix and metric in loso.columns:
                item[metric] = getattr(row, metric)
        rows.append(item)
    return rows


def aggregate_method_metrics(by_sequence: pd.DataFrame, prefix: str) -> list[dict[str, Any]]:
    if by_sequence.empty:
        return []
    metrics = [
        "TP",
        "FP",
        "FN",
        "precision",
        "recall",
        "F1",
        "observed_false_track_occupancy_per_100_frames",
        "observed_false_track_occupancy_rows",
        "false_new_tracks",
        "false_new_tracks_per_100_frames",
        "track_initiation_precision",
    ]
    rows: list[dict[str, Any]] = []
    data = by_sequence[by_sequence["method"].isin(["tracker_baseline", "selective_trust_quarantine"])]
    for (tracker, method), group in data.groupby(["tracker", "method"], sort=False):
        item: dict[str, Any] = {"tracker": tracker, "method": method, "n_sequences": int(len(group))}
        for metric in metrics:
            col = prefix + metric
            if col in group.columns:
                value = group[col].mean()
                item[metric] = None if pd.isna(value) else float(value)
            elif not prefix and metric in group.columns:
                value = group[metric].mean()
                item[metric] = None if pd.isna(value) else float(value)
        rows.append(item)
    return rows


def claim_status(stats: pd.DataFrame) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if stats.empty:
        return out
    for tracker, group in stats[stats["method"].eq("selective_trust_quarantine")].groupby("tracker", sort=False):
        h1 = group[group["hypothesis"].eq("H1_F1_noninferiority")]
        h2 = group[group["hypothesis"].eq("H2_occupancy_superiority")]
        out[str(tracker)] = {
            "H1_F1_noninferiority": h1["status"].iloc[0] if not h1.empty else "missing",
            "H2_occupancy_superiority": h2["status"].iloc[0] if not h2.empty else "missing",
            "occupancy_metric": h2["metric"].iloc[0] if not h2.empty and "metric" in h2 else "missing",
        }
    return out


def article_patch(payload: dict[str, Any], by_sequence: pd.DataFrame) -> str:
    lines = [
        "# Selective Trust Quarantine v2 Article Patch",
        "",
        f"Protocol: `{payload['protocol_id']}`",
        f"Commit: `{payload['git_commit']}`",
        "",
        "Use the v2 selective quarantine numbers only from `article_numbers_selective_v2.json`.",
        "Do not mix them with legacy RF, label-scarcity, runtime, or v542 article-number tables unless those sections are explicitly marked legacy protocol.",
        "",
        "## Claim Boundary",
    ]
    for tracker, status in payload["claim_status"].items():
        lines.append(f"- `{tracker}`: H1={status['H1_F1_noninferiority']}, H2={status['H2_occupancy_superiority']} using `{status['occupancy_metric']}`.")
    lines += [
        "",
        "## Required Wording",
        "",
        "The primary F1 is backfilled-map F1. It measures the quality of the confirmed map after buffered observations are restored. It is not a zero-delay online publication metric.",
        "",
        "Online current-frame metrics are reported separately under `online_current_frame_*`; confirmation delay must be discussed alongside backfilled-map F1.",
        "",
        "The method should be described as a tracker-dependent selective intervention layer with a safe fallback/passthrough policy.",
    ]
    if not by_sequence.empty:
        lines += ["", "## Per-Sequence Rows", "", by_sequence.head(20).to_string(index=False)]
    return "\n".join(lines) + "\n"


def git(args: list[str]) -> str:
    try:
        return subprocess.check_output(["git", *args], text=True).strip()
    except Exception:
        return "unavailable"


if __name__ == "__main__":
    main()
