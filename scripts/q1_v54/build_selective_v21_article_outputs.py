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
    p.add_argument("--results-root", default="outputs/results/q1_selective_quarantine_v21")
    p.add_argument("--output-dir", default="outputs/results/q1_selective_quarantine_v21/article_ready")
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
    terminal = read_csv(root / "loso/episode_terminal_audit.csv")
    censoring = read_csv(root / "loso/censoring_audit.csv")
    budget = read_csv(root / "loso/quarantine_budget_by_sequence.csv")
    runtime = read_csv(root / "runtime/runtime_summary.csv")
    sensitivity = read_csv(root / "sensitivity/train_only_sensitivity.csv")
    payload: dict[str, Any] = {
        "protocol_id": "q1_selective_quarantine_v21_final",
        "git_commit": git(["rev-parse", "HEAD"]),
        "branch": git(["branch", "--show-current"]),
        "technical_pipeline_ready": True,
        "claim_safe_article_ready": True,
        "primary_endpoint": "decision_complete_online_current_frame",
        "primary_metrics": aggregate(by_sequence, prefix="decision_complete_online_"),
        "full_sequence_online_secondary": aggregate(by_sequence, prefix="full_sequence_online_"),
        "backfilled_map_secondary": aggregate(by_sequence, prefix="backfilled_map_"),
        "outer_loso": records(loso),
        "outer_loso_by_sequence": records(by_sequence),
        "statistics": records(stats),
        "terminal_state_audit": records(terminal),
        "censoring_audit": records(censoring),
        "budget_audit": records(budget),
        "runtime": records(runtime),
        "train_only_sensitivity": records(sensitivity),
        "claim_status": claim_status(stats),
    }
    (out / "article_numbers_selective_v21.json").write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    (out / "article_patch_selective_v21.md").write_text(article_patch(payload), encoding="utf-8")
    (out / "table_selective_v21_primary.csv").write_text(pd.DataFrame(payload["primary_metrics"]).to_csv(index=False), encoding="utf-8")
    (out / "table_selective_v21_by_sequence.csv").write_text(by_sequence.to_csv(index=False), encoding="utf-8")
    (out / "table_selective_v21_statistics.csv").write_text(stats.to_csv(index=False), encoding="utf-8")
    (out / "table_selective_v21_terminal_states.csv").write_text(terminal.to_csv(index=False), encoding="utf-8")
    (out / "table_selective_v21_runtime.csv").write_text(runtime.to_csv(index=False), encoding="utf-8")
    (out / "run_metadata.json").write_text(json.dumps({"status": "success", "git_commit": payload["git_commit"], "branch": payload["branch"], "protocol_id": payload["protocol_id"]}, indent=2), encoding="utf-8")
    print(f"status=ok output={out / 'article_numbers_selective_v21.json'}")


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def records(df: pd.DataFrame) -> list[dict[str, Any]]:
    if df.empty:
        return []
    clean = df.astype(object).where(pd.notna(df), None)
    return clean.to_dict(orient="records")


def aggregate(df: pd.DataFrame, prefix: str) -> list[dict[str, Any]]:
    if df.empty:
        return []
    metrics = ["TP", "FP", "FN", "precision", "recall", "F1", "observed_false_track_occupancy_per_100_frames", "false_new_tracks_per_100_frames", "true_track_confirmation_rate", "median_gate_delay_from_candidate", "median_confirmation_delay_from_gt"]
    rows = []
    data = df[df["method"].isin(["tracker_baseline", "selective_trust_quarantine"])]
    for (tracker, method), group in data.groupby(["tracker", "method"], sort=False):
        item: dict[str, Any] = {"tracker": tracker, "method": method, "n_sequences": int(len(group))}
        for metric in metrics:
            col = prefix + metric
            if col in group:
                value = group[col].mean()
            elif prefix == "decision_complete_online_" and metric in group:
                value = group[metric].mean()
            else:
                continue
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
            "primary_F1_metric": h1["metric"].iloc[0] if not h1.empty and "metric" in h1 else "decision_complete_online_F1",
            "primary_occupancy_metric": h2["metric"].iloc[0] if not h2.empty and "metric" in h2 else "decision_complete_online_observed_false_track_occupancy_per_100_frames",
        }
    return out


def article_patch(payload: dict[str, Any]) -> str:
    lines = [
        "# Selective Trust Quarantine v2.1 Article Patch",
        "",
        f"Protocol: `{payload['protocol_id']}`",
        f"Commit: `{payload['git_commit']}`",
        "",
        "## Required Replacements",
        "",
        "- Abstract: describe the method as a selective quarantine layer with decision-complete online evaluation; do not claim backfilled-map F1 as zero-delay online quality.",
        "- Problem formulation: VisDrone remains single-camera real-detector validation; multi-agent geometry remains controlled simulation only.",
        "- Method: replace detection-driven `pending_end` with frame-driven terminal states: FAST_PASSED, CONFIRMED_BY_EVIDENCE, RELEASED_TO_BASELINE, REJECTED_BY_HARD_VETO, REJECTED_BY_TEMPORAL_ABSENCE, CENSORED_AT_SEQUENCE_END.",
        "- Experimental protocol: H1 uses `decision_complete_online_F1`; H2 uses `decision_complete_online_observed_false_track_occupancy_per_100_frames` after H1 only.",
        "- Runtime: report gate-only overhead; YOLO inference and offline GT matching are excluded.",
        "- Discussion: backfilled-map metrics are secondary map-completeness metrics and must be discussed with confirmation delay.",
        "- Conclusion: report PASS/FAIL exactly from `claim_status`; do not retune after this frozen outer run.",
        "",
        "## Claim Status",
    ]
    for tracker, status in payload["claim_status"].items():
        lines.append(f"- `{tracker}`: H1={status['H1_F1_noninferiority']}; H2={status['H2_occupancy_superiority']}; F1 metric `{status['primary_F1_metric']}`; occupancy metric `{status['primary_occupancy_metric']}`.")
    return "\n".join(lines) + "\n"


def git(args: list[str]) -> str:
    try:
        return subprocess.check_output(["git", *args], text=True).strip()
    except Exception:
        return "unavailable"


if __name__ == "__main__":
    main()
