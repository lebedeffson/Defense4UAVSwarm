#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

import pandas as pd


PARAMS = ["quarantine_fraction_max", "maximum_quarantine_frames", "relative_weight", "evidence_decay", "confirmation_support"]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--inner-results", default="outputs/results/q1_selective_quarantine_v21/loso/inner_selection_results.csv")
    p.add_argument("--output-dir", default="outputs/results/q1_selective_quarantine_v21/sensitivity")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()
    out = Path(args.output_dir)
    if out.exists() and any(out.iterdir()) and not args.overwrite:
        raise SystemExit(f"Output exists; use --overwrite: {out}")
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.inner_results) if Path(args.inner_results).exists() else pd.DataFrame()
    rows = build_rows(df)
    result = pd.DataFrame(rows)
    result.to_csv(out / "train_only_sensitivity.csv", index=False)
    (out / "train_only_sensitivity_claim_safe.md").write_text(write_report(result), encoding="utf-8")
    (out / "run_metadata.json").write_text(json.dumps({"status": "success", "git_commit": git(["rev-parse", "HEAD"]), "branch": git(["branch", "--show-current"])}, indent=2), encoding="utf-8")
    print(f"status=ok output={out / 'train_only_sensitivity.csv'} rows={len(result)}")


def build_rows(df: pd.DataFrame) -> list[dict[str, Any]]:
    if df.empty:
        return []
    data = df[df["method"].eq("selective_trust_quarantine")].copy()
    parsed = data["selected_parameter_json"].astype(str).map(parse_json)
    for param in PARAMS:
        data[param] = [item.get(param) for item in parsed]
    rows: list[dict[str, Any]] = []
    for param in PARAMS:
        if data[param].nunique(dropna=True) < 2:
            rows.append({"parameter": param, "value": "not_estimable", "number_of_inner_evaluations": int(len(data)), "feasible_fraction": pd.NA, "mean_train_delta_F1": pd.NA, "worst_train_delta_F1": pd.NA, "mean_train_delta_occupancy_per_100_frames": pd.NA})
            continue
        for value, group in data.groupby(param, dropna=False):
            feasible = group["selection_status"].astype(str).str.contains("selected|passthrough", case=False, regex=True)
            rows.append(
                {
                    "parameter": param,
                    "value": value,
                    "number_of_inner_evaluations": int(len(group)),
                    "feasible_fraction": float(feasible.mean()) if len(group) else 0.0,
                    "mean_train_delta_F1": float(group["inner_mean_F1_delta"].mean()),
                    "worst_train_delta_F1": float(group["inner_min_F1_delta"].min()),
                    "mean_train_delta_occupancy_per_100_frames": float(group["inner_mean_occupancy_per_100_frames"].mean()),
                }
            )
    return rows


def parse_json(text: str) -> dict[str, Any]:
    try:
        return json.loads(text)
    except Exception:
        return {}


def write_report(df: pd.DataFrame) -> str:
    return "# Train-Only Sensitivity\n\nThis table uses only inner-selection records. No outer-test metric is used for parameter interpretation.\n\n" + df.to_string(index=False) + "\n"


def git(args: list[str]) -> str:
    try:
        return subprocess.check_output(["git", *args], text=True).strip()
    except Exception:
        return "unavailable"


if __name__ == "__main__":
    main()
