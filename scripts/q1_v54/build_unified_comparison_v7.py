#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd


METHOD_MAP = {
    "confidence_initiation_gate": "confidence_threshold",
    "m_of_n_confirmation": "m_of_n",
    "bayesian_fixed_terminal": "bayesian_confirmation",
    "selective_trust_quarantine": "selective_quarantine_v21",
    "rf_unbounded": "rf_unbounded",
    "rf_budgeted": "rf_budgeted",
}
METHOD_ORDER = [
    "confidence_threshold",
    "m_of_n",
    "bayesian_confirmation",
    "rf_unbounded",
    "rf_budgeted",
    "selective_quarantine_v21",
]
TRACKER_NAMES = {"bytetrack": "ByteTrack", "ocsort": "OC-SORT", "sort": "SORT"}
REPO_ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", default="outputs/q1_practical_closure_v7/unified_comparison")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--n-resamples", type=int, default=10000)
    args = p.parse_args()
    out = Path(args.output_dir)
    if out.exists() and args.overwrite:
        shutil.rmtree(out)
    if out.exists() and any(out.iterdir()) and not args.overwrite:
        raise SystemExit(f"Output exists; use --overwrite: {out}")
    out.mkdir(parents=True, exist_ok=True)

    fold = load_all_folds()
    comparison, paired = build_comparison(fold, args.seed, args.n_resamples)
    comparison = comparison.sort_values(["tracker_order", "method_order"]).drop(columns=["tracker_order", "method_order"])
    comparison.to_csv(out / "comparison_by_tracker.csv", index=False)
    comparison[comparison["tracker"].eq("ByteTrack")].to_csv(out / "comparison_bytetrack.csv", index=False)
    write_article_tables(comparison, out)
    stats = {
        "status": "success",
        "protocol": "q1_practical_closure_v7_unified_v21_only",
        "git_commit": git(["rev-parse", "HEAD"]),
        "git_branch": git(["branch", "--show-current"]),
        "n_resamples": int(args.n_resamples),
        "seed": int(args.seed),
        "input_rows": int(len(fold)),
        "paired_delta_rows": int(len(paired)),
        "rows_by_tracker": {str(k): int(v) for k, v in comparison.groupby("tracker").size().to_dict().items()},
        "bytetrack_complete_methods": complete_methods(comparison[comparison["tracker"].eq("ByteTrack")]),
        "required_methods": METHOD_ORDER,
        "h1_rule": "ci95_delta_f1_low > -0.01",
        "h2_rule": "h1_pass and ci95_delta_false_rows_high < 0",
    }
    (out / "comparison_statistics.json").write_text(json.dumps(stats, indent=2, sort_keys=True), encoding="utf-8")
    paired.to_csv(out / "paired_sequence_deltas.csv", index=False)
    print(f"status=ok output={out} rows={len(comparison)}")


def load_all_folds() -> pd.DataFrame:
    inputs = [
        Path("outputs/results/q1_selective_quarantine_v21/loso/bytetrack/outer_test_by_sequence.csv"),
        Path("outputs/results/q1_selective_quarantine_v21/loso/ocsort/outer_test_by_sequence.csv"),
        Path("outputs/q1_practical_closure_v7/sort/loso/outer_test_by_sequence.csv"),
        Path("outputs/q1_practical_closure_v7/rf_v21/fold_results.csv"),
    ]
    frames = []
    for source_order, path in enumerate(inputs):
        if not path.exists():
            continue
        df = pd.read_csv(path)
        if df.empty:
            continue
        df = df.copy()
        df["input_path"] = str(path)
        df["source_order"] = int(source_order)
        frames.append(df)
    if not frames:
        raise RuntimeError("No v2.1 fold result files found")
    out = pd.concat(frames, ignore_index=True, sort=False)
    out["tracker"] = out["tracker"].astype(str).map(lambda x: TRACKER_NAMES.get(x.lower(), x))
    out["method_unified"] = out["method"].astype(str).map(METHOD_MAP).fillna(out["method"].astype(str))
    keep = set(METHOD_MAP.values()) | {"tracker_baseline"}
    out = out[out["method_unified"].isin(keep)].copy()
    out = out.sort_values(["tracker", "method_unified", "outer_test_sequence", "source_order"])
    out = out.drop_duplicates(["tracker", "method_unified", "outer_test_sequence"], keep="last")
    return out


def build_comparison(fold: pd.DataFrame, seed: int, n_resamples: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    paired_rows = []
    rng = np.random.default_rng(seed)
    for tracker, group in fold.groupby("tracker", sort=False):
        f1_col = choose_col(group, ["decision_complete_online_F1", "F1"])
        false_rows_col = choose_col(
            group,
            [
                "decision_complete_online_observed_false_track_occupancy_per_100_frames",
                "observed_false_track_occupancy_per_100_frames",
            ],
        )
        false_inits_col = choose_col(
            group,
            ["decision_complete_online_false_new_tracks_per_100_frames", "false_new_tracks_per_100_frames"],
        )
        base = group[group["method_unified"].eq("tracker_baseline")][["outer_test_sequence", f1_col, false_rows_col, false_inits_col]].copy()
        if base.empty:
            continue
        base = base.rename(columns={f1_col: "base_f1", false_rows_col: "base_false_rows", false_inits_col: "base_false_inits"})
        for method in METHOD_ORDER:
            mg = group[group["method_unified"].eq(method)].copy()
            if mg.empty:
                continue
            paired = mg.merge(base, on="outer_test_sequence", how="inner")
            if paired.empty:
                continue
            delta_f1 = paired[f1_col].astype(float).to_numpy() - paired["base_f1"].astype(float).to_numpy()
            delta_rows = paired[false_rows_col].astype(float).to_numpy() - paired["base_false_rows"].astype(float).to_numpy()
            delta_inits = paired[false_inits_col].astype(float).to_numpy() - paired["base_false_inits"].astype(float).to_numpy()
            f1_ci = bootstrap_ci(delta_f1, n_resamples, rng)
            rows_ci = bootstrap_ci(delta_rows, n_resamples, rng)
            h1 = bool(f1_ci[0] > -0.01)
            h2 = bool(h1 and rows_ci[1] < 0.0)
            rows.append(
                {
                    "tracker": tracker,
                    "method": method,
                    "mean_delta_f1": float(delta_f1.mean()),
                    "sample_std_delta_f1": sample_std(delta_f1),
                    "ci95_delta_f1_low": float(f1_ci[0]),
                    "ci95_delta_f1_high": float(f1_ci[1]),
                    "h1_pass": h1,
                    "mean_delta_false_rows_100": float(delta_rows.mean()),
                    "sample_std_delta_false_rows_100": sample_std(delta_rows),
                    "ci95_delta_false_rows_low": float(rows_ci[0]),
                    "ci95_delta_false_rows_high": float(rows_ci[1]),
                    "h2_pass": h2,
                    "mean_delta_false_inits_100": float(delta_inits.mean()),
                    "intervention_fraction": intervention_fraction(paired),
                    "has_deterministic_budget": bool(method in {"rf_budgeted", "selective_quarantine_v21"}),
                    "tracker_order": tracker_order(tracker),
                    "method_order": METHOD_ORDER.index(method),
                }
            )
            for row, df1, drows, dinits in zip(paired.itertuples(index=False), delta_f1, delta_rows, delta_inits):
                paired_rows.append(
                    {
                        "tracker": tracker,
                        "method": method,
                        "outer_test_sequence": row.outer_test_sequence,
                        "delta_f1": float(df1),
                        "delta_false_rows_100": float(drows),
                        "delta_false_inits_100": float(dinits),
                    }
                )
    return pd.DataFrame(rows), pd.DataFrame(paired_rows)


def choose_col(df: pd.DataFrame, candidates: list[str]) -> str:
    for col in candidates:
        if col in df.columns:
            return col
    raise KeyError(f"Missing metric column; tried {candidates}")


def bootstrap_ci(values: np.ndarray, n_resamples: int, rng: np.random.Generator) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    idx = rng.integers(0, len(values), size=(int(n_resamples), len(values)))
    samples = values[idx].mean(axis=1)
    return float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))


def sample_std(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        return 0.0
    return float(values.std(ddof=1))


def intervention_fraction(df: pd.DataFrame) -> float:
    if "realized_quarantine_fraction" not in df.columns:
        return math.nan
    vals = pd.to_numeric(df["realized_quarantine_fraction"], errors="coerce").dropna()
    if vals.empty:
        return math.nan
    return float(vals.mean())


def tracker_order(tracker: str) -> int:
    return {"ByteTrack": 0, "OC-SORT": 1, "SORT": 2}.get(tracker, 99)


def write_article_tables(df: pd.DataFrame, out: Path) -> None:
    cols = [
        "tracker",
        "method",
        "mean_delta_f1",
        "ci95_delta_f1_low",
        "ci95_delta_f1_high",
        "h1_pass",
        "mean_delta_false_rows_100",
        "ci95_delta_false_rows_low",
        "ci95_delta_false_rows_high",
        "h2_pass",
        "intervention_fraction",
    ]
    article = df[cols].copy()
    article.to_csv(out / "table_article_en.csv", index=False)
    ru = article.rename(
        columns={
            "tracker": "трекер",
            "method": "метод",
            "mean_delta_f1": "среднее_delta_F1",
            "ci95_delta_f1_low": "ДИ95_delta_F1_низ",
            "ci95_delta_f1_high": "ДИ95_delta_F1_верх",
            "h1_pass": "H1_пройдена",
            "mean_delta_false_rows_100": "среднее_delta_ложных_строк_100",
            "ci95_delta_false_rows_low": "ДИ95_delta_ложных_строк_низ",
            "ci95_delta_false_rows_high": "ДИ95_delta_ложных_строк_верх",
            "h2_pass": "H2_пройдена",
            "intervention_fraction": "доля_вмешательств",
        }
    )
    ru.to_csv(out / "table_article_ru.csv", index=False)


def complete_methods(df: pd.DataFrame) -> bool:
    return set(df["method"].astype(str)) >= set(METHOD_ORDER)


def git(args: list[str]) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True).strip()
    except Exception:
        return "unavailable"


if __name__ == "__main__":
    main()
