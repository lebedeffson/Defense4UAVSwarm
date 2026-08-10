#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from defense4uavswarm.q1_v5.frame_manifest import add_gt_presence, build_visdrone_frame_manifest, merge_manifest_dimensions
from defense4uavswarm.q1_v5.initiation_gate import assign_episode_ids, split_duplicate_observation_tracklets
from defense4uavswarm.q1_v5.selective_quarantine import SelectiveTrustQuarantineConfig, selective_quarantine_gate_result
from defense4uavswarm.q1_visdrone import load_gt_protocol
from scripts.q1_v54.run_corrected_operating_curves import gate_config, load_candidates


REPO_ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", default="outputs/q1_practical_closure_v7/bound_audit")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()
    out = Path(args.output_dir)
    if out.exists() and args.overwrite:
        shutil.rmtree(out)
    if out.exists() and any(out.iterdir()) and not args.overwrite:
        raise SystemExit(f"Output exists; use --overwrite: {out}")
    out.mkdir(parents=True, exist_ok=True)

    prefix_rows = []
    for item in tracker_inputs():
        prefix_rows.extend(process_tracker(item))
    prefix = pd.DataFrame(prefix_rows)
    prefix.to_csv(out / "prefix_bound_audit.csv", index=False)
    violations = prefix[
        prefix["budget_balance_violation"] | prefix["episode_bound_violation"] | prefix["budget_bound_violation"]
    ].copy()
    violations.to_csv(out / "bound_violations.csv", index=False)
    summary = sequence_summary(prefix)
    summary.to_csv(out / "sequence_bound_summary.csv", index=False)
    write_plot(summary, out)
    stats = {
        "status": "success",
        "git_commit": git(["rev-parse", "HEAD"]),
        "git_branch": git(["branch", "--show-current"]),
        "prefix_rows": int(len(prefix)),
        "sequence_rows": int(len(summary)),
        "budget_balance_violations": int(prefix["budget_balance_violation"].sum()) if not prefix.empty else 0,
        "episode_bound_violations": int(prefix["episode_bound_violation"].sum()) if not prefix.empty else 0,
        "budget_bound_violations": int(prefix["budget_bound_violation"].sum()) if not prefix.empty else 0,
        "actual_to_episode_ratio_max": float(summary["actual_to_episode_ratio"].max()) if not summary.empty else 0.0,
        "actual_to_budget_ratio_max": float(summary["actual_to_budget_ratio"].max()) if not summary.empty else 0.0,
    }
    (out / "bound_tightness_statistics.json").write_text(json.dumps(stats, indent=2, sort_keys=True), encoding="utf-8")
    print(f"status=ok output={out} prefix_rows={len(prefix)} violations={len(violations)}")


def tracker_inputs() -> list[dict[str, str]]:
    return [
        {
            "tracker": "ByteTrack",
            "config": "configs/q1_v54/bytetrack.yaml",
            "selection": "outputs/results/q1_selective_quarantine_v21/loso/bytetrack/selected_configs_by_fold.json",
        },
        {
            "tracker": "OC-SORT",
            "config": "configs/q1_v54/ocsort.yaml",
            "selection": "outputs/results/q1_selective_quarantine_v21/loso/ocsort/selected_configs_by_fold.json",
        },
        {
            "tracker": "SORT",
            "config": "configs/q1_v54/sort.yaml",
            "selection": "outputs/q1_practical_closure_v7/sort/loso/selected_configs_by_fold.json",
        },
    ]


def process_tracker(item: dict[str, str]) -> list[dict[str, Any]]:
    cfg = yaml.safe_load(Path(item["config"]).read_text(encoding="utf-8")) or {}
    det = load_candidates(cfg)
    gt, _ignored = load_gt_protocol(cfg["dataset_root"], sorted(det["sequence_id"].unique()))
    manifest = add_gt_presence(build_visdrone_frame_manifest(cfg["dataset_root"], sorted(det["sequence_id"].unique())), gt)
    det = merge_manifest_dimensions(det, manifest)
    det = split_duplicate_observation_tracklets(det)
    det = assign_episode_ids(det, gate_config(cfg).max_track_gap)
    selected = load_selected_selective(Path(item["selection"]))
    rows = []
    for fold_idx, seq in enumerate(sorted(det["sequence_id"].unique())):
        d = det[det["sequence_id"].eq(seq)].copy()
        fm = manifest[manifest["sequence_id"].eq(seq)].copy()
        qparams = selected[(fold_idx, seq)]
        qcfg = SelectiveTrustQuarantineConfig.from_mapping(qparams)
        gate = selective_quarantine_gate_result(d, fm, qcfg)
        rows.extend(prefix_rows_for_sequence(item["tracker"], fold_idx, seq, d, gate, qcfg))
    return rows


def load_selected_selective(path: Path) -> dict[tuple[int, str], dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    out = {}
    for row in payload:
        if row.get("method") != "selective_trust_quarantine":
            continue
        out[(int(row["outer_fold"]), str(row["outer_test_sequence"]))] = json.loads(row["selected_parameter_json"])
    if not out:
        raise RuntimeError(f"No selected selective configs in {path}")
    return out


def prefix_rows_for_sequence(tracker: str, fold_idx: int, seq: str, candidates: pd.DataFrame, gate, qcfg: SelectiveTrustQuarantineConfig) -> list[dict[str, Any]]:
    meta = gate.episode_metadata.copy()
    if meta.empty:
        return []
    meta = meta.sort_values(["first_candidate_frame_id", "tracklet_id", "episode_id"]).reset_index(drop=True)
    online_mask = gate.online_acceptance_mask.astype(bool) if gate.online_acceptance_mask is not None else gate.acceptance_mask.astype(bool)
    by_episode = {key: group.copy() for key, group in candidates.groupby(["sequence_id", "tracklet_id", "episode_id"], sort=False)}
    l_q = int(qcfg.maximum_quarantine_frames)
    actual_d = 0
    n_q = 0
    cumulative_q = 0.0
    rows = []
    for prefix_idx, row in enumerate(meta.itertuples(index=False), start=1):
        key = (str(row.sequence_id), str(row.tracklet_id), int(row.episode_id))
        first_frame = int(row.first_candidate_frame_id)
        configured_q = float(getattr(row, "configured_quarantine_fraction", qcfg.quarantine_fraction_max) or 0.0)
        was_quarantined = bool(getattr(row, "was_quarantined", False))
        affected = actual_deviation_in_window(by_episode.get(key, candidates.iloc[0:0]), online_mask, first_frame, l_q)
        actual_d += int(affected)
        n_q += 1 if was_quarantined else 0
        cumulative_q += configured_q
        episode_bound = int(l_q * n_q)
        budget_quarantine_limit = int(math.floor(cumulative_q + 1e-9))
        budget_bound = int(l_q * budget_quarantine_limit)
        rows.append(
            {
                "tracker": tracker,
                "outer_fold": int(fold_idx),
                "sequence": seq,
                "prefix_index": int(prefix_idx),
                "L_Q": int(l_q),
                "q_k": configured_q,
                "cumulative_q": float(cumulative_q),
                "quarantine_count": int(n_q),
                "actual_D": int(actual_d),
                "episode_bound": int(episode_bound),
                "budget_bound": int(budget_bound),
                "budget_balance_violation": bool(n_q > budget_quarantine_limit),
                "episode_bound_violation": bool(actual_d > episode_bound),
                "budget_bound_violation": bool(episode_bound > budget_bound),
            }
        )
    return rows


def actual_deviation_in_window(group: pd.DataFrame, online_mask: pd.Series, first_frame: int, l_q: int) -> int:
    if group.empty:
        return 0
    last_frame = int(first_frame) + max(0, int(l_q)) - 1
    window = group[group["frame_id"].astype(int).between(int(first_frame), int(last_frame))]
    if window.empty:
        return 0
    return int((~online_mask.loc[window.index].astype(bool)).sum())


def sequence_summary(prefix: pd.DataFrame) -> pd.DataFrame:
    if prefix.empty:
        return pd.DataFrame(
            columns=[
                "tracker",
                "outer_fold",
                "sequence",
                "quarantine_count",
                "actual_D_max",
                "episode_bound_max",
                "budget_bound_max",
                "actual_to_episode_ratio",
                "actual_to_budget_ratio",
            ]
        )
    rows = []
    for (tracker, fold, seq), group in prefix.groupby(["tracker", "outer_fold", "sequence"], sort=False):
        actual = int(group["actual_D"].max())
        episode = int(group["episode_bound"].max())
        budget = int(group["budget_bound"].max())
        rows.append(
            {
                "tracker": tracker,
                "outer_fold": int(fold),
                "sequence": seq,
                "quarantine_count": int(group["quarantine_count"].max()),
                "actual_D_max": actual,
                "episode_bound_max": episode,
                "budget_bound_max": budget,
                "actual_to_episode_ratio": float(actual / episode) if episode else 0.0,
                "actual_to_budget_ratio": float(actual / budget) if budget else 0.0,
            }
        )
    return pd.DataFrame(rows)


def write_plot(summary: pd.DataFrame, out: Path) -> None:
    plt.figure(figsize=(9, 5))
    x = np.arange(len(summary))
    labels = [f"{r.tracker}\\n{r.sequence}" for r in summary.itertuples(index=False)]
    plt.plot(x, summary["actual_D_max"], marker="o", label="Фактическое D(n)")
    plt.plot(x, summary["episode_bound_max"], marker="o", label="Граница L_Q N_Q(n)")
    plt.plot(x, summary["budget_bound_max"], marker="o", label="Бюджетная граница")
    plt.xticks(x, labels, rotation=60, ha="right", fontsize=7)
    plt.ylabel("Максимальное значение по префиксам")
    plt.xlabel("Последовательность")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / "fig_bound_actual_vs_limit_ru.png", dpi=160)
    plt.close()


def git(args: list[str]) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True).strip()
    except Exception:
        return "unavailable"


if __name__ == "__main__":
    main()
