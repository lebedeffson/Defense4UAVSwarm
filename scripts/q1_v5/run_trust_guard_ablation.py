#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import subprocess
import warnings
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from defense4uavswarm.q1_v5.map_contamination import contamination_metrics
from defense4uavswarm.q1_v5.trust_guard import TrustGuardConfig, run_trust_guard_dataframe
from defense4uavswarm.q1_visdrone import Q1Params, compute_metrics, load_gt_protocol, method_acceptance


ABLATION_DESCRIPTIONS = {
    "A0_legacy_balanced": "Current geometry_dynamic_no_multiagent with double confidence and temporal recovery OR.",
    "A1_no_double_confidence": "Uses old instantaneous min(confidence, temporal) directly, without multiplying confidence again.",
    "A2_no_temporal_instant_min": "Removes temporal age from instantaneous T-norm; q=min(confidence, kinematic proxy).",
    "A3_evidence_memory": "Stateful quarantine with exponential evidence memory, confidence only, no veto.",
    "A4_hard_veto": "Evidence memory plus non-compensating confidence veto.",
    "A5_kinematic_veto": "Evidence memory plus working constant-velocity kinematic channel and veto.",
    "A6_relative_confidence": "Evidence memory plus past-only relative confidence, no kinematic veto.",
    "A7_full_trust_guard": "Full TrustGuard v5.2: relative confidence, kinematic consistency, evidence memory, hard veto.",
    "M_of_N_3_5": "M-of-N confirmation baseline with 3 hits in a 5-frame window.",
}


def read_config(path: str | Path) -> dict[str, Any]:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}


def frame_count(gt: pd.DataFrame) -> int:
    return max(1, gt[["sequence_id", "frame_id"]].drop_duplicates().shape[0])


def enrich(method: str, accepted: pd.Series, det: pd.DataFrame, gt: pd.DataFrame) -> dict[str, Any]:
    row = compute_metrics(det, accepted, len(gt), frame_count(gt), method)
    row.update(contamination_metrics(det, accepted))
    row["true_track_confirmation_rate"] = true_track_confirmation_rate(det, accepted)
    row["median_confirmation_delay"] = median_confirmation_delay(det, accepted)
    row["candidate_conditional_recall"] = row["recall"]
    return row


def true_track_confirmation_rate(det: pd.DataFrame, accepted: pd.Series) -> float:
    tp = det[det["eval_is_tp"].astype(bool)]
    if tp.empty:
        return 0.0
    accepted_tp = det[pd.Series(accepted, index=det.index).astype(bool) & det["eval_is_tp"].astype(bool)]
    return accepted_tp["matched_gt_id"].nunique() / max(1, tp["matched_gt_id"].nunique())


def median_confirmation_delay(det: pd.DataFrame, accepted: pd.Series) -> float:
    acc_tp = det[pd.Series(accepted, index=det.index).astype(bool) & det["eval_is_tp"].astype(bool)].copy()
    all_tp = det[det["eval_is_tp"].astype(bool)].copy()
    if acc_tp.empty or all_tp.empty:
        return 0.0
    first_visible = all_tp.groupby(["sequence_id", "matched_gt_id"], sort=False)["frame_id"].min()
    first_confirmed = acc_tp.groupby(["sequence_id", "matched_gt_id"], sort=False)["frame_id"].min()
    delays = []
    for key, frame in first_confirmed.items():
        if key in first_visible:
            delays.append(int(frame) - int(first_visible[key]))
    return float(np.median(delays)) if delays else 0.0


def temporal_support(det: pd.DataFrame) -> pd.Series:
    return (det["temporal_age"].astype(float).clip(0, 3) / 3.0).clip(0, 1)


def q_series(det: pd.DataFrame, include_temporal: bool, include_kinematic: bool) -> pd.Series:
    parts = [det["confidence"].astype(float).clip(0, 1)]
    if include_temporal:
        parts.append(temporal_support(det))
    if include_kinematic:
        parts.append(det.get("k_i", pd.Series(1.0, index=det.index)).astype(float).clip(0, 1))
    return pd.concat(parts, axis=1).min(axis=1)


def m_of_n(det: pd.DataFrame, m: int = 3, n: int = 5, threshold: float = 0.10) -> pd.Series:
    out = pd.Series(False, index=det.index)
    for _, group in det.sort_values(["sequence_id", "tracklet_id", "frame_id"]).groupby(["sequence_id", "tracklet_id"], sort=False):
        hits = (group["confidence"].astype(float) >= threshold).astype(int)
        out.loc[group.index] = hits.rolling(window=n, min_periods=1).sum().to_numpy() >= m
    return out


def trust_guard_acceptance(det: pd.DataFrame, cfg: TrustGuardConfig) -> tuple[pd.Series, pd.DataFrame]:
    events = run_trust_guard_dataframe(det, cfg)
    accepted = pd.Series(False, index=det.index)
    accepted.loc[events["row_index"].to_numpy(dtype=int)] = events["accepted"].to_numpy(dtype=bool)
    return accepted, events


def ablation_acceptance(det: pd.DataFrame, cfg: dict[str, Any], method: str) -> tuple[pd.Series, pd.DataFrame | None]:
    q_threshold = float(cfg.get("legacy_thresholds", {}).get("q_threshold", 0.20))
    base = TrustGuardConfig.from_mapping(cfg.get("trust_guard", {}))
    if method == "A0_legacy_balanced":
        return method_acceptance(det, "geometry_dynamic_no_multiagent", Q1Params()), None
    if method == "A1_no_double_confidence":
        return q_series(det, include_temporal=True, include_kinematic=False) >= q_threshold, None
    if method == "A2_no_temporal_instant_min":
        return q_series(det, include_temporal=False, include_kinematic=True) >= q_threshold, None
    if method == "A3_evidence_memory":
        return trust_guard_acceptance(det, replace(base, use_relative_confidence=False, use_kinematic=False, use_veto=False, use_evidence=True))
    if method == "A4_hard_veto":
        return trust_guard_acceptance(det, replace(base, use_relative_confidence=False, use_kinematic=False, use_veto=True, use_evidence=True))
    if method == "A5_kinematic_veto":
        return trust_guard_acceptance(det, replace(base, use_relative_confidence=False, use_kinematic=True, use_veto=True, use_evidence=True))
    if method == "A6_relative_confidence":
        return trust_guard_acceptance(det, replace(base, use_relative_confidence=True, use_kinematic=False, use_veto=True, use_evidence=True))
    if method == "A7_full_trust_guard":
        return trust_guard_acceptance(det, base)
    if method == "M_of_N_3_5":
        th = float(cfg.get("legacy_thresholds", {}).get("m_of_n_confidence", 0.10))
        return m_of_n(det, 3, 5, th), None
    raise ValueError(f"Unknown ablation method: {method}")


def run_ablation(det: pd.DataFrame, gt: pd.DataFrame, cfg: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    event_frames: list[pd.DataFrame] = []
    acceptances: dict[str, pd.Series] = {}
    for method in ABLATION_DESCRIPTIONS:
        accepted, events = ablation_acceptance(det, cfg, method)
        acceptances[method] = accepted.astype(bool)
        row = enrich(method, accepted, det, gt)
        row["description"] = ABLATION_DESCRIPTIONS[method]
        rows.append(row)
        if events is not None:
            ev = events.copy()
            ev["method"] = method
            event_frames.append(ev)
    summary = pd.DataFrame(rows)
    disagreement = disagreement_table(acceptances)
    event_frames = [ev for ev in event_frames if not ev.empty]
    if event_frames:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            events_out = pd.concat(event_frames, ignore_index=True)
    else:
        events_out = pd.DataFrame()
    return summary, events_out, disagreement


def disagreement_table(acceptances: dict[str, pd.Series]) -> pd.DataFrame:
    rows = []
    methods = list(acceptances)
    for i, a in enumerate(methods):
        for b in methods[i + 1 :]:
            av = acceptances[a].to_numpy(dtype=bool)
            bv = acceptances[b].to_numpy(dtype=bool)
            rows.append(
                {
                    "method_a": a,
                    "method_b": b,
                    "disagreement_rate": float(np.mean(av != bv)),
                    "a_only": int(np.sum(av & ~bv)),
                    "b_only": int(np.sum(~av & bv)),
                }
            )
    return pd.DataFrame(rows)


def write_claim(summary: pd.DataFrame, output: Path) -> None:
    legacy = summary[summary["method"].eq("A0_legacy_balanced")].iloc[0]
    full = summary[summary["method"].eq("A7_full_trust_guard")].iloc[0]
    mon = summary[summary["method"].eq("M_of_N_3_5")].iloc[0]
    lines = [
        "# TrustGuard v5.2 Claim-Safe Ablation",
        "",
        "Scope: candidate-space analysis over the saved VisDrone YOLO feature audit. This is not a new tracker-level result.",
        "",
        f"Legacy balanced: F1={legacy['F1']:.6f}, false_new={legacy['false_new_tracks']:.0f}, occupancy={legacy['false_track_occupancy_frames']:.0f}.",
        f"M-of-N 3/5: F1={mon['F1']:.6f}, false_new={mon['false_new_tracks']:.0f}, occupancy={mon['false_track_occupancy_frames']:.0f}.",
        f"Full TrustGuard: F1={full['F1']:.6f}, false_new={full['false_new_tracks']:.0f}, occupancy={full['false_track_occupancy_frames']:.0f}.",
        "",
        "Interpretation rule: claim an improvement only if TrustGuard reduces false-track occupancy without exceeding the predefined F1 loss budget.",
        "If TrustGuard converges to M-of-N, the article should describe it as quarantine confirmation with veto, not as a stronger numerical T-norm.",
    ]
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_metadata(out: Path, args: argparse.Namespace, cfg: dict[str, Any]) -> None:
    meta = {
        "command": " ".join(["scripts/q1_v5/run_trust_guard_ablation.py", "--config", args.config, "--output-dir", args.output_dir]),
        "git_commit": git(["rev-parse", "HEAD"]),
        "config": cfg,
    }
    (out / "run_metadata.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")


def git(args: list[str]) -> str:
    try:
        return subprocess.check_output(["git", *args], text=True).strip()
    except Exception:
        return "unavailable"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/q1_v5/trust_guard_v52.yaml")
    p.add_argument("--output-dir", default="outputs/results/q1_v5/trust_guard_ablation/bytetrack")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    cfg = read_config(args.config)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "resolved_config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    if args.dry_run:
        print(f"dry_run=ok output={out}")
        return
    det = pd.read_csv(cfg["feature_audit"])
    gt, _ = load_gt_protocol(cfg["dataset_root"], sorted(det["sequence_id"].unique()))
    keys = det[["sequence_id", "frame_id"]].drop_duplicates()
    gt = gt.merge(keys, on=["sequence_id", "frame_id"], how="inner")
    summary, events, disagreement = run_ablation(det, gt, cfg)
    summary.to_csv(out / "trust_guard_ablation_summary.csv", index=False)
    events.to_csv(out / "trust_guard_events.csv", index=False)
    disagreement.to_csv(out / "acceptance_disagreement.csv", index=False)
    write_claim(summary, out / "trust_guard_claim_safe.md")
    write_metadata(out, args, cfg)
    print(f"status=ok output={out / 'trust_guard_ablation_summary.csv'} rows={len(summary)} events={len(events)}")


if __name__ == "__main__":
    main()
