#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from defense4uavswarm.q1_v5.map_contamination import contamination_metrics
from defense4uavswarm.q1_v5.pareto import dominance_matrix, pareto_front
from defense4uavswarm.q1_v5.trust_layer import TrustConfig, TrustFeatures, decide, temporal_from_age
from defense4uavswarm.q1_visdrone import Q1Params, compute_metrics, load_gt_protocol, method_acceptance


def read_config(path: str | Path) -> dict[str, Any]:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}


def cfg_hash(cfg: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:12]


def frame_count(gt: pd.DataFrame) -> int:
    return max(1, gt[["sequence_id", "frame_id"]].drop_duplicates().shape[0])


def true_new_tracks(acc: pd.DataFrame) -> int:
    tp = acc[acc["eval_is_tp"].astype(bool)].copy()
    if tp.empty:
        return 0
    return int(tp.sort_values("frame_id").groupby("tracklet_id", sort=False).head(1).shape[0])


def confirmation_delay(acc: pd.DataFrame) -> tuple[float, float]:
    tp = acc[acc["eval_is_tp"].astype(bool)].copy()
    if tp.empty or "temporal_age" not in tp:
        return 0.0, 0.0
    first = tp.sort_values("frame_id").groupby("matched_gt_id", sort=False).head(1)
    return float(first["temporal_age"].mean()), float(first["temporal_age"].median())


def enrich(row: dict[str, Any], det: pd.DataFrame, accepted: pd.Series, gt: pd.DataFrame) -> dict[str, Any]:
    metrics = compute_metrics(det, accepted, len(gt), frame_count(gt), row["method"])
    acc = det[pd.Series(accepted, index=det.index).astype(bool)]
    contam = contamination_metrics(det, accepted)
    tn = true_new_tracks(acc)
    mean_delay, med_delay = confirmation_delay(acc)
    metrics.update(contam)
    metrics.update(
        {
            "true_new_tracks": tn,
            "track_initiation_precision": tn / max(1, tn + metrics["false_new_tracks"]),
            "track_initiation_recall": tn / max(1, det[det["eval_is_tp"].astype(bool)]["matched_gt_id"].nunique()),
            "mean_confirmation_delay": mean_delay,
            "median_confirmation_delay": med_delay,
        }
    )
    row.update(metrics)
    return row


def confidence_threshold(det: pd.DataFrame, threshold: float) -> pd.Series:
    return det["confidence"].astype(float) >= threshold


def m_of_n(det: pd.DataFrame, m: int, n: int, threshold: float) -> pd.Series:
    out = np.zeros(len(det), dtype=bool)
    for _, group in det.sort_values(["sequence_id", "tracklet_id", "frame_id"]).groupby("tracklet_id", sort=False):
        hits = (group["confidence"].astype(float) >= threshold).astype(int)
        counts = hits.rolling(window=n, min_periods=1).sum().to_numpy()
        out[det.index.get_indexer(group.index)] = counts >= m
    return pd.Series(out, index=det.index)


def bayesian_fixed(det: pd.DataFrame, threshold: float) -> pd.Series:
    return pd.Series(bayesian_score(det, 0.90, 0.35, -0.25) >= threshold, index=det.index)


def bayesian_score(det: pd.DataFrame, high: float, mid: float, neg: float) -> np.ndarray:
    out = np.zeros(len(det), dtype=float)
    for _, group in det.sort_values(["sequence_id", "tracklet_id", "frame_id"]).groupby("tracklet_id", sort=False):
        conf = group["confidence"].to_numpy(float)
        increments = np.where(conf >= 0.50, high, np.where(conf >= 0.30, mid, neg))
        out[det.index.get_indexer(group.index)] = np.cumsum(increments)
    return out


def bayesian(det: pd.DataFrame, high: float, mid: float, neg: float, threshold: float) -> pd.Series:
    out = np.zeros(len(det), dtype=bool)
    for _, group in det.sort_values(["sequence_id", "tracklet_id", "frame_id"]).groupby("tracklet_id", sort=False):
        conf = group["confidence"].astype(float)
        increments = np.where(conf >= 0.50, high, np.where(conf >= 0.30, mid, neg))
        out[group.index.to_numpy(dtype=int)] = np.cumsum(increments) >= threshold
    return pd.Series(out, index=det.index)


def trust_acceptance(det: pd.DataFrame, cfg: dict[str, Any], threshold: float, mode: str) -> pd.Series:
    tc = TrustConfig(
        mode=cfg["mode"],
        active_channels=tuple(cfg["active_channels"]),
        beta=float(cfg["beta"]),
        acceptance_threshold=float(threshold),
        temporal_age_scale=float(cfg["temporal_age_scale"]),
        temporal_floor=float(cfg.get("temporal_floor", 0.0)),
        recovery_min_age=cfg.get("recovery_min_age"),
        recovery_min_confidence=cfg.get("recovery_min_confidence"),
    )
    temporal = temporal_from_age(det["temporal_age"].to_numpy(float), tc.temporal_age_scale, tc.temporal_floor)
    features = TrustFeatures(
        confidence=det["confidence"].to_numpy(float),
        kinematic=det.get("k_i", pd.Series(1.0, index=det.index)).to_numpy(float),
        temporal=temporal,
        geometric=det.get("s_i", pd.Series(1.0, index=det.index)).to_numpy(float),
    )
    return pd.Series(decide(features, tc, temporal_age=det["temporal_age"].to_numpy(float)).accepted, index=det.index)


def run_points(det: pd.DataFrame, gt: pd.DataFrame, cfg: dict[str, Any], config_hash: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    tracker = cfg.get("tracker", "unknown")
    detector = cfg.get("detector", "unknown")
    tracker_acc = method_acceptance(det, "bytetrack", Q1Params())
    rows.append(enrich({"method": "tracker_baseline", "tracker": tracker, "detector": detector, "parameter": "legacy_bytetrack", "parameter_value": 0, "config_hash": config_hash}, det, tracker_acc, gt))
    legacy_trust = method_acceptance(det, "geometry_dynamic_no_multiagent", Q1Params())
    rows.append(enrich({"method": "legacy_geometry_dynamic_no_multiagent", "tracker": tracker, "detector": detector, "parameter": "legacy_q1params", "parameter_value": 0, "config_hash": config_hash}, det, legacy_trust, gt))
    for th in np.round(np.arange(0.05, 0.91, 0.01), 2):
        rows.append(enrich({"method": "confidence_threshold", "tracker": tracker, "detector": detector, "parameter": "threshold", "parameter_value": th, "config_hash": config_hash}, det, confidence_threshold(det, float(th)), gt))
    for th in np.round(np.arange(0.20, 0.91, 0.01), 2):
        rows.append(enrich({"method": "trust_strict", "tracker": tracker, "detector": detector, "parameter": "acceptance_threshold", "parameter_value": th, "config_hash": config_hash}, det, trust_acceptance(det, cfg["trust"]["strict"], float(th), "strict"), gt))
        rows.append(enrich({"method": "trust_balanced", "tracker": tracker, "detector": detector, "parameter": "acceptance_threshold", "parameter_value": th, "config_hash": config_hash}, det, trust_acceptance(det, cfg["trust"]["balanced"], float(th), "balanced"), gt))
    # Representative M-of-N grid. The exhaustive grid is expensive on full VisDrone;
    # this staged run keeps the standard baseline in the operating-curve table
    # without making the reproducibility command impractical.
    for m, n in [(1, 2), (2, 3), (3, 5)]:
        for _n in [n]:
            if m > n:
                continue
            for th in [0.10, 0.30, 0.50]:
                rows.append(enrich({"method": "m_of_n_confirmation", "tracker": tracker, "detector": detector, "parameter": f"M={m};N={n};conf", "parameter_value": th, "M": m, "N": n, "config_hash": config_hash}, det, m_of_n(det, m, n, th), gt))
    bayes_scores = bayesian_score(det, 0.90, 0.35, -0.25)
    for th in np.round(np.arange(0.0, 3.01, 0.05), 2):
        rows.append(enrich({"method": "bayesian_fixed", "tracker": tracker, "detector": detector, "parameter": "logodds_threshold", "parameter_value": th, "config_hash": config_hash}, det, pd.Series(bayes_scores >= float(th), index=det.index), gt))
    return pd.DataFrame(rows)


def by_sequence(points: pd.DataFrame, det: pd.DataFrame, gt: pd.DataFrame, cfg: dict[str, Any], config_hash: str) -> pd.DataFrame:
    rows = []
    # Keep by-sequence output compact: key selected legacy-equivalent points plus baseline.
    selected = points.sort_values(["method", "parameter_value"]).groupby("method", sort=False).head(1)[["method", "parameter", "parameter_value"]]
    for seq in sorted(det["sequence_id"].unique()):
        sub_det = det[det["sequence_id"].eq(seq)]
        sub_gt = gt[gt["sequence_id"].eq(seq)]
        for r in selected.itertuples():
            method = str(r.method)
            val = float(r.parameter_value)
            if method == "tracker_baseline":
                accepted = method_acceptance(sub_det, "bytetrack", Q1Params())
            elif method == "legacy_geometry_dynamic_no_multiagent":
                accepted = method_acceptance(sub_det, "geometry_dynamic_no_multiagent", Q1Params())
            elif method == "confidence_threshold":
                accepted = confidence_threshold(sub_det, val)
            elif method == "trust_strict":
                accepted = trust_acceptance(sub_det, cfg["trust"]["strict"], val, "strict")
            elif method == "trust_balanced":
                accepted = trust_acceptance(sub_det, cfg["trust"]["balanced"], val, "balanced")
            elif method == "bayesian_fixed":
                accepted = bayesian_fixed(sub_det, val)
            else:
                continue
            row = enrich({"method": method, "sequence_id": seq, "parameter": r.parameter, "parameter_value": val, "config_hash": config_hash}, sub_det, accepted, sub_gt)
            rows.append(row)
    return pd.DataFrame(rows)


def plot_curve(df: pd.DataFrame, output: Path, metric: str = "F1") -> None:
    plt.figure(figsize=(7, 5), dpi=180)
    for method, group in df.groupby("method", sort=True):
        if method == "tracker_baseline":
            plt.scatter(group["false_new_tracks"], group[metric], label=method, s=55)
        else:
            g = group.sort_values("false_new_tracks")
            plt.plot(g["false_new_tracks"], g[metric], label=method, linewidth=1.6)
    plt.xlabel("False new tracks")
    plt.ylabel(metric)
    plt.grid(alpha=0.25)
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(output)
    plt.close()
    plt.figure(figsize=(7, 5), dpi=180)
    for method, group in df.groupby("method", sort=True):
        g = group.sort_values("false_new_tracks")
        plt.plot(g["false_new_tracks"], g["recall"], label=method, linewidth=1.6)
    plt.xlabel("False new tracks")
    plt.ylabel("Recall")
    plt.grid(alpha=0.25)
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(output.with_name(output.name.replace("f1", "recall")))
    plt.close()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    cfg = read_config(args.config)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "resolved_config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    if args.dry_run:
        print(f"dry_run=ok output={out}")
        return
    ch = cfg_hash(cfg)
    det = pd.read_csv(cfg["feature_audit"])
    gt, _ = load_gt_protocol(cfg["dataset_root"], sorted(det["sequence_id"].unique()))
    keys = det[["sequence_id", "frame_id"]].drop_duplicates()
    gt = gt.merge(keys, on=["sequence_id", "frame_id"], how="inner")
    points = run_points(det, gt, cfg, ch)
    points["is_pareto_f1_false_new"] = pareto_front(points[["F1", "false_new_tracks"]].assign(method=points["method"]), ["F1"], ["false_new_tracks"])
    points.to_csv(out / "operating_points_raw.csv", index=False)
    points[points["is_pareto_f1_false_new"].astype(bool)].to_csv(out / "pareto_front.csv", index=False)
    dominance_matrix(points).to_csv(out / "dominance_matrix.csv", index=False)
    by_sequence(points, det, gt, cfg, ch).to_csv(out / "operating_points_by_sequence.csv", index=False)
    plot_curve(points, out / f"fig_f1_vs_false_new_{cfg.get('tracker','tracker')}.png")
    (out / "operating_curves_claim_safe.md").write_text(write_claim(points), encoding="utf-8")
    write_metadata(out, args, cfg, ch)
    print(f"status=ok output={out / 'operating_points_raw.csv'} rows={len(points)}")


def write_claim(points: pd.DataFrame) -> str:
    trust = points[points["method"].str.startswith("trust")]
    bayes = points[points["method"].eq("bayesian_fixed")]
    lines = ["# Operating Curves Claim-Safe Summary", ""]
    if not trust.empty:
        best = trust.sort_values(["false_new_tracks", "F1"], ascending=[True, False]).iloc[0]
        lines.append(f"Best low-false-new trust point: method={best['method']} F1={best['F1']:.6f}, false_new={best['false_new_tracks']:.0f}.")
    if not bayes.empty and not trust.empty:
        bayes_dom = dominance_matrix(points[points["method"].isin(["bayesian_fixed", "trust_strict", "trust_balanced"])])
        dom = bayes_dom[(bayes_dom["method_a"].eq("bayesian_fixed")) & (bayes_dom["method_b"].str.startswith("trust")) & (bayes_dom["a_dominates_b"].astype(bool))]
        lines.append(f"Bayesian fixed dominates trust curves: {'yes' if len(dom) else 'no/partial'}.")
    lines += ["", "Do not claim universal superiority. Use matched-point and non-inferiority tables for article claims."]
    return "\n".join(lines) + "\n"


def write_metadata(out: Path, args: argparse.Namespace, cfg: dict[str, Any], ch: str) -> None:
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        commit = "unavailable"
    meta = {"git_commit": commit, "config_hash": ch, "command": " ".join(["run_operating_curves.py", "--config", args.config, "--output-dir", args.output_dir]), "status": "success"}
    (out / "run_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
