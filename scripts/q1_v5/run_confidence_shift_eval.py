#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from defense4uavswarm.q1_v5.map_contamination import contamination_metrics
from defense4uavswarm.q1_visdrone import Q1Params, compute_metrics, load_gt_protocol, method_acceptance
from defense4uavswarm.q1_v5.trust_guard import TrustGuardConfig
from scripts.q1_v5.run_trust_guard_ablation import m_of_n, trust_guard_acceptance


def read_config(path: str | Path) -> dict[str, Any]:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}


def logit_shift(conf: pd.Series, temperature: float, bias: float) -> pd.Series:
    c = conf.astype(float).clip(1e-6, 1.0 - 1e-6)
    z = np.log(c / (1.0 - c))
    shifted = 1.0 / (1.0 + np.exp(-(float(temperature) * z + float(bias))))
    return pd.Series(shifted, index=conf.index).clip(0, 1)


def frame_count(gt: pd.DataFrame) -> int:
    return max(1, gt[["sequence_id", "frame_id"]].drop_duplicates().shape[0])


def score(method: str, det: pd.DataFrame, accepted: pd.Series, gt: pd.DataFrame, temperature: float, bias: float) -> dict[str, Any]:
    row = compute_metrics(det, accepted, len(gt), frame_count(gt), method)
    row.update(contamination_metrics(det, accepted))
    row.update({"temperature": temperature, "bias": bias})
    return row


def run(det: pd.DataFrame, gt: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    temps = cfg.get("shift_grid", {}).get("temperature", [0.75, 1.0, 1.25])
    biases = cfg.get("shift_grid", {}).get("bias", [-0.5, -0.25, 0.0, 0.25, 0.5])
    base_cfg = TrustGuardConfig.from_mapping(cfg.get("trust_guard", {}))
    no_rel_cfg = TrustGuardConfig.from_mapping({**cfg.get("trust_guard", {}), "use_relative_confidence": False})
    conf_threshold = float(cfg.get("legacy_thresholds", {}).get("m_of_n_confidence", 0.10))
    for temperature in temps:
        for bias in biases:
            shifted = det.copy()
            shifted["confidence"] = logit_shift(shifted["confidence"], float(temperature), float(bias))
            rows.append(score("confidence_threshold", shifted, shifted["confidence"] >= conf_threshold, gt, temperature, bias))
            rows.append(score("M_of_N_3_5", shifted, m_of_n(shifted, 3, 5, conf_threshold), gt, temperature, bias))
            rows.append(score("A0_legacy_balanced", shifted, method_acceptance(shifted, "geometry_dynamic_no_multiagent", Q1Params()), gt, temperature, bias))
            no_rel_acc, _ = trust_guard_acceptance(shifted, no_rel_cfg)
            rows.append(score("TrustGuard_no_relative", shifted, no_rel_acc, gt, temperature, bias))
            full_acc, _ = trust_guard_acceptance(shifted, base_cfg)
            rows.append(score("TrustGuard_relative", shifted, full_acc, gt, temperature, bias))
    return pd.DataFrame(rows)


def write_summary(df: pd.DataFrame, output: Path) -> None:
    lines = ["# Confidence Shift Evaluation", ""]
    for method, group in df.groupby("method", sort=True):
        worst_f1 = float(group["F1"].min())
        worst_occ = float(group["false_track_occupancy_frames"].max())
        lines.append(f"- {method}: worst_F1={worst_f1:.6f}, worst_false_occupancy={worst_occ:.0f}")
    lines += [
        "",
        "Claim rule: relative confidence supports the stale-calibration claim only if it reduces worst-case degradation versus fixed thresholds/M-of-N at a comparable F1 operating point.",
    ]
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/q1_v5/trust_guard_v52.yaml")
    p.add_argument("--output-dir", default="outputs/results/q1_v5/confidence_shift/bytetrack")
    args = p.parse_args()
    cfg = read_config(args.config)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    det = pd.read_csv(cfg["feature_audit"])
    gt, _ = load_gt_protocol(cfg["dataset_root"], sorted(det["sequence_id"].unique()))
    keys = det[["sequence_id", "frame_id"]].drop_duplicates()
    gt = gt.merge(keys, on=["sequence_id", "frame_id"], how="inner")
    df = run(det, gt, cfg)
    df.to_csv(out / "confidence_shift_summary.csv", index=False)
    write_summary(df, out / "confidence_shift_claim_safe.md")
    print(f"status=ok output={out / 'confidence_shift_summary.csv'} rows={len(df)}")


if __name__ == "__main__":
    main()
