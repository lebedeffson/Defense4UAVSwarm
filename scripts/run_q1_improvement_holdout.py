#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml

from defense4uavswarm.q1_visdrone import add_adaptive_metrics, ensure_area_norm, evaluate_methods, load_gt, train_rf


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-root", required=True)
    p.add_argument("--detections", required=True)
    p.add_argument("--feature-audit", default="")
    p.add_argument("--selected-configs", required=True)
    p.add_argument("--methods", nargs="+", required=True)
    p.add_argument("--output-dir", required=True)
    args = p.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    det = ensure_area_norm(pd.read_csv(args.feature_audit))
    gt = load_gt(args.dataset_root, sorted(det["sequence_id"].unique()))
    keys = det[["sequence_id", "frame_id"]].drop_duplicates()
    gt = gt.merge(keys, on=["sequence_id", "frame_id"], how="inner")
    from defense4uavswarm.q1_visdrone import make_chunk_split
    chunks = make_chunk_split(gt)
    det = det.merge(chunks[["sequence_id", "frame_id", "split"]], on=["sequence_id", "frame_id"], how="left")
    gt2 = gt.merge(chunks[["sequence_id", "frame_id", "split"]], on=["sequence_id", "frame_id"], how="left")
    cal_det = det[det["split"].eq("calibration")].copy()
    hold_det = det[det["split"].eq("holdout")].copy()
    hold_gt = gt2[gt2["split"].eq("holdout")].copy()
    cfgs = yaml.safe_load(Path(args.selected_configs).read_text(encoding="utf-8")) or {}
    base_methods = [m for m in args.methods if m not in {"geometry_dynamic_adaptive_balanced", "geometry_dynamic_false_new_safe"}]
    rf_model = None
    if "rf_learned_gate" in base_methods:
        rf_model, tau = train_rf(cal_det)
        from defense4uavswarm.q1_visdrone import Q1Params
        params = Q1Params(rf_threshold=tau)
    else:
        params = None
    summary = evaluate_methods(hold_det, hold_gt, base_methods, params=params, rf_model=rf_model)
    extra = []
    if "geometry_dynamic_adaptive_balanced" in args.methods:
        extra.append(add_adaptive_metrics(hold_det, hold_gt, cfgs["selected_balanced"], "geometry_dynamic_adaptive_balanced"))
    if "geometry_dynamic_false_new_safe" in args.methods:
        extra.append(add_adaptive_metrics(hold_det, hold_gt, cfgs["selected_false_new_safe"], "geometry_dynamic_false_new_safe"))
    if extra:
        summary = pd.concat([summary, pd.DataFrame(extra)], ignore_index=True)
    summary.to_csv(out / "holdout_comparison.csv", index=False)
    write_claim(out / "pareto_claim_summary.md", summary)
    print(f"status=ok output={out / 'holdout_comparison.csv'}")


def write_claim(path: Path, summary: pd.DataFrame) -> None:
    def row(method: str):
        f = summary[summary["method"].eq(method)]
        return f.iloc[0] if len(f) else None
    bt = row("bytetrack")
    base = row("geometry_dynamic_no_multiagent")
    bal = row("geometry_dynamic_adaptive_balanced")
    safe = row("geometry_dynamic_false_new_safe")
    lines = ["# Pareto Claim Summary", ""]
    for name, r in [("ByteTrack", bt), ("Geometry baseline", base), ("Adaptive balanced", bal), ("False-new safe", safe)]:
        if r is not None:
            lines.append(f"- {name}: F1 {r['F1']:.6f}, false_new {int(r['false_new_tracks'])}.")
    if bt is not None and bal is not None:
        lines.append("")
        lines.append(f"Adaptive balanced false_new reduction vs ByteTrack: {(1 - bal['false_new_tracks']/max(1, bt['false_new_tracks']))*100:.2f}%.")
    path.write_text("\\n".join(lines) + "\\n", encoding="utf-8")


if __name__ == "__main__":
    main()
