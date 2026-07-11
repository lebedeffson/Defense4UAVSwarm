#!/usr/bin/env python
from __future__ import annotations

import argparse
import gc
import json
import platform
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np
import pandas as pd

from defense4uavswarm.q1_v5.risk_ranker import train_stage_ranker, train_thresholds
from defense4uavswarm.q1_v5.two_stage_quarantine import RiskPrioritizedTwoStageConfig, risk_prioritized_two_stage_gate_result
from scripts.q1_v55.common import load_frozen_inputs, prepare_output, read_config, results_root, write_metadata


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/q1_v55/bytetrack_risk_prioritized.yaml")
    p.add_argument("--output-dir")
    p.add_argument("--repeats", type=int, default=30)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    cfg = read_config(args.config)
    root = results_root(cfg, args.output_dir)
    out = root / "runtime"
    prepare_output(out, overwrite=args.overwrite, dry_run=args.dry_run)
    if args.dry_run:
        print(f"dry_run=ok output={out}")
        return
    ranker_gate = root / "ranker" / "go_no_go.json"
    if ranker_gate.exists() and json.loads(ranker_gate.read_text(encoding="utf-8")).get("gate_b", {}).get("decision") == "STOP_RANKER_NO_GAIN":
        rows = [
            {
                "status": "not_applicable_gate_b_fail",
                "reason": "Online controller integration is blocked by Gate B.",
                "median_ms_per_frame": float("nan"),
                "p95_ms_per_frame": float("nan"),
            }
        ]
        pd.DataFrame(rows).to_csv(out / "runtime_summary.csv", index=False)
        write_metadata(out / "run_metadata.json", {"runtime_status": "not_applicable_gate_b_fail"})
        print(f"status=not_applicable_gate_b_fail output={out}")
        return
    det, _, _, manifest = load_frozen_inputs(cfg)
    episodes = pd.read_parquet(root / "episode_harm" / "episode_harm_table.parquet")
    r1 = train_stage_ranker(episodes, stage=1)
    r2 = train_stage_ranker(episodes, stage=2)
    p1 = r1.predict(episodes, horizon=1).rename(columns={"risk_score": "stage1_risk_score"})
    p2 = r2.predict(episodes[episodes["second_frame_id"].notna()].copy(), horizon=10).rename(columns={"risk_score": "stage2_risk_score"})
    thresholds = train_thresholds(episodes, p1.rename(columns={"stage1_risk_score": "risk_score"}), p2.rename(columns={"stage2_risk_score": "risk_score"}))
    scores = p1.merge(p2, on=["sequence_id", "tracklet_id", "episode_id"], how="left")
    scores["stage2_risk_score"] = scores["stage2_risk_score"].fillna(0.0)
    qcfg = RiskPrioritizedTwoStageConfig(stage1_fraction_max=0.05, stage2_fraction_max=0.03, stage2_horizon_frames=5)
    risk_prioritized_two_stage_gate_result(det, manifest, scores, tau_stage1=thresholds["tau_stage1"], tau_stage2=thresholds["tau_stage2"], cfg=qcfg)
    gc_enabled = gc.isenabled()
    gc.disable()
    times = []
    try:
        for _ in range(int(args.repeats)):
            t0 = time.perf_counter_ns()
            risk_prioritized_two_stage_gate_result(det, manifest, scores, tau_stage1=thresholds["tau_stage1"], tau_stage2=thresholds["tau_stage2"], cfg=qcfg)
            times.append(time.perf_counter_ns() - t0)
    finally:
        if gc_enabled:
            gc.enable()
    arr = np.asarray(times, dtype=float) / 1e6
    frames = int(len(manifest.drop_duplicates(["sequence_id", "frame_id"])))
    rows = [
        {
            "diagnostic_logging": False,
            "runs": int(args.repeats),
            "frames": frames,
            "mean_ms_per_frame": float(arr.mean() / frames),
            "median_ms_per_frame": float(np.median(arr) / frames),
            "p90_ms_per_frame": float(np.quantile(arr, 0.90) / frames),
            "p95_ms_per_frame": float(np.quantile(arr, 0.95) / frames),
            "max_ms_per_frame": float(arr.max() / frames),
            "microseconds_per_episode_start": float(arr.mean() * 1000.0 / max(1, len(episodes))),
            "python": platform.python_version(),
        }
    ]
    pd.DataFrame(rows).to_csv(out / "runtime_summary.csv", index=False)
    write_metadata(out / "run_metadata.json", {"runtime_rows": len(rows)})
    print(f"status=ok median_ms_per_frame={rows[0]['median_ms_per_frame']:.6f} output={out}")


if __name__ == "__main__":
    main()
