#!/usr/bin/env python
from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from statistics import mean, median
from typing import Any

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from defense4uavswarm.q1_v5.frame_manifest import add_gt_presence, build_visdrone_frame_manifest, merge_manifest_dimensions
from defense4uavswarm.q1_v5.initiation_gate import GateConfig, assign_episode_ids, split_duplicate_observation_tracklets
from defense4uavswarm.q1_v5.selective_quarantine import SelectiveTrustQuarantineConfig, selective_quarantine_gate_result
from defense4uavswarm.q1_visdrone import load_gt_protocol
from scripts.q1_v54.run_corrected_operating_curves import load_candidates


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--results-root", default="outputs/results/q1_selective_quarantine_v21")
    p.add_argument("--repeats", type=int, default=30)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()
    out = Path(args.results_root) / "runtime"
    if out.exists() and any(out.iterdir()) and not args.overwrite:
        raise SystemExit(f"Output exists; use --overwrite: {out}")
    out.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for cfg_path in [Path("configs/q1_v54/bytetrack.yaml"), Path("configs/q1_v54/ocsort.yaml")]:
        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
        tracker = str(cfg.get("tracker", cfg_path.stem))
        det, manifest, qcfg = load_runtime_inputs(cfg)
        rows.append(bench_tracker(tracker, det, manifest, qcfg, args.repeats))
    pd.DataFrame(rows).to_csv(out / "runtime_summary.csv", index=False)
    (out / "runtime_environment.json").write_text(json.dumps(environment(), indent=2), encoding="utf-8")
    (out / "runtime_claim_safe.md").write_text(write_runtime_report(pd.DataFrame(rows)), encoding="utf-8")
    (out / "run_metadata.json").write_text(json.dumps({"status": "success", "git_commit": git(["rev-parse", "HEAD"]), "branch": git(["branch", "--show-current"])}, indent=2), encoding="utf-8")
    print(f"status=ok output={out / 'runtime_summary.csv'} rows={len(rows)}")


def load_runtime_inputs(cfg: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, SelectiveTrustQuarantineConfig]:
    det = load_candidates(cfg)
    gt, _ = load_gt_protocol(cfg["dataset_root"], sorted(det["sequence_id"].unique()))
    manifest = add_gt_presence(build_visdrone_frame_manifest(cfg["dataset_root"], sorted(det["sequence_id"].unique())), gt)
    det = merge_manifest_dimensions(det, manifest)
    det = split_duplicate_observation_tracklets(det)
    det = assign_episode_ids(det, int(cfg.get("gate", {}).get("max_track_gap", 1)))
    specs = cfg.get("grids", {}).get("selective_quarantine", [])
    qcfg = SelectiveTrustQuarantineConfig.from_mapping(specs[0] if specs else {})
    return det, manifest, qcfg


def bench_tracker(tracker: str, det: pd.DataFrame, manifest: pd.DataFrame, qcfg: SelectiveTrustQuarantineConfig, repeats: int) -> dict[str, Any]:
    frame_count = int(len(manifest.drop_duplicates(["sequence_id", "frame_id"])))
    candidate_count = int(len(det))
    frame_groups = list(det.groupby(["sequence_id", "frame_id"], sort=False))
    baseline_times: list[int] = []
    gate_times: list[int] = []
    baseline_loop(frame_groups)
    selective_quarantine_gate_result(det, manifest, qcfg)
    gc_enabled = gc.isenabled()
    gc.disable()
    try:
        for _ in range(int(repeats)):
            t0 = time.perf_counter_ns()
            baseline_loop(frame_groups)
            baseline_times.append(time.perf_counter_ns() - t0)
            t0 = time.perf_counter_ns()
            selective_quarantine_gate_result(det, manifest, qcfg)
            gate_times.append(time.perf_counter_ns() - t0)
    finally:
        if gc_enabled:
            gc.enable()
    gate_ms = np.asarray(gate_times, dtype=float) / 1e6
    base_ms = np.asarray(baseline_times, dtype=float) / 1e6
    overhead_ms = gate_ms - base_ms
    return {
        "tracker": tracker,
        "repeats": int(repeats),
        "frames": frame_count,
        "candidates": candidate_count,
        "baseline_ms_per_frame_mean": float(base_ms.mean() / max(1, frame_count)),
        "gate_ms_per_frame_mean": float(gate_ms.mean() / max(1, frame_count)),
        "online_overhead_ms_per_frame_mean": float(overhead_ms.mean() / max(1, frame_count)),
        "online_overhead_ms_per_frame_median": float(np.median(overhead_ms) / max(1, frame_count)),
        "online_overhead_ms_per_frame_p95": float(np.quantile(overhead_ms, 0.95) / max(1, frame_count)),
        "microseconds_per_candidate_mean": float(gate_ms.mean() * 1000.0 / max(1, candidate_count)),
        "microseconds_per_candidate_median": float(np.median(gate_ms) * 1000.0 / max(1, candidate_count)),
        "microseconds_per_candidate_p95": float(np.quantile(gate_ms, 0.95) * 1000.0 / max(1, candidate_count)),
    }


def baseline_loop(frame_groups: list[tuple[tuple[str, int], pd.DataFrame]]) -> int:
    total = 0
    for _, frame in frame_groups:
        total += len(frame)
    return total


def environment() -> dict[str, Any]:
    return {
        "CPU": platform.processor() or platform.machine(),
        "RAM": _meminfo(),
        "OS": platform.platform(),
        "Python": platform.python_version(),
        "NumPy": np.__version__,
        "pandas": pd.__version__,
        "thread_count": os.cpu_count(),
        "commit": git(["rev-parse", "HEAD"]),
    }


def _meminfo() -> str:
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("MemTotal:"):
                return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return "unavailable"


def write_runtime_report(df: pd.DataFrame) -> str:
    return "# Selective v2.1 Runtime\n\nGate-only timing excludes YOLO inference and offline GT matching.\n\n" + df.to_string(index=False) + "\n"


def git(args: list[str]) -> str:
    try:
        return subprocess.check_output(["git", *args], text=True).strip()
    except Exception:
        return "unavailable"


if __name__ == "__main__":
    main()
