#!/usr/bin/env python
from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from defense4uavswarm.q1_visdrone import Q1Params, adaptive_geometry_acceptance, evaluate_methods, load_model, method_acceptance


def tracker_available(tracker: str) -> tuple[bool, str]:
    if tracker == "bytetrack":
        return True, "available"
    if tracker == "ocsort":
        try:
            from defense4uavswarm.trackers import OCSortAdapter

            OCSortAdapter()
            return True, "boxmot OCSort available"
        except Exception as exc:
            return False, str(exc)
    if tracker == "strongsort":
        try:
            from defense4uavswarm.trackers import StrongSORTAdapter

            StrongSORTAdapter()
            return True, "boxmot StrongSort available"
        except Exception as exc:
            return False, str(exc)
    return True, "not external"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-root", required=True)
    p.add_argument("--detections", required=True)
    p.add_argument("--methods", nargs="+", required=True)
    p.add_argument("--warmup-frames", type=int, default=100)
    p.add_argument("--measure-frames", type=int, default=1000)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--feature-audit", default="outputs/results/q1_final_corrected/yolov8s_main/feature_audit.csv")
    p.add_argument("--selected-configs", default="outputs/results/q1_improvement/yolov8s_sweep/selected_configs.yaml")
    p.add_argument("--output-dir", required=True)
    args = p.parse_args()
    import yaml

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    det = pd.read_csv(args.feature_audit)
    frames = det[["sequence_id", "frame_id"]].drop_duplicates().head(args.warmup_frames + args.measure_frames)
    det = det.merge(frames.iloc[args.warmup_frames:], on=["sequence_id", "frame_id"], how="inner")
    cfgs = yaml.safe_load(Path(args.selected_configs).read_text(encoding="utf-8")) if Path(args.selected_configs).exists() else {}
    rows = []
    for repeat in range(args.repeats):
        for method in args.methods:
            tracker_name = method.replace("_plus_trust", "")
            if tracker_name in {"ocsort", "strongsort"}:
                available, note = tracker_available(tracker_name)
                if not available:
                    rows.append({"method": method, "repeat": repeat, "available": False, "notes": note, "tracker_update_ms": np.nan, "trust_feature_computation_ms": np.nan, "trust_aggregation_ms": np.nan, "total_post_detector_ms": np.nan, "detector_included": False})
                    continue
            start = time.perf_counter()
            tracker_start = time.perf_counter()
            if method.endswith("_plus_trust"):
                base = method.replace("_plus_trust", "")
                method_acceptance(det, "bytetrack" if base in {"bytetrack", "ocsort", "strongsort"} else base, Q1Params())
            else:
                method_acceptance(det, "bytetrack" if method in {"bytetrack", "ocsort", "strongsort"} else method, Q1Params())
            tracker_ms = (time.perf_counter() - tracker_start) * 1000
            feat_start = time.perf_counter()
            _ = det[["c_i", "k_i", "confidence", "temporal_age"]].to_numpy()
            feature_ms = (time.perf_counter() - feat_start) * 1000
            agg_start = time.perf_counter()
            if method.endswith("_plus_trust"):
                adaptive_geometry_acceptance(det, cfgs.get("selected_balanced", {}))
            trust_ms = (time.perf_counter() - agg_start) * 1000
            total = (time.perf_counter() - start) * 1000
            n = max(1, frames.iloc[args.warmup_frames:].shape[0])
            rows.append({"method": method, "repeat": repeat, "available": True, "notes": "", "tracker_update_ms": tracker_ms / n, "trust_feature_computation_ms": feature_ms / n, "trust_aggregation_ms": trust_ms / n, "total_post_detector_ms": total / n, "detector_included": False})
    raw = pd.DataFrame(rows)
    raw.to_csv(out / "runtime_raw.csv", index=False)
    summary = raw.groupby("method", as_index=False).agg({c: ["mean", "std"] for c in ["tracker_update_ms", "trust_feature_computation_ms", "trust_aggregation_ms", "total_post_detector_ms"]})
    summary.columns = ["_".join(c).rstrip("_") for c in summary.columns.to_flat_index()]
    availability = raw.groupby("method", as_index=False).agg(available=("available", "max"), notes=("notes", lambda x: "; ".join(sorted(set(str(v) for v in x if str(v))))[:300]))
    summary = summary.merge(availability, on="method", how="left")
    summary.to_csv(out / "runtime_summary.csv", index=False)
    plt.figure(figsize=(8, 4))
    plt.bar(summary["method"], summary["total_post_detector_ms_mean"])
    plt.xticks(rotation=25, ha="right")
    plt.ylabel("ms/frame, detector excluded")
    plt.tight_layout()
    plt.savefig(out / "fig_runtime_bar.png", dpi=160)
    plt.close()
    (out / "runtime_summary.md").write_text("# Runtime Summary\n\nDetector excluded. Values are post-detector ms/frame.\n\n" + summary.to_string(index=False) + "\n", encoding="utf-8")
    print(f"status=ok output={out / 'runtime_summary.csv'}")


if __name__ == "__main__":
    main()
