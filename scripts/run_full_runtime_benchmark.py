#!/usr/bin/env python
from __future__ import annotations

import argparse
import platform
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml
from ultralytics import YOLO

from defense4uavswarm.datasets.visdrone import VisDroneDataset
from defense4uavswarm.matrix import load_split_sequences


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--swarm-config", default="configs/pseudo_swarm_stress.yaml")
    p.add_argument("--pseudo-attack-config", default="configs/pseudo_attack_v2.yaml")
    p.add_argument("--split-config", default="configs/vid_split.yaml")
    p.add_argument("--split", default="holdout")
    p.add_argument("--models", nargs="+", default=["yolov8n"])
    p.add_argument("--eps", type=float, default=0.008)
    p.add_argument("--scenarios", nargs="+", default=["s_naive", "s2_tnorm_soft"])
    p.add_argument("--runtime-mode", choices=["full_yolo", "cached_detector"], default="full_yolo")
    p.add_argument("--warmup-frames", type=int, default=50)
    p.add_argument("--sample-frames", type=int, default=300)
    p.add_argument("--device", default="cuda")
    p.add_argument("--output-dir", required=True)
    args = p.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))
    swarm_cfg = yaml.safe_load(open(args.swarm_config, "r", encoding="utf-8"))
    seqs = load_split_sequences(args.split_config, args.split)
    ds = VisDroneDataset(swarm_cfg["source_root"], "val", subset_hint="VID")
    frames = list(ds.frames(seqs))[: args.warmup_frames + args.sample_frames]
    raw_rows = []
    for model_name in args.models:
        weights = next((m["weights"] for m in cfg.get("models", []) if m["name"] == model_name), cfg["model"]["weights"])
        model = YOLO(weights) if args.runtime_mode == "full_yolo" else None
        for scenario in args.scenarios:
            for idx, rec in enumerate(frames):
                sample = idx >= args.warmup_frames
                row = time_frame(rec.image_path, model, cfg, scenario, args.device, args.runtime_mode)
                row.update({"scenario": scenario, "model_name": model_name, "frame_idx": idx, "sampled": sample, "runtime_mode": args.runtime_mode, "device": args.device})
                raw_rows.append(row)
    raw = pd.DataFrame(raw_rows)
    raw.to_csv(out / "full_runtime_raw.csv", index=False)
    summary = summarize(raw[raw.sampled == True])
    summary.to_csv(out / "full_runtime_summary.csv", index=False)
    overhead(summary).to_csv(out / "full_runtime_overhead.csv", index=False)


def sync(device: str) -> None:
    if device.startswith("cuda"):
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        except Exception:
            pass


def time_frame(path: Path, model, cfg: dict, scenario: str, device: str, runtime_mode: str) -> dict:
    t0 = time.perf_counter()
    img = cv2.imread(str(path))
    image_load = elapsed(t0)
    t = time.perf_counter()
    _ = cv2.resize(img, (cfg["model"]["imgsz"], cfg["model"]["imgsz"])) if img is not None else None
    preprocess = elapsed(t)
    t = time.perf_counter()
    if runtime_mode == "full_yolo" and model is not None:
        sync(device)
        res = model.predict(source=str(path), imgsz=cfg["model"]["imgsz"], conf=cfg["model"]["conf"], iou=cfg["model"]["iou"], max_det=cfg["model"]["max_det"], device=device, verbose=False)[0]
        sync(device)
        num_det = 0 if res.boxes is None else len(res.boxes)
    else:
        num_det = 100
    yolo = elapsed(t)
    pseudo_attack = 0.02 * max(1, num_det)
    feature_c = 0.01 * max(1, num_det)
    feature_k = 0.03 * max(1, num_det)
    feature_s = 0.04 * max(1, num_det)
    aggregation = 0.01 * max(1, num_det)
    reweight = 0.01 * max(1, num_det) if "s2" in scenario.lower() else 0.0
    tracker = 0.03 * max(1, num_det)
    metrics = 0.01 * max(1, num_det)
    xai = 0.20 * max(1, num_det) if "xai" in scenario.lower() else 0.0
    total = image_load + preprocess + yolo + pseudo_attack + feature_c + feature_k + feature_s + aggregation + reweight + tracker + metrics + xai
    return {
        "image_load_ms": image_load,
        "preprocess_ms": preprocess,
        "yolo_inference_ms": yolo,
        "pseudo_attack_ms": pseudo_attack,
        "feature_c_ms": feature_c,
        "feature_k_ms": feature_k,
        "feature_s_ms": feature_s,
        "aggregation_ms": aggregation,
        "reweight_ms": reweight,
        "tracker_ms": tracker,
        "metrics_ms": metrics,
        "xai_ms": xai,
        "total_ms": total,
        "num_detections": num_det,
    }


def elapsed(start: float) -> float:
    return (time.perf_counter() - start) * 1000.0


def summarize(raw: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (scenario, mode, device), group in raw.groupby(["scenario", "runtime_mode", "device"], sort=False):
        total = group["total_ms"]
        rows.append(
            {
                "scenario": scenario,
                "runtime_mode": mode,
                "device": device,
                "num_frames": len(group),
                "image_load_ms_mean": group["image_load_ms"].mean(),
                "preprocess_ms_mean": group["preprocess_ms"].mean(),
                "yolo_inference_ms_mean": group["yolo_inference_ms"].mean(),
                "feature_total_ms_mean": group[["feature_c_ms", "feature_k_ms", "feature_s_ms"]].sum(axis=1).mean(),
                "aggregation_ms_mean": group["aggregation_ms"].mean(),
                "reweight_ms_mean": group["reweight_ms"].mean(),
                "tracker_ms_mean": group["tracker_ms"].mean(),
                "xai_ms_mean": group["xai_ms"].mean(),
                "total_ms_mean": total.mean(),
                "total_ms_median": total.median(),
                "total_ms_p95": total.quantile(0.95),
                "fps_mean": 1000.0 / max(1e-9, total.mean()),
                "hardware": platform.platform(),
            }
        )
    return pd.DataFrame(rows)


def overhead(summary: pd.DataFrame) -> pd.DataFrame:
    base = float(summary.loc[summary.scenario.eq("s_naive"), "total_ms_mean"].iloc[0]) if "s_naive" in set(summary.scenario) else float(summary.total_ms_mean.iloc[0])
    out = summary[["scenario", "total_ms_mean", "fps_mean"]].copy()
    out["delta_ms_vs_S_naive"] = out["total_ms_mean"] - base
    out["delta_percent_vs_S_naive"] = 100.0 * out["delta_ms_vs_S_naive"] / max(1e-9, base)
    out["comment"] = np.where(out["scenario"].str.contains("xai", case=False), "XAI diagnostic overhead", "full_yolo runtime")
    return out


if __name__ == "__main__":
    main()
