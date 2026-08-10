#!/usr/bin/env python
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from defense4uavswarm.trackers import SortAdapter


def load_det_file(path: Path) -> dict[int, np.ndarray]:
    by_frame: dict[int, list[list[float]]] = {}
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            parts = [x.strip() for x in line.split(",")]
            if len(parts) < 8:
                continue
            frame = int(float(parts[0]))
            x, y, w, h = [float(v) for v in parts[2:6]]
            conf = float(parts[6])
            cls = float(parts[7])
            by_frame.setdefault(frame, []).append([x, y, x + w, y + h, conf, cls])
    return {k: np.asarray(v, dtype=np.float32) for k, v in by_frame.items()}


def sequence_frame_ids(seq_dir: Path, dets_by_frame: dict[int, np.ndarray]) -> list[int]:
    frames = sorted(seq_dir.glob("*.jpg"))
    if frames:
        return [int(path.stem) for path in frames]
    max_frame = max(dets_by_frame) if dets_by_frame else 0
    return list(range(1, max_frame + 1))


def run_sequence(seq_dir: Path, dets_by_frame: dict[int, np.ndarray], output_file: Path, args: argparse.Namespace) -> dict[str, float | int | str]:
    tracker = SortAdapter(max_age=args.max_age, min_hits=args.min_hits, iou_threshold=args.iou_threshold)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    rows = 0
    update_times_ns = []
    frame_ids = sequence_frame_ids(seq_dir, dets_by_frame)
    with output_file.open("w", encoding="utf-8") as out:
        for frame_id in frame_ids:
            dets = dets_by_frame.get(frame_id, np.empty((0, 6), dtype=np.float32))
            start = time.perf_counter_ns()
            tracks = np.asarray(tracker.update(dets, frame_id))
            update_times_ns.append(time.perf_counter_ns() - start)
            if tracks.size == 0:
                continue
            for tr in tracks.reshape(-1, tracks.shape[-1]):
                x1, y1, x2, y2, track_id, conf, cls = tr[:7]
                w = max(0.0, float(x2) - float(x1))
                h = max(0.0, float(y2) - float(y1))
                out.write(f"{frame_id},{int(track_id)},{float(x1):.3f},{float(y1):.3f},{w:.3f},{h:.3f},{float(conf):.6f},{int(cls)},1\n")
                rows += 1
    mean_ms = float(np.mean(update_times_ns) / 1e6) if update_times_ns else 0.0
    p95_ms = float(np.percentile(update_times_ns, 95) / 1e6) if update_times_ns else 0.0
    return {"sequence_id": seq_dir.name, "frames": len(frame_ids), "tracks": rows, "mean_update_ms": mean_ms, "p95_update_ms": p95_ms}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--mot-root", required=True)
    p.add_argument("--dataset-root", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--max-age", type=int, default=30)
    p.add_argument("--min-hits", type=int, default=3)
    p.add_argument("--iou-threshold", type=float, default=0.3)
    args = p.parse_args()
    mot_root = Path(args.mot_root)
    dataset_root = Path(args.dataset_root)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    summary = []
    for seq_mot in sorted(x for x in mot_root.iterdir() if x.is_dir()):
        seq = seq_mot.name
        dets = load_det_file(seq_mot / "det" / "det.txt")
        seq_dir = dataset_root / "sequences" / seq
        stats = run_sequence(seq_dir, dets, out / f"{seq}.txt", args)
        stats["sequence_id"] = seq
        summary.append(stats)
    pd.DataFrame(summary).to_csv(out / "sort_run_summary.csv", index=False)
    (out / "sort_environment.md").write_text(
        "# SORT Environment\n\n"
        "- source: `defense4uavswarm.trackers.SortAdapter`\n"
        "- association: Hungarian assignment over IoU\n"
        "- motion model: constant-velocity Kalman box filter\n"
        "- ReID weights downloaded: no\n"
        "- input: saved YOLO detections exported to MOT format\n",
        encoding="utf-8",
    )
    print(f"status=ok sequences={len(summary)} output={out}")


if __name__ == "__main__":
    main()
