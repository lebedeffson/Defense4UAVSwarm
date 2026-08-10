#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


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


def blank_or_image(path: Path, width: int = 1280, height: int = 720) -> np.ndarray:
    img = cv2.imread(str(path)) if path.exists() else None
    if img is None:
        return np.zeros((height, width, 3), dtype=np.uint8)
    return img


def run_sequence(seq_dir: Path, dets_by_frame: dict[int, np.ndarray], output_file: Path) -> dict[str, int]:
    from boxmot.trackers.bbox.ocsort.ocsort import OcSort

    frames = sorted(seq_dir.glob("*.jpg"))
    if not frames:
        max_frame = max(dets_by_frame) if dets_by_frame else 0
        frames = [seq_dir / f"{i:07d}.jpg" for i in range(1, max_frame + 1)]
    tracker = OcSort(det_thresh=0.1, min_conf=0.1, max_age=30, min_hits=3, iou_threshold=0.3, per_class=False)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    rows = 0
    with output_file.open("w", encoding="utf-8") as out:
        for image_path in frames:
            frame_id = int(image_path.stem)
            img = blank_or_image(image_path)
            dets = dets_by_frame.get(frame_id, np.empty((0, 6), dtype=np.float32))
            tracks = np.asarray(tracker.update(dets, img))
            if tracks.size == 0:
                continue
            for tr in tracks.reshape(-1, tracks.shape[-1]):
                x1, y1, x2, y2, track_id, conf, cls = tr[:7]
                w = max(0.0, float(x2) - float(x1))
                h = max(0.0, float(y2) - float(y1))
                out.write(f"{frame_id},{int(track_id)},{float(x1):.3f},{float(y1):.3f},{w:.3f},{h:.3f},{float(conf):.6f},{int(cls)},1\n")
                rows += 1
    return {"frames": len(frames), "tracks": rows}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--mot-root", required=True)
    p.add_argument("--dataset-root", required=True)
    p.add_argument("--output-dir", required=True)
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
        stats = run_sequence(seq_dir, dets, out / f"{seq}.txt")
        stats["sequence_id"] = seq
        summary.append(stats)

    import pandas as pd

    pd.DataFrame(summary).to_csv(out / "ocsort_run_summary.csv", index=False)
    (out / "ocsort_environment.md").write_text(
        "# OC-SORT Environment\n\n"
        "- source: `boxmot.trackers.bbox.ocsort.ocsort.OcSort`\n"
        "- reid weights downloaded: no\n"
        "- input: saved YOLO detections exported to MOT format\n",
        encoding="utf-8",
    )
    print(f"status=ok sequences={len(summary)} output={out}")


if __name__ == "__main__":
    main()
