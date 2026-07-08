#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from defense4uavswarm.q1_visdrone import load_detections


def write_seqinfo(path: Path, seq: str, seq_dir: Path, seq_length: int, width: int, height: int) -> None:
    path.write_text(
        "\n".join(
            [
                "[Sequence]",
                f"name={seq}",
                f"imDir={seq_dir}",
                "frameRate=30",
                f"seqLength={seq_length}",
                f"imWidth={width}",
                f"imHeight={height}",
                "imExt=.jpg",
                "",
            ]
        ),
        encoding="utf-8",
    )


def image_size(seq_dir: Path) -> tuple[int, int]:
    first = next(iter(sorted(seq_dir.glob("*.jpg"))), None)
    if first is None:
        return 0, 0
    img = cv2.imread(str(first))
    if img is None:
        return 0, 0
    h, w = img.shape[:2]
    return int(w), int(h)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-root", required=True)
    p.add_argument("--detections", required=True)
    p.add_argument("--detector", required=True)
    p.add_argument("--conf-threshold", type=float, default=0.1)
    p.add_argument("--output-dir", required=True)
    args = p.parse_args()

    dataset_root = Path(args.dataset_root)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    det = load_detections(args.detections)
    det = det[det["confidence"].astype(float) >= float(args.conf_threshold)].copy()

    rows = []
    for seq, group in det.groupby("sequence_id", sort=True):
        seq_dir = dataset_root / "sequences" / str(seq)
        seq_out = out / str(seq)
        det_dir = seq_out / "det"
        det_dir.mkdir(parents=True, exist_ok=True)
        width, height = image_size(seq_dir)
        frames = sorted(seq_dir.glob("*.jpg"))
        write_seqinfo(seq_out / "seqinfo.ini", str(seq), seq_dir, len(frames), width, height)
        with (det_dir / "det.txt").open("w", encoding="utf-8") as f:
            for row in group.sort_values(["frame_id", "confidence"], ascending=[True, False]).itertuples():
                x = float(row.x1)
                y = float(row.y1)
                w = max(0.0, float(row.x2) - float(row.x1))
                h = max(0.0, float(row.y2) - float(row.y1))
                cls = int(row.class_id) if str(row.class_id).lstrip("-").isdigit() else -1
                f.write(f"{int(row.frame_id)},-1,{x:.3f},{y:.3f},{w:.3f},{h:.3f},{float(row.confidence):.6f},{cls},1\n")
        rows.append({"sequence_id": seq, "frames": len(frames), "detections": int(len(group)), "width": width, "height": height})

    import pandas as pd

    pd.DataFrame(rows).to_csv(out / "mot_export_summary.csv", index=False)
    print(f"status=ok sequences={len(rows)} output={out}")


if __name__ == "__main__":
    main()
