#!/usr/bin/env python
from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

from defense4uavswarm.datasets.visdrone import VisDroneDataset
from defense4uavswarm.q1_visdrone import write_json


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-root", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--device", default="auto")
    p.add_argument("--conf", type=float, default=0.05)
    p.add_argument("--iou", type=float, default=0.7)
    p.add_argument("--limit-frames", type=int, default=0)
    p.add_argument("--output", required=True)
    p.add_argument("--runtime-output", required=True)
    args = p.parse_args()

    from ultralytics import YOLO

    ds = VisDroneDataset(args.dataset_root, "val", subset_hint="VID")
    model = YOLO(args.model)
    frames_out = []
    runtime_rows = []
    names = getattr(model, "names", {})
    for n, rec in enumerate(ds.frames()):
        if args.limit_frames and n >= args.limit_frames:
            break
        start = time.perf_counter()
        result = model.predict(source=str(rec.image_path), conf=args.conf, iou=args.iou, device=None if args.device == "auto" else args.device, verbose=False)[0]
        latency_ms = (time.perf_counter() - start) * 1000.0
        detections = []
        if result.boxes is not None:
            for j, box in enumerate(result.boxes):
                cls = int(box.cls[0])
                detections.append(
                    {
                        "det_id": f"{rec.sequence_id}_{rec.frame_id}_{j}",
                        "bbox": [float(v) for v in box.xyxy[0].tolist()],
                        "confidence": float(box.conf[0]),
                        "class_id": cls,
                        "class_name": str(names.get(cls, cls)),
                        "detector": args.model,
                    }
                )
        frames_out.append({"sequence_id": rec.sequence_id, "frame_id": rec.frame_id, "image_path": str(rec.image_path), "detections": detections})
        runtime_rows.append({"sequence_id": rec.sequence_id, "frame_id": rec.frame_id, "latency_ms": latency_ms, "num_detections": len(detections)})

    write_json(args.output, {"dataset": "VisDrone2019-VID-val", "detector": args.model, "frames": frames_out})
    out = Path(args.runtime_output)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(runtime_rows).to_csv(out, index=False)
    print(f"status=ok frames={len(frames_out)} output={args.output}")


if __name__ == "__main__":
    main()
