#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import pandas as pd

from defense4uavswarm.datasets.visdrone import VisDroneDataset
from defense4uavswarm.q1_visdrone import load_visdrone_annotations_raw


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-root", required=True)
    p.add_argument("--output-dir", required=True)
    args = p.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    ds = VisDroneDataset(args.dataset_root, "val", subset_hint="VID")
    raw = load_visdrone_annotations_raw(args.dataset_root)
    seq_rows = []
    for seq in ds.sequence_ids():
        frames = list((ds.sequences_dir / seq).glob("*.jpg"))
        ann_path = ds.annotations_dir / f"{seq}.txt"
        width = height = 0
        if frames:
            img = cv2.imread(str(frames[0]))
            if img is not None:
                height, width = img.shape[:2]
        seq_ann = raw[raw["sequence_id"].eq(seq)]
        seq_rows.append(
            {
                "sequence_id": seq,
                "num_frames": len(frames),
                "annotation_file": str(ann_path),
                "annotation_exists": ann_path.exists(),
                "num_annotation_rows": len(seq_ann),
                "num_eval_gt": int((~seq_ann["is_ignored"].astype(bool)).sum()) if len(seq_ann) else 0,
                "num_ignored_boxes": int(seq_ann["is_ignored"].astype(bool).sum()) if len(seq_ann) else 0,
                "image_width": width,
                "image_height": height,
                "first_frame_id": int(Path(frames[0]).stem) if frames else None,
                "last_frame_id": int(Path(frames[-1]).stem) if frames else None,
            }
        )
    seq = pd.DataFrame(seq_rows)
    cls = raw.groupby(["class_id", "class_name", "is_ignored"], dropna=False).size().reset_index(name="count")
    ignored = raw[raw["is_ignored"].astype(bool)].groupby(["class_id", "class_name"], dropna=False).size().reset_index(name="count")
    seq.to_csv(out / "sequence_inventory.csv", index=False)
    cls.to_csv(out / "class_distribution.csv", index=False)
    ignored.to_csv(out / "ignore_region_report.csv", index=False)
    frame_min = int(raw["frame_id"].min()) if len(raw) else 0
    report = [
        "# VisDrone Format Audit",
        "",
        "- bbox input format: `x, y, width, height` from annotation txt.",
        "- bbox converted format: `x1, y1, x2=x+width, y2=y+height`.",
        f"- frame indexing: annotation min frame id is {frame_min}; image names are 7-digit numeric ids.",
        "- ignored region handling: class_id <= 0, class_id >= 11, or score <= 0 is marked ignored for protocol audits.",
        f"- class list: {', '.join(sorted(raw['class_name'].astype(str).unique())) if len(raw) else 'none'}",
        f"- num_sequences: {seq['sequence_id'].nunique()}",
        f"- num_frames: {int(seq['num_frames'].sum())}",
        f"- num_gt_boxes: {int((~raw['is_ignored'].astype(bool)).sum()) if len(raw) else 0}",
        f"- num_ignored_boxes: {int(raw['is_ignored'].astype(bool).sum()) if len(raw) else 0}",
        "",
        "Parser sanity: bbox and frame conventions are explicit in generated CSV files.",
    ]
    (out / "visdrone_format_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"status=ok sequences={len(seq)} frames={int(seq['num_frames'].sum())} output={out}")


if __name__ == "__main__":
    main()
