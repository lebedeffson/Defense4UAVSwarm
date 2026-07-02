#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from defense4uavswarm.datasets.visdrone import VISDRONE_CLASSES, VisDroneDataset, discover_visdrone_subsets


def inspect(root: Path, subset: str | None) -> None:
    candidates = discover_visdrone_subsets(root)
    if subset:
        candidates = [c for c in candidates if subset.lower() in c["subset"].lower()]
    if not candidates:
        print(f"ERROR: no VisDrone subset with sequences/annotations under {root}")
        raise SystemExit(1)
    cand = candidates[0]
    ds = VisDroneDataset(root, "val", subset_hint=cand["subset"] if cand["subset"] != "UNKNOWN" else None)
    seqs = ds.sequence_ids()
    ann_files = sorted(ds.annotations_dir.glob("*.txt"))
    first_seq = seqs[0] if seqs else None
    first_frame = next(ds.frames([first_seq])).image_path if first_seq else None
    first_ann = ds.annotations_dir / f"{first_seq}.txt" if first_seq else None
    ann = ds.annotations(first_seq) if first_seq else None
    classes = sorted(set(int(x) for x in ann["class_id"].dropna().unique())) if ann is not None and len(ann) else []
    has_track_id = bool(ann is not None and len(ann) and (ann["gt_track_id"] >= 0).any())
    num_frames = sum(1 for _ in ds.frames())

    print(f"dataset_subset: {cand['dataset_subset']}")
    print(f"dataset_path: {ds.split_root}")
    print(f"num_sequences: {len(seqs)}")
    print(f"num_frames: {num_frames}")
    print(f"num_annotation_files: {len(ann_files)}")
    print(f"first_sequence_name: {first_seq}")
    print(f"first_frame_path: {first_frame}")
    print(f"first_annotation_path: {first_ann}")
    print(f"has_track_id: {str(has_track_id).lower()}")
    print("classes_detected: " + ", ".join(f"{i}:{VISDRONE_CLASSES.get(i, 'unknown')}" for i in classes))
    if not has_track_id:
        print("WARNING: has_track_id=false; do not compute MOTA/IDF1/IDSW/track ASR for this subset.")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="data/visdrone")
    p.add_argument("--subset", choices=["VID", "MOT"], default=None)
    args = p.parse_args()
    inspect(Path(args.root), args.subset)


if __name__ == "__main__":
    main()
