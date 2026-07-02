#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from defense4uavswarm.config import load_config
from defense4uavswarm.datasets.visdrone import VisDroneDataset, write_custom_split
from defense4uavswarm.metadata import write_metadata


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--root", default=None)
    p.add_argument("--verify", action="store_true")
    p.add_argument("--custom-split", action="store_true")
    args = p.parse_args()
    cfg = load_config(args.config)
    if args.root:
        cfg["dataset"]["root"] = args.root
    root = Path(cfg["dataset"]["root"])
    extra = {}
    if args.custom_split:
        extra["custom_split"] = write_custom_split(root, cfg["dataset"]["split_file"], cfg["dataset"]["custom_val_ratio"])
    ds = VisDroneDataset(root, "val")
    extra["sequences_val"] = len(ds.sequence_ids())
    extra["annotations_val"] = int(len(ds.all_annotations()))
    if args.verify:
        extra["gt_verify"] = ds.verify()
    write_metadata(Path(cfg["outputs"]["results_dir"]) / "metadata.json", cfg, extra)
    print(extra)


if __name__ == "__main__":
    main()
