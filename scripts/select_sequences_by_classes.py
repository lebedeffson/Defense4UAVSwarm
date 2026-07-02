#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from defense4uavswarm.class_groups import load_class_groups
from defense4uavswarm.config import load_config
from defense4uavswarm.matrix import task_dataset


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--root", default=None)
    p.add_argument("--task", default="vid", choices=["vid", "mot"])
    p.add_argument("--out", default="outputs/results/sequence_class_distribution.csv")
    args = p.parse_args()

    cfg = load_config(args.config)
    if args.root:
        cfg["dataset"]["root"] = args.root
    _, ds = task_dataset(cfg, args.task)
    groups_cfg = load_class_groups(cfg["class_groups_file"])
    group_by_class = {
        cls: group
        for group, names in groups_cfg["class_groups"].items()
        if group != "all"
        for cls in names
    }
    rows = []
    for seq in ds.sequence_ids():
        ann = ds.annotations(seq)
        counts = ann["class_name"].value_counts().to_dict()
        row = {
            "sequence_id": seq,
            "num_frames": sum(1 for _ in ds.frames([seq])),
            "num_gt": int(len(ann)),
            "vru_gt": int(sum(n for c, n in counts.items() if group_by_class.get(c) == "vru")),
            "vehicles_gt": int(sum(n for c, n in counts.items() if group_by_class.get(c) == "vehicles")),
            "classes": ",".join(sorted(counts)),
        }
        for cls, n in counts.items():
            row[f"class_{cls}"] = int(n)
        rows.append(row)
    df = pd.DataFrame(rows).fillna(0).sort_values(["vehicles_gt", "vru_gt"], ascending=False)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(df[["sequence_id", "num_frames", "num_gt", "vru_gt", "vehicles_gt", "classes"]].to_string(index=False))
    print(out)


if __name__ == "__main__":
    main()
