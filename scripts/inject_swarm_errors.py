#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from defense4uavswarm.datasets.visdrone import VisDroneDataset
from defense4uavswarm.swarm import load_frame_index
from defense4uavswarm.swarm_tnorm import _load_yaml, _pseudo_detections, attack_event_summary, inject_pseudo_attack


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--swarm-root", required=True)
    p.add_argument("--attack-config", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--source-root", default="data/visdrone")
    args = p.parse_args()

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    frame_index = load_frame_index(args.swarm_root)
    sequences = sorted(frame_index["sequence_id"].unique())
    ds = VisDroneDataset(args.source_root, "val", subset_hint="VID")
    gt = ds.all_annotations(sequences)
    detections = _pseudo_detections(gt, frame_index)
    attacked, events = inject_pseudo_attack(detections, _load_yaml(args.attack_config))
    attacked.to_csv(output / "attacked_detections.csv", index=False)
    events.to_csv(output / "attack_event_audit.csv", index=False)
    attack_event_summary(events).to_csv(output / "attack_event_summary.csv", index=False)
    print(f"output: {output}")
    print(f"num_detections_before: {len(detections)}")
    print(f"num_detections_after: {len(attacked)}")
    print(f"num_events: {len(events)}")


if __name__ == "__main__":
    main()
