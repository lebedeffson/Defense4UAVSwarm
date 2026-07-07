#!/usr/bin/env python
from __future__ import annotations

import argparse

from defense4uavswarm.v8_sim import generate_detections


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--gt-2d", required=True)
    p.add_argument("--scenario", default="combined_stress")
    p.add_argument("--tp-detection-prob", type=float, default=0.85)
    p.add_argument("--bbox-jitter-px", type=float, default=5.0)
    p.add_argument("--fp-rate-per-frame", type=float, default=2.0)
    p.add_argument("--false-burst-prob", type=float, default=0.08)
    p.add_argument("--false-burst-min-duration", type=int, default=1)
    p.add_argument("--false-burst-max-duration", type=int, default=3)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    data = generate_detections(
        manifest_path=args.manifest,
        gt_2d_path=args.gt_2d,
        scenario=args.scenario,
        tp_detection_prob=args.tp_detection_prob,
        bbox_jitter_px=args.bbox_jitter_px,
        fp_rate_per_frame=args.fp_rate_per_frame,
        false_burst_prob=args.false_burst_prob,
        false_burst_min_duration=args.false_burst_min_duration,
        false_burst_max_duration=args.false_burst_max_duration,
        seed=args.seed,
        output=args.output,
    )
    print(f"status=ok detections={len(data['detections'])} output={args.output}")


if __name__ == "__main__":
    main()
