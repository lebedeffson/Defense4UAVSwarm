#!/usr/bin/env python
from __future__ import annotations

import argparse

from defense4uavswarm.v8_sim import generate_dataset


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output-root", default="data/custom_uav_swarm_v8")
    p.add_argument("--num-scenes", type=int, default=5)
    p.add_argument("--num-agents", type=int, default=3)
    p.add_argument("--frames-per-scene", type=int, default=300)
    p.add_argument("--objects-per-scene", type=int, default=25)
    p.add_argument("--image-width", type=int, default=1280)
    p.add_argument("--image-height", type=int, default=720)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--write-images", action="store_true")
    p.add_argument("--write-gt-2d", action="store_true")
    p.add_argument("--write-gt-3d", action="store_true")
    p.add_argument("--write-poses", action="store_true")
    p.add_argument("--write-calibration", action="store_true")
    args = p.parse_args()
    manifest = generate_dataset(
        output_root=args.output_root,
        num_scenes=args.num_scenes,
        num_agents=args.num_agents,
        frames_per_scene=args.frames_per_scene,
        objects_per_scene=args.objects_per_scene,
        image_width=args.image_width,
        image_height=args.image_height,
        seed=args.seed,
        write_images=args.write_images,
    )
    print(f"status=ok dataset={args.output_root} scenes={manifest['num_scenes']} agents={manifest['num_agents']} frames={len(manifest['frames'])}")


if __name__ == "__main__":
    main()
