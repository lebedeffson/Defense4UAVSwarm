#!/usr/bin/env python
from __future__ import annotations

import argparse

from defense4uavswarm.v8_sim import package_v8


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--bundle-path", default="outputs/bundles/Defense4UAVSwarm_v8_custom_uav_swarm_bundle.zip")
    args = p.parse_args()
    package_v8(args.bundle_path)
    print(f"status=ok bundle={args.bundle_path}")


if __name__ == "__main__":
    main()
