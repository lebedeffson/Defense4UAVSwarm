#!/usr/bin/env python
from __future__ import annotations

import argparse

from defense4uavswarm.v9_geomdyn import package_v9


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--bundle-path", default="outputs/bundles/Defense4UAVSwarm_v9_geomdyn_bundle.zip")
    args = p.parse_args()
    package_v9(args.bundle_path)
    print(f"status=ok bundle={args.bundle_path}")


if __name__ == "__main__":
    main()
