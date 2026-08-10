#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

from defense4uavswarm.datasets.u2u_adapter import U2UDataAdapter


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-root", default="data/U2UData")
    p.add_argument("--backend", default="auto")
    p.add_argument("--output", required=True)
    args = p.parse_args()
    report = U2UDataAdapter(args.dataset_root).report()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"status={report['status']} scenes={report['num_scenes']} rgb={report['has_rgb']} lidar={report['has_lidar']}")


if __name__ == "__main__":
    main()
