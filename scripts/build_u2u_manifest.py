#!/usr/bin/env python
from __future__ import annotations

import argparse
import json

from defense4uavswarm.datasets.u2u_adapter import U2UDataAdapter


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-root", default="data/U2UData")
    p.add_argument("--split", default="validation")
    p.add_argument("--agents", type=int, default=3)
    p.add_argument("--min-frames", type=int, default=300)
    p.add_argument("--output-root", required=True)
    args = p.parse_args()
    adapter = U2UDataAdapter(args.dataset_root)
    if not adapter.exists():
        manifest = {
            "dataset": "U2UData",
            "status": "blocked_missing_dataset",
            "root": args.dataset_root,
            "required_next_step": "Download U2UData/U2UData+ locally and rerun this command.",
        }
        import pathlib
        out = pathlib.Path(args.output_root)
        out.mkdir(parents=True, exist_ok=True)
        (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        for name in ["scenes.csv", "agents.csv", "frames.csv", "gt_summary.csv"]:
            (out / name).write_text("status\nblocked_missing_dataset\n", encoding="utf-8")
        print("status=blocked_missing_dataset")
        return
    manifest = adapter.export_manifest(args.output_root, args.split, args.agents, args.min_frames)
    print(f"status=ok scenes={len(manifest['scenes'])} frames={len(manifest['frames'])}")


if __name__ == "__main__":
    main()
