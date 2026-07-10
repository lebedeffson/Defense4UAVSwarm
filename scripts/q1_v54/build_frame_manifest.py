#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from defense4uavswarm.q1_v5.frame_manifest import add_gt_presence, build_visdrone_frame_manifest, manifest_sha256
from defense4uavswarm.q1_visdrone import load_gt_protocol


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-root", default="data/visdrone/VisDrone2019-VID-val")
    p.add_argument("--output", default="data_manifests/visdrone_eval_frames.csv")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    out = Path(args.output)
    if out.exists() and not args.overwrite and not args.dry_run:
        raise SystemExit(f"Output exists; use --overwrite: {out}")
    manifest = build_visdrone_frame_manifest(args.dataset_root)
    gt, _ = load_gt_protocol(args.dataset_root, sorted(manifest["sequence_id"].unique()))
    manifest = add_gt_presence(manifest, gt)
    if args.dry_run:
        print(f"dry_run=ok rows={len(manifest)} sha256={manifest_sha256(manifest)}")
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(out, index=False)
    print(f"status=ok output={out} rows={len(manifest)} sha256={manifest_sha256(manifest)}")


if __name__ == "__main__":
    main()
