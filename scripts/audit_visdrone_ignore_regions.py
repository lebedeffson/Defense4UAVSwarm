#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from defense4uavswarm.q1_visdrone import annotate_ignored_detections, compute_metrics, label_detections_protocol, load_detections, load_gt_protocol


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-root", required=True)
    p.add_argument("--detections", required=True)
    p.add_argument("--output-dir", required=True)
    args = p.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    det = load_detections(args.detections)
    gt, ignored = load_gt_protocol(args.dataset_root, sorted(det["sequence_id"].unique()) if not det.empty else None)
    keys = det[["sequence_id", "frame_id"]].drop_duplicates() if not det.empty else pd.DataFrame(columns=["sequence_id", "frame_id"])
    if not gt.empty:
        gt = gt.merge(keys, on=["sequence_id", "frame_id"], how="inner")
    if not ignored.empty:
        ignored = ignored.merge(keys, on=["sequence_id", "frame_id"], how="inner")
    flagged = annotate_ignored_detections(det, ignored)
    rows = []
    num_frames = max(1, gt[["sequence_id", "frame_id"]].drop_duplicates().shape[0]) if not gt.empty else 1
    for policy in ["count_ignored", "exclude_ignored", "report_ignored"]:
        labeled = label_detections_protocol(det, gt, ignored, 0.5, "coarse_class", policy, 0.05)
        row = compute_metrics(labeled, pd.Series(True, index=labeled.index), len(gt), num_frames, f"policy_{policy}")
        row.update(
            {
                "ignore_policy": policy,
                "num_ignored_gt": len(ignored),
                "detections_inside_ignored": int(flagged["inside_ignored_region"].astype(bool).sum()) if len(flagged) else 0,
                "detections_evaluated": len(labeled),
            }
        )
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(out / "ignore_region_audit.csv", index=False)
    md = [
        "# Ignore Region Audit",
        "",
        f"- ignored GT boxes: {len(ignored)}",
        f"- detections inside ignored regions: {int(flagged['inside_ignored_region'].astype(bool).sum()) if len(flagged) else 0}",
        "- final recommendation: use `exclude_ignored` and report the policy explicitly.",
    ]
    (out / "ignore_region_audit.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"status=ok output={out / 'ignore_region_audit.csv'}")


if __name__ == "__main__":
    main()
