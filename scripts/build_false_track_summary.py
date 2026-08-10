#!/usr/bin/env python
from __future__ import annotations

import argparse
import pandas as pd


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--track-events", required=True)
    p.add_argument("--gt-root", default=None)
    p.add_argument("--iou-threshold", type=float, default=0.5)
    p.add_argument("--matching-mode", default="class_aware")
    p.add_argument("--output", required=True)
    args = p.parse_args()
    events = pd.read_csv(args.track_events)
    rows = []
    for scenario, group in events.groupby("scenario", sort=False):
        created = group[group["is_created_track"] == True]
        suppressed = group[group["is_suppressed"] == True]
        true_new = created[created["is_tp_detection"] == True]
        false_new = created[(created["is_tp_detection"] == False) | (created.get("is_high_conf_fp", False) == True)]
        suppressed_true = suppressed[suppressed["is_tp_detection"] == True]
        suppressed_false = suppressed[(suppressed["is_tp_detection"] == False) | (suppressed.get("is_high_conf_fp", False) == True)]
        num_frames = max(1, group[["sequence_id", "frame_id"]].drop_duplicates().shape[0])
        rows.append(
            {
                "scenario": scenario,
                "num_frames": num_frames,
                "new_tracks_total": len(created),
                "true_new_tracks": len(true_new),
                "false_new_tracks": len(false_new),
                "false_new_tracks_per_100_frames": len(false_new) / num_frames * 100.0,
                "suppressed_new_tracks_total": len(suppressed),
                "suppressed_true_new_tracks": len(suppressed_true),
                "suppressed_false_new_tracks": len(suppressed_false),
                "suppressed_false_rate": len(suppressed_false) / max(1, len(suppressed)),
                "suppressed_true_rate": len(suppressed_true) / max(1, len(suppressed)),
                "matching_mode": args.matching_mode,
                "iou_threshold": args.iou_threshold,
            }
        )
    out = pd.DataFrame(rows)
    if {"S_naive", "S2_tnorm_soft"}.issubset(set(out["scenario"])):
        base = out[out.scenario == "S_naive"].iloc[0]
        method = out[out.scenario == "S2_tnorm_soft"].iloc[0]
        reduction = int(base["false_new_tracks"] - method["false_new_tracks"])
        out["operator_alarm_reduction_estimate"] = out["scenario"].map({"S2_tnorm_soft": reduction}).fillna(0).astype(int)
        out["operator_alarm_reduction_percent"] = out["scenario"].map({"S2_tnorm_soft": 100.0 * reduction / max(1, int(base["false_new_tracks"]))}).fillna(0.0)
    out.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
