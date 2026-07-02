from __future__ import annotations

import pandas as pd

from defense4uavswarm.metrics.detection import match_frame
from defense4uavswarm.tracking.simple import iou


def tracking_metrics(gt: pd.DataFrame, pred: pd.DataFrame) -> dict:
    try:
        out = _motmetrics_tracking(gt, pred)
    except Exception:
        out = _fallback_tracking(gt, pred)
    out["track_breaks"] = track_breaks(gt, pred)
    return out


def track_breaks(gt: pd.DataFrame, pred: pd.DataFrame, max_gap: int = 3) -> int:
    last_frame = {}
    breaks = 0
    for (seq, frame), gt_f in gt.groupby(["sequence_id", "frame_id"]):
        pred_f = pred[(pred.sequence_id == seq) & (pred.frame_id == frame)]
        _, _, _, pairs = match_frame(gt_f, pred_f)
        for gi, _ in pairs:
            gtid = int(gt.loc[gi, "gt_track_id"])
            key = (seq, gtid)
            if key in last_frame and frame - last_frame[key] > max_gap:
                breaks += 1
            last_frame[key] = frame
    return breaks


def _motmetrics_tracking(gt: pd.DataFrame, pred: pd.DataFrame) -> dict:
    import motmetrics as mm
    import numpy as np

    acc = mm.MOTAccumulator(auto_id=True)
    seq_offsets = {seq: i * 1_000_000 for i, seq in enumerate(sorted(gt["sequence_id"].dropna().unique()))}
    for (seq, frame), gt_f in gt.groupby(["sequence_id", "frame_id"]):
        pred_f = pred[(pred.sequence_id == seq) & (pred.frame_id == frame)]
        pred_f = pred_f[pred_f["accepted"] == True] if "accepted" in pred_f else pred_f
        offset = seq_offsets.get(seq, 0)
        gt_ids = [offset + int(r.gt_track_id) for r in gt_f.itertuples()]
        pred_ids = [offset + int(r.pred_track_id) for r in pred_f.itertuples() if pd.notna(r.pred_track_id)]
        pred_rows = [r for r in pred_f.itertuples() if pd.notna(r.pred_track_id)]
        dists = np.full((len(gt_ids), len(pred_ids)), np.nan)
        for gi, g in enumerate(gt_f.itertuples()):
            for pi, p in enumerate(pred_rows):
                score = iou((g.x1, g.y1, g.x2, g.y2), (p.x1, p.y1, p.x2, p.y2))
                if score >= 0.5:
                    dists[gi, pi] = 1.0 - score
        acc.update(gt_ids, pred_ids, dists)
    mh = mm.metrics.create()
    s = mh.compute(acc, metrics=["mota", "idf1", "num_switches", "num_false_positives", "num_misses"], name="run")
    return {
        "MOTA": float(s.loc["run", "mota"]),
        "IDF1": float(s.loc["run", "idf1"]),
        "IDSW": int(s.loc["run", "num_switches"]),
        "tracking_error": int(s.loc["run", "num_false_positives"] + s.loc["run", "num_misses"] + s.loc["run", "num_switches"]),
    }


def _fallback_tracking(gt: pd.DataFrame, pred: pd.DataFrame) -> dict:
    idsw = 0
    matched = {}
    fp = fn = tp = 0
    for (seq, frame), gt_f in gt.groupby(["sequence_id", "frame_id"]):
        pred_f = pred[(pred.sequence_id == seq) & (pred.frame_id == frame)]
        a, b, c, pairs = match_frame(gt_f, pred_f)
        tp += a
        fp += b
        fn += c
        for gi, pi in pairs:
            gtid = int(gt.loc[gi, "gt_track_id"])
            ptid = pred.loc[pi, "pred_track_id"]
            key = (seq, gtid)
            if key in matched and matched[key] != ptid:
                idsw += 1
            matched[key] = ptid
    mota = 1 - (fn + fp + idsw) / max(1, len(gt))
    idf1 = 2 * tp / max(1, 2 * tp + fp + fn)
    return {"MOTA": mota, "IDF1": idf1, "IDSW": idsw, "tracking_error": fp + fn + idsw}
