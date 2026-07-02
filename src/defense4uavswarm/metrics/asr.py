from __future__ import annotations

import pandas as pd

from defense4uavswarm.metrics.detection import match_frame


def attack_success_rate(gt: pd.DataFrame, clean: pd.DataFrame, attacked: pd.DataFrame) -> float:
    clean_hits = set()
    bad_tracks = set()
    last_pred = {}
    clean_frames = {k: v for k, v in clean.groupby(["sequence_id", "frame_id"])}
    attacked_frames = {k: v for k, v in attacked.groupby(["sequence_id", "frame_id"])}
    for (seq, frame), gt_f in gt.groupby(["sequence_id", "frame_id"]):
        clean_f = clean_frames.get((seq, frame), clean.iloc[0:0])
        _, _, _, clean_pairs = match_frame(gt_f, clean_f)
        hit_gt = {gi for gi, _ in clean_pairs}
        for gi in hit_gt:
            gtid = int(gt.loc[gi, "gt_track_id"])
            clean_hits.add((seq, gtid))
        atk_f = attacked_frames.get((seq, frame), attacked.iloc[0:0])
        _, fp, _, atk_pairs = match_frame(gt_f, atk_f)
        atk_hit = {gi: pi for gi, pi in atk_pairs}
        if fp > 0:
            for key in clean_hits:
                if key[0] == seq:
                    bad_tracks.add(key)
        for gi in hit_gt:
            gtid = int(gt.loc[gi, "gt_track_id"])
            key = (seq, gtid)
            if gi not in atk_hit:
                bad_tracks.add(key)
                continue
            ptid = attacked.loc[atk_hit[gi], "pred_track_id"]
            prev = last_pred.get(key)
            if prev is not None and prev != ptid:
                bad_tracks.add(key)
            last_pred[key] = ptid
    return len(bad_tracks) / max(1, len(clean_hits))


def attack_success_breakdown(gt: pd.DataFrame, clean: pd.DataFrame, attacked: pd.DataFrame, delta: float = 0.05) -> dict:
    clean_hits = set()
    missed, broken, idsw = set(), set(), set()
    matched_pred_ids, fp_pred_ids = set(), set()
    last_pred, last_frame = {}, {}
    bad_frames = total_frames = 0
    clean_frames = {k: v for k, v in clean.groupby(["sequence_id", "frame_id"])}
    attacked_frames = {k: v for k, v in attacked.groupby(["sequence_id", "frame_id"])}
    for (seq, frame), gt_f in gt.groupby(["sequence_id", "frame_id"]):
        clean_f = clean_frames.get((seq, frame), clean.iloc[0:0])
        atk_f = attacked_frames.get((seq, frame), attacked.iloc[0:0])
        c_tp, c_fp, c_fn, clean_pairs = match_frame(gt_f, clean_f)
        a_tp, a_fp, a_fn, atk_pairs = match_frame(gt_f, atk_f)
        c_f1 = 2 * c_tp / max(1, 2 * c_tp + c_fp + c_fn)
        a_f1 = 2 * a_tp / max(1, 2 * a_tp + a_fp + a_fn)
        total_frames += 1
        if a_f1 < c_f1 - delta:
            bad_frames += 1
        hit_gt = {gi for gi, _ in clean_pairs}
        atk_hit = {gi: pi for gi, pi in atk_pairs}
        matched_pred_ids.update(_pred_ids(attacked, [pi for _, pi in atk_pairs]))
        accepted = atk_f[atk_f["accepted"] == True] if "accepted" in atk_f else atk_f
        unmatched_pred = set(accepted.index) - {pi for _, pi in atk_pairs}
        fp_pred_ids.update(_pred_ids(attacked, unmatched_pred))
        for gi in hit_gt:
            gtid = int(gt.loc[gi, "gt_track_id"])
            key = (seq, gtid)
            clean_hits.add(key)
            if gi not in atk_hit:
                missed.add(key)
                continue
            ptid = attacked.loc[atk_hit[gi], "pred_track_id"]
            if pd.isna(ptid):
                continue
            if key in last_frame and frame - last_frame[key] > 3:
                broken.add(key)
            if key in last_pred and last_pred[key] != ptid and frame - last_frame[key] <= 1:
                idsw.add(key)
            last_pred[key] = ptid
            last_frame[key] = frame
    num_gt = max(1, len(clean_hits))
    bad_gt = missed | broken | idsw
    fp_tracks = len(fp_pred_ids - matched_pred_ids)
    return {
        "num_gt_tracks": len(clean_hits),
        "missed_tracks": len(missed),
        "false_positive_tracks": fp_tracks,
        "track_break_tracks": len(broken),
        "id_switch_tracks": len(idsw),
        "asr_total": min(1.0, (len(bad_gt) + fp_tracks) / num_gt),
        "asr_miss": len(missed) / num_gt,
        "asr_fp": fp_tracks / num_gt,
        "asr_break": len(broken) / num_gt,
        "asr_idsw": len(idsw) / num_gt,
        "ASR_frame": bad_frames / max(1, total_frames),
    }


def _pred_ids(df: pd.DataFrame, indices) -> set:
    ids = set()
    if "pred_track_id" not in df:
        return ids
    for idx in indices:
        value = df.loc[idx, "pred_track_id"]
        ids.add(f"det_{idx}" if pd.isna(value) else value)
    return ids
