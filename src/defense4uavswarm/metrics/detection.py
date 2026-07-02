from __future__ import annotations

import pandas as pd
import numpy as np

from defense4uavswarm.class_groups import normalized_names
from defense4uavswarm.tracking.simple import iou


def match_frame(
    gt: pd.DataFrame,
    pred: pd.DataFrame,
    iou_thr: float = 0.5,
    class_agnostic: bool = True,
    aliases: dict[str, str] | None = None,
) -> tuple[int, int, int, list[tuple[int, int]]]:
    pred = pred[pred["accepted"] == True] if "accepted" in pred else pred
    if len(gt) == 0:
        return 0, len(pred), 0, []
    if len(pred) == 0:
        return 0, 0, len(gt), []
    gt_idx = gt.index.to_numpy()
    pred_idx = pred.index.to_numpy()
    g = gt[["x1", "y1", "x2", "y2"]].to_numpy(dtype=float)
    p = pred[["x1", "y1", "x2", "y2"]].to_numpy(dtype=float)
    ix1 = np.maximum(g[:, None, 0], p[None, :, 0])
    iy1 = np.maximum(g[:, None, 1], p[None, :, 1])
    ix2 = np.minimum(g[:, None, 2], p[None, :, 2])
    iy2 = np.minimum(g[:, None, 3], p[None, :, 3])
    inter = np.maximum(0.0, ix2 - ix1) * np.maximum(0.0, iy2 - iy1)
    g_area = np.maximum(0.0, g[:, 2] - g[:, 0]) * np.maximum(0.0, g[:, 3] - g[:, 1])
    p_area = np.maximum(0.0, p[:, 2] - p[:, 0]) * np.maximum(0.0, p[:, 3] - p[:, 1])
    denom = g_area[:, None] + p_area[None, :] - inter
    scores = np.divide(inter, denom, out=np.zeros_like(inter), where=denom > 0)
    if not class_agnostic:
        gt_names = normalized_names(gt, aliases or {}).to_numpy()
        pred_names = normalized_names(pred, aliases or {}).to_numpy()
        scores = np.where(gt_names[:, None] == pred_names[None, :], scores, 0.0)
    gi, pi = np.where(scores >= iou_thr)
    order = np.argsort(scores[gi, pi])[::-1]
    used_g, used_p, matches = set(), set(), []
    for n in order:
        a, b = int(gi[n]), int(pi[n])
        if a in used_g or b in used_p:
            continue
        used_g.add(a)
        used_p.add(b)
        matches.append((int(gt_idx[a]), int(pred_idx[b])))
    tp = len(matches)
    fp = len(pred) - tp
    fn = len(gt) - tp
    return tp, fp, fn, matches


def ap50(gt: pd.DataFrame, pred: pd.DataFrame, iou_thr: float = 0.5, class_agnostic: bool = True, aliases: dict[str, str] | None = None) -> float:
    if len(gt) == 0:
        return 0.0
    pred = pred[pred["accepted"] == True] if "accepted" in pred else pred
    pred = pred.sort_values("confidence", ascending=False)
    gt_by_frame = {
        key: frame[["x1", "y1", "x2", "y2"]].to_numpy(dtype=float)
        for key, frame in gt.groupby(["sequence_id", "frame_id"])
    }
    gt_idx_by_frame = {
        key: frame.index.to_numpy()
        for key, frame in gt.groupby(["sequence_id", "frame_id"])
    }
    used_gt = set()
    tps, fps = [], []
    for p in pred.itertuples():
        key = (p.sequence_id, p.frame_id)
        gt_boxes = gt_by_frame.get(key)
        gt_indices = gt_idx_by_frame.get(key)
        if gt_boxes is None or len(gt_boxes) == 0:
            tps.append(0)
            fps.append(1)
            continue
        pbox = np.array([p.x1, p.y1, p.x2, p.y2], dtype=float)
        ix1 = np.maximum(gt_boxes[:, 0], pbox[0])
        iy1 = np.maximum(gt_boxes[:, 1], pbox[1])
        ix2 = np.minimum(gt_boxes[:, 2], pbox[2])
        iy2 = np.minimum(gt_boxes[:, 3], pbox[3])
        inter = np.maximum(0.0, ix2 - ix1) * np.maximum(0.0, iy2 - iy1)
        g_area = np.maximum(0.0, gt_boxes[:, 2] - gt_boxes[:, 0]) * np.maximum(0.0, gt_boxes[:, 3] - gt_boxes[:, 1])
        p_area = max(0.0, (pbox[2] - pbox[0]) * (pbox[3] - pbox[1]))
        denom = g_area + p_area - inter
        scores = np.divide(inter, denom, out=np.zeros_like(inter), where=denom > 0)
        if not class_agnostic:
            gt_names = normalized_names(gt.loc[gt_indices], aliases or {}).to_numpy()
            pred_name = normalized_names(pd.DataFrame([p._asdict()]), aliases or {}).iloc[0]
            scores = np.where(gt_names == pred_name, scores, 0.0)
        order = np.argsort(scores)[::-1]
        best_gi, best = None, 0.0
        for pos in order:
            gi = int(gt_indices[pos])
            if gi in used_gt:
                continue
            best_gi, best = gi, float(scores[pos])
            break
        if best_gi is not None and best >= iou_thr:
            used_gt.add(best_gi)
            tps.append(1)
            fps.append(0)
        else:
            tps.append(0)
            fps.append(1)
    if not tps:
        return 0.0
    cum_tp = pd.Series(tps).cumsum()
    cum_fp = pd.Series(fps).cumsum()
    recall = cum_tp / len(gt)
    precision = cum_tp / (cum_tp + cum_fp)
    ap = 0.0
    prev_r = 0.0
    for r, p in zip(recall, precision):
        ap += max(0.0, float(r - prev_r)) * float(p)
        prev_r = float(r)
    return ap


def precision_recall_f1(
    gt: pd.DataFrame,
    pred: pd.DataFrame,
    iou_thr: float = 0.5,
    include_map: bool = True,
    class_agnostic: bool = True,
    aliases: dict[str, str] | None = None,
) -> dict:
    tp = fp = fn = 0
    pred_frames = {k: v for k, v in pred.groupby(["sequence_id", "frame_id"])} if len(pred) else {}
    empty_pred = pred.iloc[0:0]
    for (seq, frame), gt_f in gt.groupby(["sequence_id", "frame_id"]):
        pred_f = pred_frames.get((seq, frame), empty_pred)
        a, b, c, _ = match_frame(gt_f, pred_f, iou_thr, class_agnostic=class_agnostic, aliases=aliases)
        tp += a
        fp += b
        fn += c
    pred_accepted = pred[pred["accepted"] == True] if "accepted" in pred else pred
    extra = pred_accepted.merge(gt[["sequence_id", "frame_id"]].drop_duplicates(), on=["sequence_id", "frame_id"], how="left", indicator=True)
    fp += int((extra["_merge"] == "left_only").sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "F1": f1,
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "mAP": ap50(gt, pred, iou_thr, class_agnostic=class_agnostic, aliases=aliases) if include_map else None,
    }


def choose_threshold(
    gt: pd.DataFrame,
    pred: pd.DataFrame,
    grid: list[float],
    score_col: str,
    tie_eps: float,
    class_agnostic: bool = True,
    aliases: dict[str, str] | None = None,
    recall_floor: float | None = None,
) -> tuple[float, pd.DataFrame]:
    rows = []
    best_tau, best_f1 = grid[0], -1.0
    for tau in grid:
        cand = pred.copy()
        cand["accepted"] = cand[score_col] >= tau
        m = precision_recall_f1(gt, cand, include_map=False, class_agnostic=class_agnostic, aliases=aliases)
        m["recall_floor"] = recall_floor
        m["passes_recall_floor"] = recall_floor is None or m["recall"] >= recall_floor
        rows.append({"tau": tau, **m})
        if recall_floor is not None and m["recall"] < recall_floor:
            continue
        if m["F1"] > best_f1 + tie_eps or abs(m["F1"] - best_f1) < tie_eps and tau < best_tau:
            best_tau, best_f1 = tau, m["F1"]
    return best_tau, pd.DataFrame(rows)
