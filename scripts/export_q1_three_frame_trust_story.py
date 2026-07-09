#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml

from defense4uavswarm.q1_visdrone import adaptive_geometry_acceptance


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-root", required=True)
    p.add_argument("--detections", required=True)
    p.add_argument("--method", default="geometry_dynamic_adaptive_balanced")
    p.add_argument("--feature-audit", default="outputs/results/q1_final_corrected/yolov8s_main/feature_audit.csv")
    p.add_argument("--selected-configs", default="outputs/results/q1_improvement/yolov8s_sweep/selected_configs.yaml")
    p.add_argument("--output-dir", required=True)
    args = p.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    det = pd.read_csv(args.feature_audit)
    cfgs = yaml.safe_load(Path(args.selected_configs).read_text(encoding="utf-8"))
    cfg = cfgs["selected_balanced"] if args.method.endswith("adaptive_balanced") else cfgs.get("selected_false_new_safe", cfgs["selected_balanced"])
    det = det.copy()
    det["accepted"] = adaptive_geometry_acceptance(det, cfg)
    det["Q_i"] = np.minimum(det["c_i"].astype(float).clip(0, 1), det["k_i"].astype(float).clip(0, 1))
    det["decision"] = np.where(det["accepted"].astype(bool), "accepted", np.where(det["temporal_age"].astype(float) > 0, "delayed", "rejected"))
    det["reason"] = det.apply(limiting_reason, axis=1)

    story = choose_story(det)
    if story.empty:
        raise ValueError("Could not find a three-frame story candidate")
    canvas_ru = render_story(story, lang="ru")
    canvas_en = render_story(story, lang="en")
    cv2.imwrite(str(out / "fig_three_frame_trust_story_ru.png"), canvas_ru)
    cv2.imwrite(str(out / "fig_three_frame_trust_story_en.png"), canvas_en)
    write_summary(out / "three_frame_trust_story_summary.md", story, args.method)
    print(f"status=ok sequence={story.iloc[0].sequence_id} frames={list(story.frame_id.astype(int))} output={out}")


def choose_story(det: pd.DataFrame) -> pd.DataFrame:
    d = det[det["image_path"].astype(str).str.len().gt(0)].copy()
    d = d[d["tracklet_id"].astype(str).str.len().gt(0)]
    candidates = []
    for _, group in d.groupby("tracklet_id", sort=False):
        frames = group.sort_values("frame_id")
        if frames["frame_id"].nunique() < 3:
            continue
        accepted_count = int(frames["accepted"].astype(bool).sum())
        accepted_tp = int((frames["accepted"].astype(bool) & frames["eval_is_tp"].astype(bool)).sum()) if "eval_is_tp" in frames else 0
        rejected_count = int((~frames["accepted"].astype(bool)).sum())
        delayed_count = int(frames["decision"].eq("delayed").sum()) if "decision" in frames else 0
        score = 10 * accepted_tp + 2 * rejected_count + delayed_count + accepted_count + float(frames["confidence"].mean())
        candidates.append((score, frames))
    if not candidates:
        return pd.DataFrame()
    candidates.sort(key=lambda x: x[0], reverse=True)
    for _, frames in candidates[:200]:
        rows = consecutive_three(frames)
        if not rows.empty:
            return rows
    return consecutive_three(candidates[0][1])


def consecutive_three(group: pd.DataFrame) -> pd.DataFrame:
    g = group.sort_values("frame_id").drop_duplicates("frame_id")
    frame_ids = list(g["frame_id"].astype(int))
    best = pd.DataFrame()
    for a, b, c in zip(frame_ids, frame_ids[1:], frame_ids[2:]):
        if b == a + 1 and c == b + 1:
            window = g[g["frame_id"].astype(int).isin([a, b, c])].sort_values("frame_id").head(3)
            has_accepted_tp = bool((window["accepted"].astype(bool) & window.get("eval_is_tp", pd.Series(False, index=window.index)).astype(bool)).any())
            has_not_accepted = bool((~window["accepted"].astype(bool)).any())
            if has_accepted_tp and has_not_accepted:
                return window
            if best.empty:
                best = window
    if not best.empty:
        return best
    return g.head(3) if len(g) >= 3 else pd.DataFrame()


def render_story(story: pd.DataFrame, lang: str) -> np.ndarray:
    panels = []
    for _, row in story.iterrows():
        img = cv2.imread(str(row.image_path))
        if img is None:
            img = np.zeros((720, 1280, 3), dtype=np.uint8)
        img = resize_keep(img, 520, 330)
        x_scale = img.shape[1] / max(1.0, float(row.get("image_width", 1920)))
        y_scale = img.shape[0] / max(1.0, float(row.get("image_height", 1080)))
        x1 = int(float(row.x1) * x_scale)
        y1 = int(float(row.y1) * y_scale)
        x2 = int(float(row.x2) * x_scale)
        y2 = int(float(row.y2) * y_scale)
        color = (40, 190, 40) if bool(row.accepted) else (40, 40, 220)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        status = str(row.decision)
        if lang == "ru":
            title = f"frame {int(row.frame_id)}: {status}"
            label = f"c={float(row.c_i):.2f} k={float(row.k_i):.2f} Q={float(row.Q_i):.2f}"
            reason = "proxy: temporal/geometric; min: " + str(row.reason)
        else:
            title = f"frame {int(row.frame_id)}: {status}"
            label = f"c={float(row.c_i):.2f} k={float(row.k_i):.2f} Q={float(row.Q_i):.2f}"
            reason = "temporal/geometric proxy; min: " + str(row.reason)
        cv2.rectangle(img, (0, 0), (img.shape[1], 58), (255, 255, 255), -1)
        cv2.putText(img, title, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(img, label, (8, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(img, reason[:70], (8, img.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1, cv2.LINE_AA)
        panels.append(img)
    sep = np.full((panels[0].shape[0], 8, 3), 255, dtype=np.uint8)
    return np.hstack([panels[0], sep, panels[1], sep, panels[2]])


def resize_keep(img: np.ndarray, width: int, height: int) -> np.ndarray:
    resized = cv2.resize(img, (width, height))
    return resized


def limiting_reason(row) -> str:
    c = float(row.get("c_i", row.get("confidence", 0)))
    k = float(row.get("k_i", 0))
    age = float(row.get("temporal_age", 0))
    vals = {
        "detector_confidence": c,
        "kinematic_consistency": k,
        "temporal_support": min(1.0, age / 3.0),
    }
    ordered = sorted(vals.items(), key=lambda x: x[1])
    if len(ordered) > 1 and abs(ordered[0][1] - ordered[1][1]) < 0.05:
        return "mixed_or_tie"
    return ordered[0][0]


def write_summary(path: Path, story: pd.DataFrame, method: str) -> None:
    seq = str(story.iloc[0]["sequence_id"])
    frames = ", ".join(str(int(x)) for x in story["frame_id"])
    cols = ["frame_id", "det_id", "confidence", "c_i", "k_i", "Q_i", "temporal_age", "decision", "reason", "eval_is_tp"]
    accepted_tp = bool((story["decision"].eq("accepted") & story.get("eval_is_tp", pd.Series(False, index=story.index)).astype(bool)).any())
    ready_article = "yes" if accepted_tp else "no; the selected transition is useful diagnostically but the accepted endpoint is not matched as TP"
    lines = [
        "# Three-Frame Trust Story Summary",
        "",
        f"method: `{method}`",
        f"sequence: `{seq}`",
        f"frames: `{frames}`",
        "",
        "Available features shown: detector confidence `c_i`, temporal/geometric proxy `k_i`, `Q_i=min(c_i,k_i)`, temporal age, decision, limiting feature.",
        "VisDrone is single-camera; no inter-agent consistency is shown or claimed.",
        "Example selection: a tracklet with at least three frames and visible accepted/rejected trust decisions.",
        "Label readability: generated as a wide three-panel figure; use in supplementary if page space is tight.",
        f"Ready for article: {ready_article}.",
        "Ready for supplementary: yes.",
        "",
        "```text",
        story[[c for c in cols if c in story.columns]].to_string(index=False),
        "```",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
