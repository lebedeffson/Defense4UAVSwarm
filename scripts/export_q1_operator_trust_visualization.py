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
    p.add_argument("--num-examples", type=int, default=8)
    p.add_argument("--feature-audit", default="outputs/results/q1_final_corrected/yolov8s_main/feature_audit.csv")
    p.add_argument("--selected-configs", default="outputs/results/q1_improvement/yolov8s_sweep/selected_configs.yaml")
    p.add_argument("--output-dir", required=True)
    args = p.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    det = pd.read_csv(args.feature_audit)
    cfgs = yaml.safe_load(Path(args.selected_configs).read_text(encoding="utf-8"))
    accepted = adaptive_geometry_acceptance(det, cfgs["selected_balanced"])
    det = det.copy()
    det["accepted"] = accepted
    det["reason"] = det.apply(reason, axis=1)
    sample = choose(det, args.num_examples)
    images = []
    lines = ["# Operator Trust Visualization", ""]
    for i, (_, row) in enumerate(sample.iterrows(), 1):
        img = cv2.imread(str(row.image_path))
        if img is None:
            continue
        frame = det[(det.sequence_id.eq(row.sequence_id)) & (det.frame_id.eq(row.frame_id))].sort_values("confidence", ascending=False).head(40)
        for r in frame.itertuples():
            status = "accepted" if bool(r.accepted) else "rejected"
            color = (0, 200, 0) if status == "accepted" else (0, 0, 255)
            x1, y1, x2, y2 = [int(getattr(r, a)) for a in ["x1", "y1", "x2", "y2"]]
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            q = min(float(getattr(r, "c_i", 0)), float(getattr(r, "k_i", 0)))
            cv2.putText(img, f"Q={q:.2f} {r.reason} {status}", (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
        path = out / f"example_{i:02d}.png"
        cv2.imwrite(str(path), img)
        images.append(img)
        lines.append(f"- example_{i:02d}.png: {row.sequence_id} frame {int(row.frame_id)}")
    if images:
        sheet = contact_sheet(images)
        cv2.imwrite(str(out / "operator_visualization_contact_sheet.png"), sheet)
    (out / "operator_visualization_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"status=ok examples={len(images)} output={out}")


def choose(det: pd.DataFrame, n: int) -> pd.DataFrame:
    parts = [
        det[det["accepted"].astype(bool)].sort_values("confidence", ascending=False).head(max(2, n // 3)),
        det[~det["accepted"].astype(bool)].sort_values("confidence", ascending=False).head(max(2, n // 3)),
        det[(~det["accepted"].astype(bool)) & (~det["eval_is_tp"].astype(bool))].sort_values("confidence", ascending=False).head(max(2, n // 3)),
        det.sort_values("num_detections_in_frame", ascending=False).head(n),
    ]
    return pd.concat(parts).drop_duplicates("det_id").head(n)


def contact_sheet(images: list[np.ndarray]) -> np.ndarray:
    thumbs = []
    for img in images:
        thumbs.append(cv2.resize(img, (480, 270)))
    rows = []
    for i in range(0, len(thumbs), 2):
        row = thumbs[i : i + 2]
        if len(row) == 1:
            row.append(np.zeros_like(row[0]))
        rows.append(np.hstack(row))
    return np.vstack(rows)


def reason(row) -> str:
    c = float(row.get("c_i", row.get("confidence", 0)))
    k = float(row.get("k_i", 0))
    age = float(row.get("temporal_age", 0))
    vals = {
        "low detector confidence": c,
        "unstable motion": k,
        "weak temporal support": min(1.0, age / 3.0),
    }
    ordered = sorted(vals.items(), key=lambda x: x[1])
    if len(ordered) > 1 and abs(ordered[0][1] - ordered[1][1]) < 0.05:
        return "mixed limitation"
    return ordered[0][0]


if __name__ == "__main__":
    main()
