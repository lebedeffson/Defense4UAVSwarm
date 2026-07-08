#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import pandas as pd
import yaml

from defense4uavswarm.q1_visdrone import (
    add_adaptive_metrics,
    add_single_camera_features,
    adaptive_geometry_acceptance,
    label_detections_protocol,
    load_detections,
    load_gt_protocol,
    method_acceptance,
)


COLORS = {
    "gt": (0, 180, 0),
    "bytetrack": (0, 165, 255),
    "geometry_dynamic_adaptive_balanced": (255, 80, 0),
    "geometry_dynamic_false_new_safe": (255, 0, 255),
    "rf_learned_gate": (120, 120, 255),
}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-root", required=True)
    p.add_argument("--detections", required=True)
    p.add_argument("--methods", nargs="+", required=True)
    p.add_argument("--selected-configs", default="")
    p.add_argument("--output-dir", required=True)
    args = p.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    raw = load_detections(args.detections)
    gt, ignored = load_gt_protocol(args.dataset_root, sorted(raw["sequence_id"].unique()) if not raw.empty else None)
    keys = raw[["sequence_id", "frame_id"]].drop_duplicates()
    gt = gt.merge(keys, on=["sequence_id", "frame_id"], how="inner")
    ignored = ignored.merge(keys, on=["sequence_id", "frame_id"], how="inner") if not ignored.empty else ignored
    det = add_single_camera_features(label_detections_protocol(raw, gt, ignored, 0.5, "coarse_class", "exclude_ignored", 0.05))
    cfgs = yaml.safe_load(Path(args.selected_configs).read_text(encoding="utf-8")) if args.selected_configs and Path(args.selected_configs).exists() else {}
    accepts = {}
    for method in args.methods:
        if method == "geometry_dynamic_adaptive_balanced" and cfgs:
            accepts[method] = adaptive_geometry_acceptance(det, cfgs["selected_balanced"])
        elif method == "geometry_dynamic_false_new_safe" and cfgs:
            accepts[method] = adaptive_geometry_acceptance(det, cfgs["selected_false_new_safe"])
        elif method == "rf_learned_gate":
            accepts[method] = pd.Series(False, index=det.index)
        else:
            from defense4uavswarm.q1_visdrone import Q1Params

            accepts[method] = method_acceptance(det, method, Q1Params())
    examples = choose_examples(det, accepts)
    lines = ["# Q1 Qualitative Examples", ""]
    count = 0
    for label, idx in examples:
        if idx is None:
            continue
        row = det.loc[idx]
        img = cv2.imread(str(row.image_path))
        if img is None:
            continue
        frame_gt = gt[(gt["sequence_id"].eq(row.sequence_id)) & (gt["frame_id"].eq(row.frame_id))]
        for g in frame_gt.head(80).itertuples():
            draw_box(img, (g.x1, g.y1, g.x2, g.y2), COLORS["gt"], f"GT {g.class_name}")
        frame_det = det[(det["sequence_id"].eq(row.sequence_id)) & (det["frame_id"].eq(row.frame_id))]
        for method, acc in accepts.items():
            accepted = frame_det[acc.loc[frame_det.index].astype(bool)].head(40)
            for d in accepted.itertuples():
                draw_box(img, (d.x1, d.y1, d.x2, d.y2), COLORS.get(method, (255, 255, 0)), method[:8])
        count += 1
        path = out / f"example_{count:02d}_{label}.png"
        cv2.imwrite(str(path), img)
        lines.append(f"- {label}: `{path.name}` sequence={row.sequence_id} frame={int(row.frame_id)}")
    (out / "qualitative_examples.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"status=ok exported={count} output={out}")


def choose_examples(det: pd.DataFrame, accepts: dict[str, pd.Series]) -> list[tuple[str, int | None]]:
    out: list[tuple[str, int | None]] = []
    bt = accepts.get("bytetrack", pd.Series(False, index=det.index)).astype(bool)
    geom = accepts.get("geometry_dynamic_adaptive_balanced", pd.Series(False, index=det.index)).astype(bool)
    safe = accepts.get("geometry_dynamic_false_new_safe", pd.Series(False, index=det.index)).astype(bool)
    fp = ~det["eval_is_tp"].astype(bool)
    tp = det["eval_is_tp"].astype(bool)
    candidates = [
        ("bytetrack_false_new", det[bt & ~geom & fp]),
        ("geometry_suppression", det[geom & fp]),
        ("geometry_miss", det[bt & ~geom & tp]),
        ("dense_scene", det.loc[det.groupby(["sequence_id", "frame_id"])["det_id"].transform("count").sort_values(ascending=False).index]),
        ("small_objects", det.sort_values("bbox_area").head(200)),
    ]
    seen_frames: set[tuple[str, int]] = set()
    for name, frame in candidates:
        idx = None
        for r in frame.sort_values("confidence", ascending=False).itertuples():
            key = (r.sequence_id, int(r.frame_id))
            if key not in seen_frames:
                seen_frames.add(key)
                idx = int(r.Index)
                break
        out.append((name, idx))
    return out


def draw_box(img, box, color, label: str) -> None:
    x1, y1, x2, y2 = [int(round(float(v))) for v in box]
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    cv2.putText(img, label, (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)


if __name__ == "__main__":
    main()
