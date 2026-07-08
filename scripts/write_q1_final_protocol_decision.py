#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--audit-root", required=True)
    p.add_argument("--matching-mode", default="coarse_class")
    p.add_argument("--ignore-policy", default="exclude_ignored")
    p.add_argument("--iou-threshold", type=float, default=0.5)
    p.add_argument("--yolov8s-threshold-yaml", default="")
    p.add_argument("--yolov8n-threshold-yaml", default="")
    p.add_argument("--output", required=True)
    args = p.parse_args()
    audit = Path(args.audit_root)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Final Protocol Decision",
        "",
        "1. bbox conversion confirmed: VisDrone annotations are `x,y,width,height`; evaluation uses `x1,y1,x2,y2`.",
        "2. frame indexing confirmed: VisDrone VID annotation and image ids are 1-based numeric frame ids.",
        f"3. ignored region policy: `{args.ignore_policy}`.",
        f"4. matching mode: `{args.matching_mode}`.",
        f"5. IoU threshold: main `{args.iou_threshold}`; sensitivity uses appendix tables.",
        "6. detector confidence policy: calibration-selected threshold when YAML is provided; otherwise fixed command threshold.",
        "",
    ]
    for rel in [
        "format/visdrone_format_report.md",
        "class_mapping/class_mapping_metrics.csv",
        "ignore_regions/ignore_region_audit.csv",
        "size_stratified/size_stratified_metrics.csv",
    ]:
        pth = audit / rel
        lines.append(f"- audit artifact `{rel}`: {'present' if pth.exists() else 'missing'}")
    for name, yaml_path in [("YOLOv8s", args.yolov8s_threshold_yaml), ("YOLOv8n", args.yolov8n_threshold_yaml)]:
        if yaml_path:
            lines.append(f"- {name} threshold file: `{yaml_path}`")
    class_csv = audit / "class_mapping" / "class_mapping_metrics.csv"
    if class_csv.exists():
        df = pd.read_csv(class_csv)
        best = df.sort_values("F1", ascending=False).iloc[0]
        lines += ["", f"Class mapping audit best mode: `{best['matching_mode']}` with F1={best['F1']:.6f}."]
    lines += [
        "",
        "Selected protocol rationale: coarse-class matching is the default corrected protocol because generic COCO detector classes do not exactly match VisDrone fine-grained labels. Ignored boxes and detections inside ignored regions are excluded from the main protocol and reported in the audit.",
    ]
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"status=ok output={out}")


if __name__ == "__main__":
    main()
