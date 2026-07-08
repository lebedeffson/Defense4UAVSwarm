#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--audit-root", required=True)
    p.add_argument("--final-root", required=True)
    p.add_argument("--output-md", required=True)
    p.add_argument("--output-csv", required=True)
    args = p.parse_args()
    audit = Path(args.audit_root)
    final = Path(args.final_root)
    rows: list[dict] = []
    md = ["# Q1 Final Practice Summary", ""]
    append_audit(md, rows, audit)
    append_final(md, rows, final)
    md += [
        "## Claim-Safe Interpretation",
        "",
        "- VisDrone is used as real-detector single-UAV validation, not as real multi-UAV validation.",
        "- Low absolute F1 should be interpreted together with raw detector PR, class mapping, ignored-region handling, and size-stratified results.",
        "- The defensible claim is an interpretable no-label/low-label trade-off, not universal superiority over RF or ByteTrack.",
        "",
        "## Article Tables",
        "",
        "- Use `q1_final_key_tables.csv` for compact article-ready values.",
        "- Use audit CSV files as appendix evidence for protocol correctness.",
    ]
    out_md = Path(args.output_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(md) + "\n", encoding="utf-8")
    out_csv = Path(args.output_csv)
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"status=ok output={out_md}")


def append_audit(md: list[str], rows: list[dict], audit: Path) -> None:
    md += ["## Audit", ""]
    fmt = audit / "format" / "sequence_inventory.csv"
    if fmt.exists():
        seq = pd.read_csv(fmt)
        rows.append({"section": "audit", "metric": "num_sequences", "value": seq["sequence_id"].nunique()})
        rows.append({"section": "audit", "metric": "num_frames", "value": int(seq["num_frames"].sum())})
        rows.append({"section": "audit", "metric": "num_gt", "value": int(seq["num_eval_gt"].sum())})
        rows.append({"section": "audit", "metric": "num_ignored", "value": int(seq["num_ignored_boxes"].sum())})
        md.append(f"- dataset: {seq['sequence_id'].nunique()} sequences, {int(seq['num_frames'].sum())} frames, {int(seq['num_eval_gt'].sum())} eval GT boxes.")
    for det in ["yolov8s", "yolov8n"]:
        pr = audit / f"{det}_detector_pr" / "detector_pr_curve.csv"
        if pr.exists():
            df = pd.read_csv(pr)
            best = df.sort_values("F1", ascending=False).iloc[0]
            rows.append({"section": "raw_detector", "detector": det, "metric": "best_F1", "value": best["F1"], "condition": f"conf={best['conf_threshold']} iou={best['iou_threshold']} mode={best['matching_mode']}"})
            md.append(f"- {det} raw detector best F1: {best['F1']:.6f} at conf={best['conf_threshold']}, IoU={best['iou_threshold']}, mode={best['matching_mode']}.")
    cls = audit / "class_mapping" / "class_mapping_metrics.csv"
    if cls.exists():
        df = pd.read_csv(cls)
        best = df.sort_values("F1", ascending=False).iloc[0]
        rows.append({"section": "class_mapping", "metric": "best_mode", "value": best["matching_mode"], "F1": best["F1"]})
        md.append(f"- class mapping: best mode `{best['matching_mode']}` with F1={best['F1']:.6f}.")
    size = audit / "size_stratified" / "size_stratified_metrics.csv"
    if size.exists():
        df = pd.read_csv(size)
        for _, r in df.iterrows():
            rows.append({"section": "size_stratified", "metric": "F1", "condition": r["size_bin"], "value": r["F1"]})
        md.append("- size-stratified metrics are available in `size_stratified_metrics.csv`.")
    md.append("")


def append_final(md: list[str], rows: list[dict], final: Path) -> None:
    md += ["## Corrected Final Results", ""]
    for det in ["yolov8s", "yolov8n"]:
        path = final / f"{det}_main" / "main_comparison_table.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        md += [f"### {det}", "", "```text", df[["method", "F1", "false_new_tracks"]].to_string(index=False), "```", ""]
        for _, r in df.iterrows():
            rows.append({"section": "final_main", "detector": det, "method": r["method"], "F1": r["F1"], "false_new_tracks": r["false_new_tracks"]})
    for det in ["yolov8s", "yolov8n"]:
        path = final / f"label_scarcity_{det}" / "label_budget_mean_std.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        for _, r in df.iterrows():
            rows.append({"section": "label_scarcity", "detector": det, "label_budget": r["label_budget"], "method": r["method"], "F1": r.get("F1_mean"), "false_new_tracks": r.get("false_new_tracks_mean")})


if __name__ == "__main__":
    main()
