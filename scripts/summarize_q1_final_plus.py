#!/usr/bin/env python
from __future__ import annotations

import argparse
import shutil
import zipfile
from pathlib import Path

import pandas as pd


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="outputs/results/q1_final_plus")
    p.add_argument("--output-md", default="outputs/results/q1_final_plus/q1_final_plus_summary.md")
    p.add_argument("--output-csv", default="outputs/results/q1_final_plus/q1_final_plus_key_tables.csv")
    p.add_argument("--docx", default="Defense4UAVSwarm_article_final_ru_Q1_final_v7.docx")
    p.add_argument("--scrubbed-docx", default="Defense4UAVSwarm_article_final_ru_Q1_final_v7_scrubbed.docx")
    args = p.parse_args()
    root = Path(args.root)
    rows = []
    md = ["# Q1 Final Plus Summary", ""]
    add_table(md, rows, root / "tracker_comparison_yolov8s" / "tracker_comparison_summary.csv", "tracker_comparison")
    add_table(md, rows, root / "statistics_yolov8s" / "bootstrap_ci.csv", "statistics_bootstrap")
    add_table(md, rows, root / "agent_scaling" / "agent_scaling_summary.csv", "agent_scaling")
    add_table(md, rows, root / "runtime" / "runtime_summary.csv", "runtime")
    add_table(md, rows, root / "rf_trust_hybrid" / "rf_trust_hybrid_summary.csv", "rf_trust_hybrid")
    add_table(md, rows, root / "density_size_analysis" / "density_metrics.csv", "density")
    md += [
        "## Claim-Safe Position",
        "",
        "- VisDrone remains real-detector single-UAV validation, not real multi-UAV validation.",
        "- External OC-SORT/StrongSORT results are reported only if imports/runs succeeded; otherwise logs are included.",
        "- Controlled agent scaling is simulation/proxy evidence, not real-world swarm validation.",
    ]
    out_md = Path(args.output_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(md) + "\n", encoding="utf-8")
    pd.DataFrame(rows).to_csv(args.output_csv, index=False)
    write_docx(Path(args.docx), out_md)
    shutil.copyfile(args.docx, args.scrubbed_docx)
    print(f"status=ok output={out_md}")


def add_table(md: list[str], rows: list[dict], path: Path, section: str) -> None:
    if not path.exists():
        md += [f"## {section}", "", f"Missing: `{path}`", ""]
        return
    df = pd.read_csv(path)
    md += [f"## {section}", "", "```text", df.head(30).to_string(index=False), "```", ""]
    for _, r in df.head(200).iterrows():
        d = r.to_dict()
        d["section"] = section
        rows.append(d)


def write_docx(path: Path, summary_md: Path) -> None:
    from docx import Document

    doc = Document()
    doc.add_heading("Defense4UAVSwarm Q1 Final v7", 0)
    doc.add_paragraph("Итоговая claim-safe версия по дополнительным экспериментам без новых данных.")
    text = summary_md.read_text(encoding="utf-8").splitlines()
    for line in text:
        if line.startswith("# "):
            doc.add_heading(line[2:], level=1)
        elif line.startswith("## "):
            doc.add_heading(line[3:], level=2)
        elif line.strip().startswith("- "):
            doc.add_paragraph(line.strip()[2:], style="List Bullet")
        elif line.strip() and not line.startswith("```"):
            doc.add_paragraph(line)
    doc.add_heading("Ограничения", level=1)
    doc.add_paragraph("Работа не заявляет real-world multi-UAV validation на VisDrone. RF и сильные трекеры остаются важными supervised/engineering baselines.")
    doc.save(path)


if __name__ == "__main__":
    main()
