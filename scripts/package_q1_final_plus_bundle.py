#!/usr/bin/env python
from __future__ import annotations

import argparse
import subprocess
import zipfile
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="outputs/bundles/Defense4UAVSwarm_q1_final_plus_bundle.zip")
    args = p.parse_args()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    root = Path(args.output).stem
    include_roots = [
        Path("outputs/results/q1_final_plus"),
        Path("outputs/results/q1_final_corrected"),
        Path("outputs/results/q1_improvement"),
        Path("outputs/results/q1_summary"),
        Path("docs"),
        Path("configs"),
        Path("scripts"),
        Path("tests"),
    ]
    written: set[str] = set()
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as z:
        write_text_unique(
            z,
            written,
            f"{root}/README_Q1_FINAL_PLUS.md",
            "Q1 final-plus/v9 bundle. Large detector JSON/PKL/raw images/external repos/model weights are omitted.\n"
            "V9 adds real BoxMOT OC-SORT evaluation, StrongSORT no-ReID limitation logs, Russian statistics plots, and v9 article DOCX/PDF.\n",
        )
        for base in include_roots:
            if not base.exists():
                continue
            for pth in sorted(x for x in base.rglob("*") if x.is_file()):
                if should_skip(pth):
                    continue
                write_file_unique(z, written, pth, f"{root}/{pth.as_posix()}")
        for doc in [
            "Defense4UAVSwarm_article_final_ru_Q1_final_v7.docx",
            "Defense4UAVSwarm_article_final_ru_Q1_final_v7_scrubbed.docx",
            "Defense4UAVSwarm_article_final_ru_Q1_final_v9_tracker_stats.docx",
            "Defense4UAVSwarm_article_final_ru_Q1_final_v9_tracker_stats_scrubbed.docx",
            "outputs/results/q1_final_plus/article_render/Defense4UAVSwarm_article_final_ru_Q1_final_v9_tracker_stats_scrubbed.pdf",
        ]:
            if Path(doc).exists():
                write_file_unique(z, written, Path(doc), f"{root}/{doc}")
        try:
            git = subprocess.check_output(["git", "log", "-1", "--oneline"], text=True).strip()
        except Exception:
            git = "unavailable"
        write_text_unique(z, written, f"{root}/reproducibility/git_info.txt", git + "\n")
    print(f"status=ok bundle={out}")


def write_file_unique(z: zipfile.ZipFile, written: set[str], src: Path, arcname: str) -> None:
    if arcname in written:
        return
    z.write(src, arcname)
    written.add(arcname)


def write_text_unique(z: zipfile.ZipFile, written: set[str], arcname: str, content: str) -> None:
    if arcname in written:
        return
    z.writestr(arcname, content)
    written.add(arcname)


def should_skip(path: Path) -> bool:
    s = path.as_posix()
    return (
        path.suffix in {".pkl", ".pt", ".pth", ".onnx"}
        or s.endswith("_visdrone.json")
        or "/images/" in s
        or "raw" in path.name.lower()
        or "external_boxmot" in s
    )


if __name__ == "__main__":
    main()
