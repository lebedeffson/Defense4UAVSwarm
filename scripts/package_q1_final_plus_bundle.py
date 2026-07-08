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
    root = "Defense4UAVSwarm_q1_final_plus_bundle"
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
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr(f"{root}/README_Q1_FINAL_PLUS.md", "Q1 final-plus bundle. Large detector JSON/PKL/raw images are omitted.\n")
        for base in include_roots:
            if not base.exists():
                continue
            for pth in sorted(x for x in base.rglob("*") if x.is_file()):
                if should_skip(pth):
                    continue
                z.write(pth, f"{root}/{pth.as_posix()}")
        for doc in ["Defense4UAVSwarm_article_final_ru_Q1_final_v7.docx", "Defense4UAVSwarm_article_final_ru_Q1_final_v7_scrubbed.docx"]:
            if Path(doc).exists():
                z.write(doc, f"{root}/{doc}")
        try:
            git = subprocess.check_output(["git", "log", "-1", "--oneline"], text=True).strip()
        except Exception:
            git = "unavailable"
        z.writestr(f"{root}/reproducibility/git_info.txt", git + "\n")
    print(f"status=ok bundle={out}")


def should_skip(path: Path) -> bool:
    s = path.as_posix()
    return path.suffix in {".pkl", ".pt"} or s.endswith("_visdrone.json") or "/images/" in s or "raw" in path.name.lower()


if __name__ == "__main__":
    main()
