#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import re
import shutil
import subprocess
from pathlib import Path


DEFAULT_INCLUDE = [
    "*.json",
    "*.yaml",
    "*.yml",
    "*.txt",
    "*.csv",
    "*.png",
    "*.jpg",
    "*.jpeg",
    "*.pcd",
    "*.bin",
    "*.npy",
    "*.npz",
]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--archive-root", default="data/U2UData")
    p.add_argument("--output-root", default="data/U2UData_extracted_minimal")
    p.add_argument("--max-frames", type=int, default=300)
    p.add_argument("--include", nargs="+", default=DEFAULT_INCLUDE)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    if shutil.which("unrar") is None:
        raise SystemExit("unrar is required")
    archives = sorted(Path(args.archive_root).glob("*.rar"))
    if not archives:
        raise SystemExit(f"No .rar archives found in {args.archive_root}")
    out = Path(args.output_root)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for archive in archives:
        names = list_archive(archive)
        selected = select_subset(names, args.max_frames, args.include)
        rows.append({"archive": archive.name, "files_total": len(names), "files_selected": len(selected), "status": "dry_run" if args.dry_run else "extracted"})
        print(f"{archive.name}: total={len(names)} selected={len(selected)}")
        if args.dry_run:
            for name in selected[:20]:
                print(f"  {name}")
            continue
        list_file = out / f"{archive.stem}_extract_list.txt"
        list_file.write_text("\n".join(selected) + "\n", encoding="utf-8")
        subprocess.run(["unrar", "x", "-o+", f"@{list_file}", str(archive), str(out / archive.stem)], check=True)
    with (out / "extraction_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["archive", "files_total", "files_selected", "status"])
        writer.writeheader()
        writer.writerows(rows)


def list_archive(path: Path) -> list[str]:
    result = subprocess.run(["unrar", "lb", str(path)], check=True, capture_output=True, text=True)
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def select_subset(names: list[str], max_frames: int, include_patterns: list[str]) -> list[str]:
    selected = []
    seen_frames: set[int] = set()
    for name in names:
        lower = name.lower()
        if not matches_any(lower, include_patterns):
            continue
        frame_id = infer_frame_id(lower)
        if frame_id is not None:
            if frame_id >= max_frames:
                continue
            seen_frames.add(frame_id)
        selected.append(name)
    return selected


def matches_any(name: str, patterns: list[str]) -> bool:
    suffix = Path(name).suffix.lower()
    return any(pattern.startswith("*.") and suffix == pattern[1:].lower() for pattern in patterns)


def infer_frame_id(name: str) -> int | None:
    nums = re.findall(r"\d+", Path(name).stem)
    if not nums:
        return None
    return int(nums[-1])


if __name__ == "__main__":
    main()
