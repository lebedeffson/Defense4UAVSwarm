#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import os
from collections import Counter
from pathlib import Path

from huggingface_hub import HfApi


KEYWORDS = (
    "validate",
    "validation",
    "val",
    "test",
    "train",
    "chunk",
    "label",
    "labels",
    "annotation",
    "annotations",
    "yaml",
    "json",
    "pkl",
    "opencood",
    "gt",
    "bbox",
    "box",
)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--repo-id", required=True)
    p.add_argument("--repo-type", default="dataset")
    p.add_argument("--output", required=True)
    p.add_argument("--report", required=True)
    args = p.parse_args()

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    api = HfApi(token=token)
    paths = sorted(api.list_repo_files(args.repo_id, repo_type=args.repo_type, token=token))
    infos = {}
    # get_paths_info handles file lists, but batching avoids large-request failures.
    for i in range(0, len(paths), 100):
        for info in api.get_paths_info(args.repo_id, paths[i : i + 100], repo_type=args.repo_type, token=token):
            infos[info.path] = getattr(info, "size", None)

    rows = []
    for path in paths:
        lower = path.lower()
        suffix = "".join(Path(lower).suffixes[-2:]) if lower.endswith((".tar.gz", ".zip.part")) else Path(lower).suffix
        hits = [kw for kw in KEYWORDS if kw in lower]
        rows.append(
            {
                "path": path,
                "size_bytes": infos.get(path) or "",
                "size_gb": f"{((infos.get(path) or 0) / (1024**3)):.4f}" if infos.get(path) is not None else "",
                "suffix": suffix,
                "keyword_hits": ";".join(hits),
                "likely_relevant": bool(hits),
            }
        )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["path", "size_bytes", "size_gb", "suffix", "keyword_hits", "likely_relevant"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    write_report(Path(args.report), args.repo_id, args.repo_type, rows)
    total_gb = sum(int(r["size_bytes"] or 0) for r in rows) / (1024**3)
    relevant = [r for r in rows if r["likely_relevant"]]
    print(f"repo={args.repo_id} files={len(rows)} relevant={len(relevant)} total_gb={total_gb:.2f}")


def write_report(path: Path, repo_id: str, repo_type: str, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix_counts = Counter(str(r["suffix"]) or "<none>" for r in rows)
    relevant = [r for r in rows if r["likely_relevant"]]
    total_gb = sum(int(r["size_bytes"] or 0) for r in rows) / (1024**3)
    relevant_gb = sum(int(r["size_bytes"] or 0) for r in relevant) / (1024**3)
    top = sorted(relevant, key=lambda r: int(r["size_bytes"] or 0), reverse=True)[:40]
    lines = [
        "# Hugging Face Repository Inventory",
        "",
        f"- repo_id: `{repo_id}`",
        f"- repo_type: `{repo_type}`",
        f"- files_total: {len(rows)}",
        f"- files_relevant_by_keyword: {len(relevant)}",
        f"- total_size_gb: {total_gb:.2f}",
        f"- relevant_size_gb: {relevant_gb:.2f}",
        "",
        "## Suffix counts",
        "",
        "```text",
    ]
    lines.extend(f"{suffix}: {count}" for suffix, count in sorted(suffix_counts.items()))
    lines.extend(["```", "", "## Largest relevant files", "", "```text"])
    lines.extend(f"{r['size_gb']} GB\t{r['path']}\t{r['keyword_hits']}" for r in top)
    lines.extend(["```", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
