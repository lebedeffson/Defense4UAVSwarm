#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download


PREFERRED_HINTS = (
    "validate",
    "validation",
    "val",
    "test",
    "yaml",
    "json",
    "meta",
    "label",
    "calib",
    "camera",
    "pose",
)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--repo-id", default="fengtt42/U2USim-2")
    p.add_argument("--repo-type", default="dataset")
    p.add_argument("--output-root", default="data/U2UData")
    p.add_argument("--max-files", type=int, default=200)
    p.add_argument("--max-file-mb", type=float, default=512.0)
    p.add_argument("--max-total-gb", type=float, default=12.0)
    p.add_argument("--allow-ext", nargs="+", default=[".json", ".yaml", ".yml", ".txt", ".csv", ".png", ".jpg", ".jpeg", ".pcd", ".bin", ".npy", ".npz"])
    p.add_argument("--include-archives", action="store_true")
    p.add_argument("--extract-rar", action="store_true")
    p.add_argument("--min-free-disk-gb", type=float, default=20.0)
    p.add_argument("--retries", type=int, default=5)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    ensure_free_space(Path(args.output_root), args.min_free_disk_gb)
    api = HfApi(token=token)
    files = api.list_repo_files(args.repo_id, repo_type=args.repo_type, token=token)
    allow_ext = set(args.allow_ext)
    if args.include_archives:
        allow_ext |= {".rar", ".zip", ".tar", ".gz", ".7z"}
    selected = select_files(files, allow_ext, args.max_files)
    infos = {i.path: getattr(i, "size", None) for i in api.get_paths_info(args.repo_id, selected, repo_type=args.repo_type, token=token)}
    selected = cap_total_size(selected, infos, args.max_total_gb)
    total_gb = sum((infos.get(name) or 0) for name in selected) / (1024**3)
    print(f"repo={args.repo_id} files_total={len(files)} selected={len(selected)} selected_gb={total_gb:.2f} dry_run={args.dry_run}")
    for name in selected[:50]:
        print(f"{name}\t{((infos.get(name) or 0) / (1024**3)):.2f} GB")
    if args.dry_run:
        return
    out = Path(args.output_root)
    out.mkdir(parents=True, exist_ok=True)
    downloaded = 0
    skipped = 0
    for name in selected:
        size = infos.get(name)
        if size is not None and size > args.max_file_mb * 1024 * 1024:
            skipped += 1
            print(f"skipped {name}: file exceeds --max-file-mb")
            continue
        local = None
        last_exc: Exception | None = None
        for attempt in range(1, args.retries + 1):
            try:
                local = hf_hub_download(
                    repo_id=args.repo_id,
                    filename=name,
                    repo_type=args.repo_type,
                    token=token,
                    local_dir=out,
                    local_dir_use_symlinks=False,
                    resume_download=True,
                )
                break
            except Exception as exc:
                last_exc = exc
                print(f"retry {attempt}/{args.retries} failed for {name}: {type(exc).__name__}: {exc}")
        if local is None:
            skipped += 1
            print(f"skipped {name}: {type(last_exc).__name__ if last_exc else 'unknown'}: {last_exc}")
            continue
        downloaded += 1
        print(f"downloaded {local}")
        if args.extract_rar and str(local).lower().endswith(".rar"):
            extract_rar(Path(local), out)
    print(f"status=done downloaded={downloaded} skipped={skipped} output={out}")


def select_files(files: list[str], allow_ext: set[str], max_files: int) -> list[str]:
    def score(name: str) -> tuple[int, int, str]:
        lower = name.lower()
        hint = 0 if any(part in lower for part in PREFERRED_HINTS) else 1
        image_or_lidar = 0 if Path(lower).suffix in {".png", ".jpg", ".jpeg", ".pcd", ".bin", ".npy", ".npz"} else 1
        return hint, image_or_lidar, name

    filtered = [f for f in files if Path(f.lower()).suffix in allow_ext and not f.endswith("/")]
    # U2UData-2 stores one RAR per drone. Prefer the first 3 drones for a minimal swarm subset.
    drone_preferred = [
        f for f in filtered
        if any(token in f.lower() for token in ["drone_1.rar", "drone_2.rar", "drone_3.rar"])
    ]
    other = [f for f in filtered if f not in set(drone_preferred)]
    return (sorted(drone_preferred, key=score) + sorted(other, key=score))[:max_files]


def cap_total_size(files: list[str], sizes: dict[str, int | None], max_total_gb: float) -> list[str]:
    total = 0
    selected = []
    limit = max_total_gb * 1024**3
    for name in files:
        size = sizes.get(name) or 0
        if selected and total + size > limit:
            continue
        selected.append(name)
        total += size
    return selected


def ensure_free_space(path: Path, min_free_gb: float) -> None:
    target = path if path.exists() else path.parent
    target.mkdir(parents=True, exist_ok=True)
    free_gb = shutil.disk_usage(target).free / (1024**3)
    if free_gb < min_free_gb:
        raise SystemExit(f"Not enough free disk: {free_gb:.1f} GB < required {min_free_gb:.1f} GB")


def extract_rar(path: Path, output_root: Path) -> None:
    if shutil.which("unrar") is None:
        raise RuntimeError("unrar is not installed; install it or omit --extract-rar")
    dest = output_root / path.stem
    dest.mkdir(parents=True, exist_ok=True)
    subprocess.run(["unrar", "x", "-o+", str(path), str(dest)], check=True)


if __name__ == "__main__":
    main()
