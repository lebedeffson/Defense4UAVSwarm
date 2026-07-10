#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import pandas as pd


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def table(paths: list[Path]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"path": p.as_posix(), "sha256": sha256(p), "bytes": p.stat().st_size}
            for p in sorted(paths)
            if p.exists() and p.is_file()
        ]
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--results-root", default="outputs/results/q1_v542")
    p.add_argument("--output-dir", default="outputs/results/q1_v542")
    args = p.parse_args()
    root = Path(args.results_root)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    input_paths = [
        Path("configs/q1_v54/bytetrack.yaml"),
        Path("configs/q1_v54/ocsort.yaml"),
        Path("data_manifests/visdrone_eval_frames.csv"),
        Path("outputs/results/q1_final_corrected/yolov8s_main/feature_audit.csv"),
    ]
    output_paths = [p for p in root.rglob("*") if p.is_file() and p.name not in {"input_checksums.csv", "output_checksums.csv"}]
    table(input_paths).to_csv(out / "input_checksums.csv", index=False)
    table(output_paths).to_csv(out / "output_checksums.csv", index=False)
    print(f"status=ok inputs={out/'input_checksums.csv'} outputs={out/'output_checksums.csv'}")


if __name__ == "__main__":
    main()
