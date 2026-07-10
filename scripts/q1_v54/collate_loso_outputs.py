#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--inputs", nargs="+", default=["outputs/results/q1_v542/loso/bytetrack", "outputs/results/q1_v542/loso/ocsort"])
    p.add_argument("--output-dir", default="outputs/results/q1_v542/loso")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if args.dry_run:
        print(f"dry_run=ok inputs={len(args.inputs)} output={out}")
        return
    frames = []
    selections = []
    folds_payload = []
    summaries = []
    for item in args.inputs:
        root = Path(item)
        if not root.exists():
            continue
        if (root / "outer_test_by_sequence.csv").exists():
            frames.append(pd.read_csv(root / "outer_test_by_sequence.csv"))
        if (root / "inner_selection_results.csv").exists():
            selections.append(pd.read_csv(root / "inner_selection_results.csv"))
        if (root / "outer_test_summary.csv").exists():
            summaries.append(pd.read_csv(root / "outer_test_summary.csv"))
        if (root / "outer_folds.json").exists():
            folds_payload.extend(json.loads((root / "outer_folds.json").read_text(encoding="utf-8")))
    if frames:
        pd.concat(frames, ignore_index=True).to_csv(out / "outer_test_by_sequence.csv", index=False)
    if selections:
        pd.concat(selections, ignore_index=True).to_csv(out / "inner_selection_results.csv", index=False)
        (out / "selected_configs_by_fold.json").write_text(json.dumps(pd.concat(selections, ignore_index=True).to_dict(orient="records"), indent=2), encoding="utf-8")
    if summaries:
        pd.concat(summaries, ignore_index=True).to_csv(out / "outer_test_summary.csv", index=False)
    (out / "outer_folds.json").write_text(json.dumps(folds_payload, indent=2), encoding="utf-8")
    print(f"status=ok output={out} rows={sum(len(x) for x in frames) if frames else 0}")


if __name__ == "__main__":
    main()
