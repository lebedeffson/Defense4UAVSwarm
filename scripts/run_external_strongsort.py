#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--mot-root", required=True)
    p.add_argument("--dataset-root", required=True)
    p.add_argument("--output-dir", required=True)
    args = p.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    try:
        from boxmot.trackers.bbox.strongsort.strongsort import StrongSort  # noqa: F401

        import_status = "ok"
    except Exception as exc:  # pragma: no cover - depends on external tracker env
        import_status = f"failed: {exc}"
    report = "\n".join(
        [
            "# StrongSORT Run Status",
            "",
            f"- import: {import_status}",
            "- run status: skipped",
            "- reason: StrongSORT requires a ReID model or externally supplied appearance embeddings for a valid run.",
            "- ReID weights downloaded: no",
            "- policy: external model weights were not downloaded because the task explicitly requires stopping before additional ReID downloads.",
            "",
        ]
    )
    (out / "strongsort_unavailable.md").write_text(report, encoding="utf-8")
    (out / "strongsort_run_summary.csv").write_text(
        "tracker,available,reason,reid_weights_downloaded\n"
        "strongsort,false,requires_reid_model_or_embeddings,no\n",
        encoding="utf-8",
    )
    print(f"status=unavailable output={out / 'strongsort_unavailable.md'}")


if __name__ == "__main__":
    main()
