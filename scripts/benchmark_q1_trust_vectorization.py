#!/usr/bin/env python
from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--detections", required=True)
    p.add_argument("--feature-audit", default="outputs/results/q1_final_corrected/yolov8s_main/feature_audit.csv")
    p.add_argument("--frames", type=int, default=1000)
    p.add_argument("--repeats", type=int, default=10)
    p.add_argument("--output-dir", required=True)
    args = p.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    det = pd.read_csv(args.feature_audit)
    keep = det[["sequence_id", "frame_id"]].drop_duplicates().head(args.frames)
    det = det.merge(keep, on=["sequence_id", "frame_id"], how="inner").reset_index(drop=True)
    rows = []
    for repeat in range(args.repeats):
        for method, fn in [
            ("current_loop_python", loop_python),
            ("vectorized_numpy", vectorized_numpy),
            ("cached_features_vectorized", cached_vectorized),
        ]:
            start = time.perf_counter()
            accepted = fn(det)
            elapsed = time.perf_counter() - start
            rows.append(
                {
                    "method": method,
                    "repeat": repeat,
                    "n_rows": len(det),
                    "n_frames": len(keep),
                    "accepted": int(np.asarray(accepted).sum()),
                    "runtime_ms_total": elapsed * 1000,
                    "runtime_ms_per_frame": elapsed * 1000 / max(1, len(keep)),
                }
            )
    raw = pd.DataFrame(rows)
    raw.to_csv(out / "runtime_vectorization_raw.csv", index=False)
    summary = raw.groupby("method", as_index=False)["runtime_ms_per_frame"].agg(["mean", "std", "min"]).reset_index()
    summary.to_csv(out / "runtime_vectorization_summary.csv", index=False)
    plot_runtime(summary, out / "fig_runtime_vectorization_ru.png")
    write_claim(out / "runtime_vectorization_claim_safe.md", summary)
    print(f"status=ok output={out}")


def loop_python(det: pd.DataFrame) -> list[bool]:
    out = []
    for row in det.itertuples():
        c = float(row.confidence)
        k = float(getattr(row, "k_i", 0.35))
        age = float(getattr(row, "temporal_age", 0))
        conf_rw = c * (0.6 + 0.4 * max(k, min(1.0, age / 3.0)))
        out.append(bool((conf_rw >= 0.45) or (age >= 2 and c >= 0.10)))
    return out


def vectorized_numpy(det: pd.DataFrame) -> np.ndarray:
    c = det["confidence"].to_numpy(dtype=float)
    k = det["k_i"].to_numpy(dtype=float)
    age = det["temporal_age"].to_numpy(dtype=float)
    q = np.maximum(k, np.clip(age / 3.0, 0, 1))
    conf_rw = c * (0.6 + 0.4 * q)
    return (conf_rw >= 0.45) | ((age >= 2) & (c >= 0.10))


def cached_vectorized(det: pd.DataFrame) -> np.ndarray:
    c = det["confidence"].to_numpy(dtype=float)
    q = np.maximum(det["k_i"].to_numpy(dtype=float), np.clip(det["temporal_age"].to_numpy(dtype=float) / 3.0, 0, 1))
    return (c * (0.6 + 0.4 * q) >= 0.45) | ((det["temporal_age"].to_numpy(dtype=float) >= 2) & (c >= 0.10))


def plot_runtime(summary: pd.DataFrame, path: Path) -> None:
    plt.figure(figsize=(7, 4))
    plt.bar(summary["method"], summary["mean"])
    plt.ylabel("ms/frame")
    plt.xticks(rotation=25, ha="right")
    plt.title("Runtime overhead: Python loop vs vectorized")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def write_claim(path: Path, summary: pd.DataFrame) -> None:
    cur = val(summary, "current_loop_python")
    vec = val(summary, "vectorized_numpy")
    cached = val(summary, "cached_features_vectorized")
    speedup = cur / vec if vec and vec > 0 else float("nan")
    lines = [
        "# Runtime Vectorization Claim-Safe Notes",
        "",
        "This benchmark isolates trust aggregation on saved features. It is not detector runtime.",
        "",
        "```text",
        summary.to_string(index=False),
        "```",
        "",
        f"current_loop_python_ms: {cur:.6f}",
        f"vectorized_numpy_ms: {vec:.6f}",
        f"cached_vectorized_ms: {cached:.6f}",
        f"speedup_loop_to_vectorized: {speedup:.2f}x",
        "Safe claim: the measured Python loop overhead is implementation-dependent; vectorization gives the measured reduction above.",
        "Forbidden claim: do not claim embedded/CUDA runtime or full detector pipeline speed from this microbenchmark.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def val(summary: pd.DataFrame, method: str) -> float:
    hit = summary[summary["method"].eq(method)]
    return float(hit.iloc[0]["mean"]) if len(hit) else float("nan")


if __name__ == "__main__":
    main()
