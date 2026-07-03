#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from defense4uavswarm.tracking.simple import iou


METRICS = [
    "TP",
    "FP",
    "FN",
    "precision",
    "recall",
    "F1",
    "IDF1",
    "IDSW",
    "track_breaks",
    "MOTA",
    "failures_per_100_frames",
    "FP_per_100_frames",
    "FN_per_100_frames",
    "IDSW_per_100_frames",
    "track_breaks_per_100_frames",
]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--primary-dir", required=True)
    p.add_argument("--calibration-dir", default="outputs/results/vid_calibration_v17")
    p.add_argument("--figures-dir", default="outputs/figures/vid_holdout_v18_primary")
    p.add_argument("--robustness-dir", default=None)
    p.add_argument("--config", default=None)
    p.add_argument("--split-config", default=None)
    p.add_argument("--split", default="holdout")
    args = p.parse_args()
    primary = Path(args.primary_dir)
    figures = Path(args.figures_dir)
    figures.mkdir(parents=True, exist_ok=True)
    make_primary(primary, Path(args.calibration_dir), figures)
    if args.robustness_dir:
        make_robustness(Path(args.robustness_dir), args.config, args.split_config, args.split)


def make_primary(primary: Path, calibration: Path, figures: Path) -> None:
    frame = with_rates(pd.read_csv(primary / "summary_metrics_tracking_selected.csv"), infer_frames(primary))
    comp = frame[["scenario", *[m for m in METRICS if m in frame.columns]]].copy()
    comp.to_csv(primary / "holdout_comparison_summary.csv", index=False)
    s1 = one(frame, "S1")
    s3 = one(frame, "S3_safe_recovery")
    deltas = []
    for metric in METRICS:
        if metric not in frame.columns:
            continue
        a, b = float(s1.get(metric, 0)), float(s3.get(metric, 0))
        deltas.append({"metric": metric, "S1": a, "S3_safe_recovery": b, "delta_abs": b - a, "delta_percent": (b - a) / abs(a) if a else 0.0})
    pd.DataFrame(deltas).to_csv(primary / "holdout_delta_vs_s1.csv", index=False)
    cal = with_rates(pd.read_csv(calibration / "summary_metrics_tracking_selected.csv"))
    quality = recovery_quality(primary)
    table = []
    for split, source in [("calibration", cal), ("holdout", frame)]:
        for scenario in ["S1", "S3_safe_recovery"]:
            row = one(source, scenario).to_dict()
            row["split"] = split
            row["recovered_TP_rate"] = None if scenario == "S1" else quality.get("recovered_TP_rate")
            row["recovered_FP_rate"] = None if scenario == "S1" else quality.get("recovered_FP_rate")
            table.append(row)
    pd.DataFrame(table)[["split", "scenario", "TP", "FP", "FN", "F1", "IDF1", "IDSW", "track_breaks", "recovered_TP_rate", "recovered_FP_rate"]].to_csv(primary / "table_calibration_holdout.csv", index=False)
    main = comp[comp["scenario"].isin(["S1", "S3_safe_recovery"])].copy()
    main["Interpretation"] = main["scenario"].map({"S1": "attack baseline", "S3_safe_recovery": "recovery-based defense"})
    main.rename(columns={"scenario": "Scenario", "track_breaks": "Track breaks"}).to_csv(primary / "table_main_vid_result.csv", index=False)
    pd.DataFrame([quality]).to_csv(primary / "table_recovery_quality.csv", index=False)
    plot_bar(frame, "IDF1", figures / "idf1_comparison.png")
    plot_bar(frame, "IDSW", figures / "idsw_comparison.png")
    plot_bar(frame, "track_breaks", figures / "track_breaks_comparison.png")
    plot_fp_fn(frame, figures / "fp_fn_tradeoff.png")
    plot_quality(quality, figures / "recovery_quality.png")


def make_robustness(path: Path, config: str | None = None, split_config: str | None = None, split: str = "holdout") -> None:
    frame = recompute_robustness(path, config, split_config, split) if config and split_config else pd.read_csv(path / "summary_metrics.csv")
    if "class_group" in frame.columns:
        frame = frame[frame["class_group"] == "all"]
    keep = ["model_name", "eps", "scenario", "F1", "IDF1", "IDSW", "track_breaks", "FP", "FN", "recovered_TP_rate", "recovered_FP_rate"]
    out = frame[[c for c in keep if c in frame.columns]].copy()
    out.to_csv(path / "robustness_summary.csv", index=False)
    rows = []
    for (model, eps), group in frame.groupby(["model_name", "eps"]):
        if "S1" not in set(group["scenario"]) or "S3_safe_recovery" not in set(group["scenario"]):
            continue
        s1, s3 = one(group, "S1"), one(group, "S3_safe_recovery")
        fp_delta = float(s3.FP - s1.FP) / max(1.0, float(s1.FP))
        rows.append(
            {
                "model_name": model,
                "eps": eps,
                "IDF1_delta": s3.IDF1 - s1.IDF1,
                "IDSW_delta": s3.IDSW - s1.IDSW,
                "track_break_delta": s3.track_breaks - s1.track_breaks,
                "FP_delta_percent": fp_delta,
                "FN_delta": s3.FN - s1.FN,
                "success_flag": (s3.IDSW - s1.IDSW) < 0 and (s3.track_breaks - s1.track_breaks) <= 0 and fp_delta <= 0.15,
            }
        )
    pd.DataFrame(rows).to_csv(path / "robustness_delta_vs_s1.csv", index=False)


def recompute_robustness(path: Path, config: str, split_config: str, split: str) -> pd.DataFrame:
    from defense4uavswarm.config import load_config
    from defense4uavswarm.matrix import load_split_sequences, task_dataset
    from defense4uavswarm.pipeline import evaluate

    cfg = load_config(config)
    sequences = load_split_sequences(split_config, split)
    _, ds = task_dataset(cfg, "vid")
    gt = ds.all_annotations(sequences)
    rows = []
    for s1_path in sorted(path.glob("vid_*_s1_fgsm_eps_*.csv")):
        stem = s1_path.stem
        model = stem.split("_s1_fgsm_eps_")[0].replace("vid_", "")
        eps = float(stem.split("_eps_")[1])
        s0 = pd.read_csv(path / f"vid_{model}_s0_baseline.csv")
        s1 = pd.read_csv(s1_path)
        s3_path = path / f"vid_{model}_s3_safe_recovery_eps_{eps}.csv"
        if not s3_path.exists():
            continue
        s3 = pd.read_csv(s3_path)
        for scenario, pred in [("S0", s0), ("S1", s1), ("S3_safe_recovery", s3)]:
            metric_eps = 0.0 if scenario == "S0" else eps
            row = evaluate(gt, pred, scenario, metric_eps, cfg=cfg, class_group="all", include_map=False, include_tracking=True)
            row.update({"model_name": model, "eps": metric_eps, "class_group": "all"})
            if scenario == "S3_safe_recovery":
                q = recovered_quality_from_frame(gt, pred)
                row.update(q)
            rows.append(row)
    return pd.DataFrame(rows)


def recovered_quality_from_frame(gt: pd.DataFrame, pred: pd.DataFrame) -> dict:
    if "is_recovered" not in pred:
        return {"recovered_TP_rate": 0.0, "recovered_FP_rate": 0.0}
    rec = pred[pred["is_recovered"].astype(str).str.lower().isin(["true", "1"])]
    gt_frames = {k: v for k, v in gt.groupby(["sequence_id", "frame_id"])}
    tp = 0
    for row in rec.itertuples(index=False):
        gt_f = gt_frames.get((row.sequence_id, row.frame_id), gt.iloc[0:0])
        best = 0.0
        for g in gt_f.itertuples(index=False):
            best = max(best, iou((row.x1, row.y1, row.x2, row.y2), (g.x1, g.y1, g.x2, g.y2)))
        tp += int(best >= 0.5)
    total = len(rec)
    fp = total - tp
    return {"recovered_TP_rate": tp / max(1, total), "recovered_FP_rate": fp / max(1, total)}


def with_rates(frame: pd.DataFrame, frames: int | None = None) -> pd.DataFrame:
    if frames is None:
        frames = max(1, int(frame.get("num_frames", pd.Series([1])).dropna().iloc[0]))
    frame = frame.copy()
    frame["failures_per_100_frames"] = (frame["FP"] + frame["FN"] + frame["IDSW"].fillna(0) + frame["track_breaks"].fillna(0)) / frames * 100.0
    frame["FP_per_100_frames"] = frame["FP"] / frames * 100.0
    frame["FN_per_100_frames"] = frame["FN"] / frames * 100.0
    frame["IDSW_per_100_frames"] = frame["IDSW"].fillna(0) / frames * 100.0
    frame["track_breaks_per_100_frames"] = frame["track_breaks"].fillna(0) / frames * 100.0
    return frame


def infer_frames(path: Path) -> int:
    for candidate in sorted(path.glob("vid_*_s1_fgsm_eps_*.csv")):
        frame = pd.read_csv(candidate, usecols=["sequence_id", "frame_id"])
        return max(1, len(frame.drop_duplicates()))
    return 1


def one(frame: pd.DataFrame, scenario: str) -> pd.Series:
    sub = frame[frame["scenario"] == scenario]
    if "class_group" in sub.columns:
        sub = sub[sub["class_group"] == "all"]
    return sub.iloc[0]


def recovery_quality(path: Path) -> dict:
    summary = pd.read_csv(path / "recovery_safety_summary.csv")
    row = summary[summary.get("selected", False) == True].iloc[0] if "selected" in summary.columns and summary["selected"].any() else summary.iloc[0]
    return {
        "recovery_mode": row.get("recovery_mode"),
        "horizon": row.get("recovery_horizon"),
        "decay": row.get("decay"),
        "recovered boxes": row.get("num_recovered_boxes"),
        "recovered TP": row.get("recovered_TP"),
        "recovered FP": row.get("recovered_FP"),
        "recovered_TP_rate": row.get("recovered_TP_rate"),
        "recovered_FP_rate": row.get("recovered_FP_rate"),
    }


def plot_bar(frame: pd.DataFrame, metric: str, path: Path) -> None:
    import matplotlib.pyplot as plt

    sub = frame[frame["scenario"].isin(["S1", "S3_safe_recovery"])]
    plt.figure(figsize=(4, 3))
    plt.bar(sub["scenario"], sub[metric])
    plt.title(metric)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def plot_fp_fn(frame: pd.DataFrame, path: Path) -> None:
    import matplotlib.pyplot as plt

    sub = frame[frame["scenario"].isin(["S1", "S3_safe_recovery"])]
    x = range(len(sub))
    plt.figure(figsize=(5, 3))
    plt.bar([i - 0.2 for i in x], sub["FP"], width=0.4, label="FP")
    plt.bar([i + 0.2 for i in x], sub["FN"], width=0.4, label="FN")
    plt.xticks(list(x), sub["scenario"])
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def plot_quality(quality: dict, path: Path) -> None:
    import matplotlib.pyplot as plt

    plt.figure(figsize=(4, 3))
    plt.bar(["TP rate", "FP rate"], [quality["recovered_TP_rate"], quality["recovered_FP_rate"]])
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


if __name__ == "__main__":
    main()
