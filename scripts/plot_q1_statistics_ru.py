#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def load_sequence_stats(statistics_dir: Path) -> pd.DataFrame:
    candidates = [
        statistics_dir / "sequence_deltas.csv",
        statistics_dir / "sequence_level_deltas.csv",
        statistics_dir.parent / "tracker_comparison_yolov8s" / "tracker_comparison_by_sequence.csv",
    ]
    for path in candidates:
        if path.exists():
            return pd.read_csv(path)
    raise FileNotFoundError("No sequence-level statistics found for Russian plots")


def select_byte_rows(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if "tracker" in df.columns and "trust_mode" in df.columns:
        byte = df[df["tracker"].astype(str).str.contains("byte", case=False, na=False)]
        base = byte[byte["trust_mode"].eq("none")]
        trust = byte[byte["trust_mode"].eq("geometry_dynamic_no_multiagent")]
        if not base.empty and not trust.empty:
            return base, trust
    if "method" in df.columns:
        base = df[df["method"].astype(str).str.contains("bytetrack", case=False, na=False)]
        trust = df[df["method"].astype(str).str.contains("trust|geometry", case=False, na=False)]
        if not base.empty and not trust.empty:
            return base, trust
    raise ValueError("Cannot identify ByteTrack baseline and trust rows")


def aligned(base: pd.DataFrame, trust: pd.DataFrame, value: str) -> pd.DataFrame:
    cols = ["sequence_id", value]
    a = base[cols].rename(columns={value: "baseline"})
    b = trust[cols].rename(columns={value: "trust"})
    return a.merge(b, on="sequence_id", how="inner")


def boxplot_pair(data: pd.DataFrame, value: str, ylabel: str, title: str, output: Path) -> None:
    plt.figure(figsize=(6, 4), dpi=160)
    plt.boxplot([data["baseline"].astype(float), data["trust"].astype(float)], tick_labels=["Без слоя", "Со слоем"], showmeans=True)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(output)
    plt.close()


def effect_plot(data: pd.DataFrame, output: Path) -> None:
    d = data.copy()
    d["delta"] = d["trust"].astype(float) - d["baseline"].astype(float)
    d = d.sort_values("delta")
    plt.figure(figsize=(7, 4), dpi=160)
    plt.bar(d["sequence_id"].astype(str), d["delta"].astype(float), color="#4b77be")
    plt.axhline(0, color="black", linewidth=0.8)
    plt.xticks(rotation=35, ha="right", fontsize=8)
    plt.ylabel("Изменение ложных новых треков")
    plt.title("Эффект доверительного слоя по последовательностям")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(output)
    plt.close()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--statistics-dir", required=True)
    p.add_argument("--output-dir", required=True)
    args = p.parse_args()
    stats_dir = Path(args.statistics_dir)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = load_sequence_stats(stats_dir)
    base, trust = select_byte_rows(df)
    f1 = aligned(base, trust, "F1")
    false_new = aligned(base, trust, "false_new_tracks")
    boxplot_pair(f1, "F1", "F1", "F1 по последовательностям VisDrone", out / "fig_boxplot_f1_yolov8s_ru.png")
    boxplot_pair(false_new, "false_new_tracks", "Ложные новые треки", "Ложные новые треки по последовательностям VisDrone", out / "fig_boxplot_false_new_yolov8s_ru.png")
    effect_plot(false_new, out / "fig_effect_size_false_new_yolov8s_ru.png")
    (out / "statistics_ru_note.md").write_text(
        "# Русские версии статистических графиков\n\n"
        "Графики показывают распределение F1 и ложных новых треков по последовательностям. "
        "Их следует интерпретировать вместе с Holm-corrected p-value; отсутствие значимости после поправки не скрывается.\n",
        encoding="utf-8",
    )
    print(f"status=ok output={out}")


if __name__ == "__main__":
    main()
