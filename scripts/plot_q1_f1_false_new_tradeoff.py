#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

from defense4uavswarm.q1_visdrone import pareto_flags


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--tracker-summary", required=True)
    p.add_argument("--main-summary", required=True)
    p.add_argument("--confidence-summary", required=True)
    p.add_argument("--output-dir", required=True)
    args = p.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    points = collect_points(Path(args.tracker_summary), Path(args.main_summary), Path(args.confidence_summary))
    points["pareto"] = pareto_flags(points, "F1", "false_new_tracks")
    points.to_csv(out / "tradeoff_points.csv", index=False)
    plot(points, out / "fig_tradeoff_f1_false_new_ru.png", lang="ru")
    plot(points, out / "fig_tradeoff_f1_false_new_en.png", lang="en")
    write_claim_safe(out / "tradeoff_claim_safe.md", points)
    print(f"status=ok points={len(points)} output={out}")


def collect_points(tracker_summary: Path, main_summary: Path, confidence_summary: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    main = pd.read_csv(main_summary) if main_summary.exists() else pd.DataFrame()
    tracker = pd.read_csv(tracker_summary) if tracker_summary.exists() else pd.DataFrame()
    confidence = pd.read_csv(confidence_summary)
    tracker_fallback = tracker_summary.parent.parent / f"{tracker_summary.parent.name}_v9" / tracker_summary.name

    rows += method_rows(main, "bytetrack", "ByteTrack", "tracker")
    rows += method_rows(main, "geometry_dynamic_no_multiagent", "ByteTrack + geometry trust", "trust")
    rows += method_rows(main, "geometry_dynamic_adaptive_balanced", "ByteTrack + adaptive balanced", "trust")
    rows += method_rows(main, "geometry_dynamic_false_new_safe", "ByteTrack + false-new-safe", "trust")
    rows += method_rows(main, "rf_learned_gate", "RF", "supervised")

    rows += tracker_rows(tracker, "ocsort", "none", "OC-SORT", "tracker", tracker_fallback)
    rows += tracker_rows(tracker, "ocsort", "geometry_dynamic_no_multiagent", "OC-SORT + geometry trust", "trust", tracker_fallback)

    for _, r in confidence.iterrows():
        rows.append(
            {
                "label": f"conf >= {float(r['threshold']):.1f}",
                "family": "confidence_threshold",
                "method": "confidence_threshold_baseline",
                "threshold": float(r["threshold"]),
                "F1": float(r["F1"]),
                "false_new_tracks": float(r["false_new_tracks"]),
                "source": str(confidence_summary),
            }
        )
    df = pd.DataFrame(rows)
    df = df.dropna(subset=["F1", "false_new_tracks"]).drop_duplicates(["label", "family"], keep="first")
    return df.reset_index(drop=True)


def method_rows(df: pd.DataFrame, method: str, label: str, family: str) -> list[dict[str, Any]]:
    if df.empty or "method" not in df:
        return []
    hit = df[df["method"].astype(str).eq(method)]
    if hit.empty:
        return []
    r = hit.iloc[0]
    return [
        {
            "label": label,
            "family": family,
            "method": method,
            "threshold": pd.NA,
            "F1": float(r["F1"]),
            "false_new_tracks": float(r["false_new_tracks"]),
            "source": "main_summary",
        }
    ]


def tracker_rows(df: pd.DataFrame, tracker: str, trust_mode: str, label: str, family: str, fallback: Path | None = None) -> list[dict[str, Any]]:
    if df.empty or not {"tracker", "trust_mode"}.issubset(df.columns):
        return []
    ok = df.copy()
    if "available" in ok:
        ok = ok[ok["available"].fillna(False).astype(bool)]
    hit = ok[ok["tracker"].astype(str).eq(tracker) & ok["trust_mode"].astype(str).eq(trust_mode)]
    if hit.empty:
        # Prefer the saved v9 summary when the current import-time run was unavailable.
        if fallback is not None and fallback.exists():
            fb = pd.read_csv(fallback)
            return tracker_rows(fb, tracker, trust_mode, label, family, None)
        return []
    r = hit.iloc[0]
    return [
        {
            "label": label,
            "family": family,
            "method": str(r.get("method", trust_mode)),
            "threshold": pd.NA,
            "F1": float(r["F1"]),
            "false_new_tracks": float(r["false_new_tracks"]),
            "source": "tracker_summary",
        }
    ]


def plot(points: pd.DataFrame, path: Path, lang: str) -> None:
    colors = {
        "tracker": "#4c78a8",
        "trust": "#d62728",
        "supervised": "#7f7f7f",
        "confidence_threshold": "#2ca02c",
    }
    markers = {
        "tracker": "o",
        "trust": "D",
        "supervised": "s",
        "confidence_threshold": "^",
    }
    plt.figure(figsize=(8.8, 5.4))
    for family, g in points.groupby("family", sort=False):
        plt.scatter(
            g["false_new_tracks"],
            g["F1"],
            s=90 if family != "confidence_threshold" else 65,
            label=family_label(family, lang),
            color=colors.get(family, "#333333"),
            marker=markers.get(family, "o"),
            alpha=0.9,
            edgecolor="white",
            linewidth=0.6,
        )
        for _, r in g.iterrows():
            text = short_label(str(r["label"]))
            plt.annotate(text, (r["false_new_tracks"], r["F1"]), xytext=(5, 5), textcoords="offset points", fontsize=7)
    title = "Компромисс F1 / ложные новые треки" if lang == "ru" else "F1 / false-new track trade-off"
    xlabel = "Ложные новые треки" if lang == "ru" else "False-new tracks"
    ylabel = "F1"
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.25)
    plt.legend(loc="best", fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()


def family_label(family: str, lang: str) -> str:
    labels = {
        "ru": {
            "tracker": "трекер",
            "trust": "trust layer",
            "supervised": "supervised baseline",
            "confidence_threshold": "порог уверенности",
        },
        "en": {
            "tracker": "tracker",
            "trust": "trust layer",
            "supervised": "supervised baseline",
            "confidence_threshold": "confidence threshold",
        },
    }
    return labels[lang].get(family, family)


def short_label(label: str) -> str:
    replacements = {
        "ByteTrack + geometry trust": "BT+trust",
        "ByteTrack + adaptive balanced": "BT+adapt",
        "ByteTrack + false-new-safe": "BT+safe",
        "OC-SORT + geometry trust": "OC+trust",
        "ByteTrack": "BT",
        "OC-SORT": "OC",
    }
    for old, new in replacements.items():
        label = label.replace(old, new)
    return label


def write_claim_safe(path: Path, points: pd.DataFrame) -> None:
    bt = one(points, "ByteTrack")
    trust = one(points, "ByteTrack + geometry trust")
    conf = points[points["family"].eq("confidence_threshold")].copy()
    safe = one(points, "ByteTrack + false-new-safe")
    recommended = trust if trust is not None else safe
    lines = [
        "# F1 / False-New Trade-off Claim-Safe Notes",
        "",
        "Points:",
        "",
        "```text",
        points[["label", "family", "F1", "false_new_tracks", "pareto"]].to_string(index=False),
        "```",
        "",
    ]
    if bt is not None and trust is not None:
        dominates_bt = trust["F1"] >= bt["F1"] and trust["false_new_tracks"] <= bt["false_new_tracks"] and (
            trust["F1"] > bt["F1"] or trust["false_new_tracks"] < bt["false_new_tracks"]
        )
        lines += [
            "1. Does the trust layer Pareto-dominate ByteTrack?",
            "Yes." if dominates_bt else "No. It reduces false-new tracks but loses F1 relative to ByteTrack.",
        ]
    if trust is not None and not conf.empty:
        conf_dominates_trust = conf[(conf["F1"] >= trust["F1"]) & (conf["false_new_tracks"] <= trust["false_new_tracks"])]
        trust_dominates_conf = conf[(conf["F1"] <= trust["F1"]) & (conf["false_new_tracks"] >= trust["false_new_tracks"])]
        closest = conf.iloc[(conf["F1"] - trust["F1"]).abs().argsort().iloc[0]]
        lines += [
            "2. Does the trust layer dominate the confidence-threshold baseline?",
            "Yes for all tested confidence thresholds." if len(trust_dominates_conf) == len(conf) else "No / partial; inspect the threshold points.",
            "3. Best point for the article:",
            f"{recommended['label']} (F1={recommended['F1']:.6f}, false_new={recommended['false_new_tracks']:.0f})." if recommended is not None else "Unavailable.",
            "4. Most conservative point:",
            f"{safe['label']} (F1={safe['F1']:.6f}, false_new={safe['false_new_tracks']:.0f})." if safe is not None else "Lowest false_new point in the figure.",
            "5. Safe claim:",
            safe_tradeoff_claim(trust, conf, closest, len(conf_dominates_trust) > 0),
            "6. Forbidden claim:",
            "Do not claim universal Pareto dominance or simultaneous F1/false-new improvement unless all plotted baselines support it.",
        ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def one(points: pd.DataFrame, label: str) -> dict[str, Any] | None:
    hit = points[points["label"].eq(label)]
    if hit.empty:
        return None
    return hit.iloc[0].to_dict()


def safe_tradeoff_claim(trust: dict[str, Any], conf: pd.DataFrame, closest: pd.Series, conf_dominates_trust: bool) -> str:
    if conf_dominates_trust:
        return "Fixed confidence threshold explains a substantial part of the effect; trust should be claimed as interpretable multi-feature filtering, not numerically dominant."
    if float(closest["false_new_tracks"]) > float(trust["false_new_tracks"]):
        return "Trust shifts the operating point toward fewer false-new tracks at a small F1 cost compared with ByteTrack and is not matched by the closest-F1 confidence threshold."
    return "Trust and confidence threshold occupy overlapping trade-off regions; claim only an interpretable trade-off, not universal superiority."


if __name__ == "__main__":
    main()
