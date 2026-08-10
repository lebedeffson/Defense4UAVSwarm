#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from defense4uavswarm.q1_v5.evaluation_matching import evaluate_gate_result
from defense4uavswarm.q1_v5.frame_manifest import add_gt_presence, build_visdrone_frame_manifest, merge_manifest_dimensions
from defense4uavswarm.q1_v5.initiation_gate import assign_episode_ids, split_duplicate_observation_tracklets, tracker_baseline_gate_result
from defense4uavswarm.q1_v5.selective_quarantine import SelectiveTrustQuarantineConfig, selective_quarantine_gate_result
from defense4uavswarm.q1_visdrone import load_gt_protocol
from scripts.q1_v54.run_corrected_operating_curves import gate_config, load_candidates, matching_config


REPO_ROOT = Path(__file__).resolve().parents[2]
KEY_COLS = ["sequence_id", "tracklet_id", "episode_id"]
ROW_KEY_COLS = ["sequence_id", "tracklet_id", "episode_id", "frame_id", "det_id"]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", default="outputs/q1_practical_closure_v7/lifecycle")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()
    out = Path(args.output_dir)
    if out.exists() and args.overwrite:
        shutil.rmtree(out)
    if out.exists() and any(out.iterdir()) and not args.overwrite:
        raise SystemExit(f"Output exists; use --overwrite: {out}")
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    for item in tracker_inputs():
        rows.extend(process_tracker(item))
    lifecycle = pd.DataFrame(rows)
    lifecycle.to_csv(out / "false_episode_lifecycle.csv", index=False)
    summary = summarize(lifecycle)
    summary.to_csv(out / "false_episode_lifecycle_summary.csv", index=False)
    write_plots(lifecycle, out)
    manifest = {
        "status": "success",
        "git_commit": git(["rev-parse", "HEAD"]),
        "git_branch": git(["branch", "--show-current"]),
        "trackers": sorted(lifecycle["tracker"].unique().tolist()) if not lifecycle.empty else [],
        "modes": sorted(lifecycle["mode"].unique().tolist()) if not lifecycle.empty else [],
        "false_episode_rows": int(len(lifecycle)),
        "observed_frame_count_rule": "nunique(frame_id)",
        "planner_claim": "not_modeled",
    }
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(f"status=ok output={out} rows={len(lifecycle)}")


def tracker_inputs() -> list[dict[str, str]]:
    return [
        {
            "tracker": "ByteTrack",
            "config": "configs/q1_v54/bytetrack.yaml",
            "selection": "outputs/results/q1_selective_quarantine_v21/loso/bytetrack/selected_configs_by_fold.json",
        },
        {
            "tracker": "OC-SORT",
            "config": "configs/q1_v54/ocsort.yaml",
            "selection": "outputs/results/q1_selective_quarantine_v21/loso/ocsort/selected_configs_by_fold.json",
        },
        {
            "tracker": "SORT",
            "config": "configs/q1_v54/sort.yaml",
            "selection": "outputs/q1_practical_closure_v7/sort/loso/selected_configs_by_fold.json",
        },
    ]


def process_tracker(item: dict[str, str]) -> list[dict[str, Any]]:
    cfg = yaml.safe_load(Path(item["config"]).read_text(encoding="utf-8")) or {}
    det = load_candidates(cfg)
    gt, ignored = load_gt_protocol(cfg["dataset_root"], sorted(det["sequence_id"].unique()))
    manifest = add_gt_presence(build_visdrone_frame_manifest(cfg["dataset_root"], sorted(det["sequence_id"].unique())), gt)
    det = merge_manifest_dimensions(det, manifest)
    det = split_duplicate_observation_tracklets(det)
    det = assign_episode_ids(det, gate_config(cfg).max_track_gap)
    mc = matching_config(cfg)
    gc = gate_config(cfg)
    selected = load_selected_selective(Path(item["selection"]))
    rows = []
    for fold_idx, seq in enumerate(sorted(det["sequence_id"].unique())):
        d = det[det["sequence_id"].eq(seq)].copy()
        g = gt[gt["sequence_id"].eq(seq)].copy()
        ig = ignored[ignored["sequence_id"].eq(seq)].copy() if ignored is not None and not ignored.empty else ignored
        fm = manifest[manifest["sequence_id"].eq(seq)].copy()
        baseline_gate = tracker_baseline_gate_result(d, gc)
        baseline_eval = evaluate_gate_result(d, baseline_gate, g, ig, fm, mc)
        qcfg = SelectiveTrustQuarantineConfig.from_mapping(selected[(fold_idx, seq)])
        selective_gate = selective_quarantine_gate_result(d, fm, qcfg)
        selective_eval = evaluate_gate_result(d, selective_gate, g, ig, fm, mc)
        rows.extend(episode_rows(item["tracker"], seq, "baseline", d, baseline_gate, baseline_eval, baseline_eval))
        rows.extend(episode_rows(item["tracker"], seq, "selective_v21", d, selective_gate, selective_eval, baseline_eval))
    return rows


def load_selected_selective(path: Path) -> dict[tuple[int, str], dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    out = {}
    for row in payload:
        if row.get("method") != "selective_trust_quarantine":
            continue
        params = json.loads(row["selected_parameter_json"])
        out[(int(row["outer_fold"]), str(row["outer_test_sequence"]))] = params
    if not out:
        raise RuntimeError(f"No selected selective quarantine configs in {path}")
    return out


def episode_rows(
    tracker: str,
    sequence: str,
    mode: str,
    candidates: pd.DataFrame,
    gate,
    result,
    baseline_result,
) -> list[dict[str, Any]]:
    baseline_false = false_rows(baseline_result.matched_detections)
    if baseline_false.empty:
        return []
    published_false = false_rows(result.matched_detections)
    baseline_keys = set(row_keys(baseline_false))
    published_keys = set(row_keys(published_false))
    candidate_key_to_index = {row_key(r): idx for idx, r in candidates.iterrows()}
    final_mask = gate.acceptance_mask.astype(bool)
    online_mask = gate.online_acceptance_mask.astype(bool) if gate.online_acceptance_mask is not None else final_mask
    final_keys = {key for key, idx in candidate_key_to_index.items() if bool(final_mask.loc[idx])}
    online_keys = {key for key, idx in candidate_key_to_index.items() if bool(online_mask.loc[idx])}
    meta = gate.episode_metadata.copy()
    rows = []
    for key, base_group in baseline_false.groupby(KEY_COLS, sort=False):
        if not isinstance(key, tuple):
            key = (key,)
        seq, track_id, episode_id = str(key[0]), str(key[1]), int(key[2])
        episode_base_keys = set(row_keys(base_group))
        episode_published_keys = episode_base_keys & published_keys
        episode_final_keys = episode_base_keys & final_keys
        episode_online_keys = episode_base_keys & online_keys
        published_group = base_group[base_group.apply(lambda r: row_key(r) in episode_published_keys, axis=1)].copy()
        frame_source = published_group if not published_group.empty else base_group.iloc[0:0].copy()
        frames = sorted(frame_source["frame_id"].astype(int).unique().tolist())
        base_frames = sorted(base_group["frame_id"].astype(int).unique().tolist())
        decision = metadata_for(meta, seq, track_id, episode_id)
        first_frame = int(frames[0]) if frames else int(base_frames[0])
        last_frame = int(frames[-1]) if frames else int(base_frames[-1])
        decision_frame = decision_frame_from_metadata(decision)
        duration = int(last_frame - first_frame + 1) if frames else 0
        rows.append(
            {
                "sequence_id": sequence,
                "tracker": tracker,
                "track_id": track_id,
                "episode_id": int(episode_id),
                "mode": mode,
                "first_frame": first_frame,
                "last_frame": last_frame,
                "duration_span_frames": duration,
                "observed_frame_count": int(len(set(frames))),
                "published_false_row_count": int(len(episode_published_keys)),
                "max_consecutive_observations": int(max_run(frames)),
                "gap_count": int(gap_stats(frames)[0]),
                "max_gap": int(gap_stats(frames)[1]),
                "decision_status": str(decision.get("selective_terminal_status") or decision.get("terminal_status") or ("baseline_published" if mode == "baseline" else "")),
                "decision_frame": decision_frame,
                "time_to_decision": int(decision_frame - int(base_frames[0])) if decision_frame is not None else math.nan,
                "temporarily_hidden_rows": int(len(episode_final_keys - episode_online_keys)) if mode != "baseline" else 0,
                "permanently_removed_rows": int(len(episode_base_keys - episode_final_keys)) if mode != "baseline" else 0,
                "right_censored": bool(decision.get("right_censored", False)),
            }
        )
    return rows


def false_rows(matched: pd.DataFrame) -> pd.DataFrame:
    if matched.empty:
        return matched.copy()
    return matched[~matched["eval_is_tp"].fillna(False).astype(bool)].copy()


def metadata_for(meta: pd.DataFrame, seq: str, track_id: str, episode_id: int) -> dict[str, Any]:
    if meta.empty:
        return {}
    hit = meta[
        meta["sequence_id"].astype(str).eq(seq)
        & meta["tracklet_id"].astype(str).eq(track_id)
        & meta["episode_id"].astype(int).eq(int(episode_id))
    ]
    if hit.empty:
        return {}
    return hit.iloc[0].to_dict()


def decision_frame_from_metadata(row: dict[str, Any]) -> int | None:
    for col in ["confirmation_frame_id", "rejection_frame_id", "finalization_frame_id", "timeout_frame_id"]:
        val = row.get(col)
        if pd.notna(val):
            return int(val)
    return None


def row_keys(df: pd.DataFrame) -> list[tuple[Any, ...]]:
    return [row_key(r) for _, r in df.iterrows()]


def row_key(row: pd.Series) -> tuple[Any, ...]:
    return tuple(str(row[c]) if c in {"sequence_id", "tracklet_id", "det_id"} else int(row[c]) for c in ROW_KEY_COLS)


def max_run(frames: list[int]) -> int:
    if not frames:
        return 0
    best = cur = 1
    for prev, curr in zip(frames, frames[1:]):
        cur = cur + 1 if curr == prev + 1 else 1
        best = max(best, cur)
    return best


def gap_stats(frames: list[int]) -> tuple[int, int]:
    if len(frames) < 2:
        return 0, 0
    gaps = [curr - prev - 1 for prev, curr in zip(frames, frames[1:]) if curr - prev > 1]
    return len(gaps), max(gaps) if gaps else 0


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (tracker, mode), group in df.groupby(["tracker", "mode"], sort=False):
        durations = group["duration_span_frames"].astype(float)
        counts = group["published_false_row_count"].astype(float)
        rows.append(
            {
                "tracker": tracker,
                "mode": mode,
                "false_episode_count": int(len(group)),
                **summary_stats("duration_span_frames", durations),
                **summary_stats("published_false_row_count", counts),
                "duration_0_fraction": float((durations == 0).mean()),
                "duration_1_fraction": float((durations == 1).mean()),
                "duration_2_fraction": float((durations == 2).mean()),
                "duration_3_fraction": float((durations == 3).mean()),
                "duration_4_10_fraction": float(((durations >= 4) & (durations <= 10)).mean()),
                "duration_gt_10_fraction": float((durations > 10).mean()),
            }
        )
    return pd.DataFrame(rows)


def summary_stats(prefix: str, values: pd.Series) -> dict[str, float]:
    return {
        f"{prefix}_mean": float(values.mean()),
        f"{prefix}_sample_std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
        f"{prefix}_median": float(values.median()),
        f"{prefix}_q1": float(values.quantile(0.25)),
        f"{prefix}_q3": float(values.quantile(0.75)),
        f"{prefix}_p90": float(values.quantile(0.90)),
        f"{prefix}_p95": float(values.quantile(0.95)),
        f"{prefix}_max": float(values.max()),
    }


def write_plots(df: pd.DataFrame, out: Path) -> None:
    plt.figure(figsize=(9, 5))
    for mode, group in df.groupby("mode", sort=False):
        plt.hist(group["duration_span_frames"], bins=30, alpha=0.55, label=label_mode(mode))
    plt.xlabel("Длительность опубликованного ложного эпизода, кадры")
    plt.ylabel("Число эпизодов")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / "false_episode_duration_histogram_ru.png", dpi=160)
    plt.close()

    plt.figure(figsize=(9, 5))
    for mode, group in df.groupby("mode", sort=False):
        values = np.sort(group["duration_span_frames"].astype(float).to_numpy())
        survival = 1.0 - np.arange(1, len(values) + 1) / max(1, len(values))
        plt.step(values, survival, where="post", label=label_mode(mode))
    plt.xlabel("Длительность опубликованного ложного эпизода, кадры")
    plt.ylabel("Доля эпизодов дольше порога")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / "false_episode_survival_curve_ru.png", dpi=160)
    plt.close()

    plt.figure(figsize=(9, 5))
    for mode, group in df.groupby("mode", sort=False):
        plt.hist(group["published_false_row_count"], bins=30, alpha=0.55, label=label_mode(mode))
    plt.xlabel("Число опубликованных ложных строк")
    plt.ylabel("Число эпизодов")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / "false_episode_rows_distribution_ru.png", dpi=160)
    plt.close()

    plt.figure(figsize=(9, 5))
    for mode, group in df.groupby("mode", sort=False):
        plt.scatter(group["duration_span_frames"], group["published_false_row_count"], s=8, alpha=0.35, label=label_mode(mode))
    plt.xlabel("Длительность опубликованного ложного эпизода, кадры")
    plt.ylabel("Опубликованные ложные строки")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / "false_episode_duration_vs_rows_ru.png", dpi=160)
    plt.close()


def label_mode(mode: str) -> str:
    return {"baseline": "Базовый трекер", "selective_v21": "Selective quarantine v2.1"}.get(mode, mode)


def git(args: list[str]) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True).strip()
    except Exception:
        return "unavailable"


if __name__ == "__main__":
    main()
