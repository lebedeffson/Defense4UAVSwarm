#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from defense4uavswarm.v9_geomdyn import load_config, prepare_v9_features, v9_acceptance
from defense4uavswarm.v8_sim import metric_summary, read_json, train_rf_model


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--gt-2d", required=True)
    p.add_argument("--gt-3d", default="")
    p.add_argument("--detections", required=True)
    p.add_argument("--scenarios", nargs="+", required=True)
    p.add_argument("--selected-params", default="outputs/results/v9_geomdyn/main/v9_selected_params.yaml")
    p.add_argument("--rf-model", default="")
    p.add_argument("--sync-delay-frames", nargs="+", type=int, default=[0, 1, 2, 3])
    p.add_argument("--pose-noise-translation-m", nargs="+", type=float, default=[0, 0.5, 1.0, 2.0])
    p.add_argument("--pose-noise-yaw-deg", nargs="+", type=float, default=[0, 1, 3, 5])
    p.add_argument("--agent-dropout-prob", nargs="+", type=float, default=[0.0, 0.1, 0.3, 0.5])
    p.add_argument("--combined-stress", action="store_true")
    p.add_argument("--output-dir", required=True)
    args = p.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    cfg = load_config()
    rf_model = train_fixed_rf(args, cfg) if "s2_learned_fp_gate" in [s.lower() for s in args.scenarios] else None
    sync_rows = []
    for delay in args.sync_delay_frames:
        sync_rows.extend(eval_condition(args, cfg, rf_model, {"sync_delay_frames": delay}, f"sync_delay_{delay}"))
    pd.DataFrame(sync_rows).to_csv(out / "sync_delay_sensitivity.csv", index=False)
    pose_rows = []
    # Compact diagonal keeps v9 first-pass runtime bounded while covering low/high perturbations.
    pairs = list(zip(args.pose_noise_translation_m, args.pose_noise_yaw_deg))
    for trans, yaw in pairs:
        pose_rows.extend(eval_condition(args, cfg, rf_model, {"pose_noise_translation_m": trans, "pose_noise_yaw_deg": yaw}, f"pose_{trans}_{yaw}"))
    pd.DataFrame(pose_rows).to_csv(out / "pose_noise_sensitivity.csv", index=False)
    drop_rows = []
    for prob in args.agent_dropout_prob:
        drop_rows.extend(eval_condition(args, cfg, rf_model, {"agent_dropout_prob": prob}, f"dropout_{prob}"))
    pd.DataFrame(drop_rows).to_csv(out / "agent_dropout_sensitivity.csv", index=False)
    combined = (
        eval_condition(
            args,
            cfg,
            rf_model,
            {"pose_noise_translation_m": 1.0, "pose_noise_yaw_deg": 3.0, "sync_delay_frames": 2, "agent_dropout_prob": 0.3},
            "combined_stress",
        )
        if args.combined_stress
        else []
    )
    pd.DataFrame(combined).to_csv(out / "combined_stress_summary.csv", index=False)
    all_rows = pd.DataFrame(sync_rows + pose_rows + drop_rows + combined)
    write_robustness_summary(all_rows, out)
    (out / "rf_fixed_model_note.md").write_text(
        "RF was trained on clean calibration and applied unchanged to perturbed robustness conditions.\n"
        "No retraining was performed for pose noise, sync delay, dropout, or combined stress.\n",
        encoding="utf-8",
    )
    print(f"status=ok output={out}")


def train_fixed_rf(args: argparse.Namespace, cfg: dict) -> object:
    manifest = read_json(args.manifest)
    gt = pd.DataFrame(read_json(args.gt_2d)["boxes"])
    det = pd.DataFrame(read_json(args.detections)["detections"])
    det = prepare_v9_features(det, manifest, cfg, modifiers={})
    return train_rf_model(det, gt)


def eval_condition(args: argparse.Namespace, cfg: dict, rf_model: object | None, modifiers: dict[str, float], condition: str) -> list[dict]:
    manifest = read_json(args.manifest)
    gt = pd.DataFrame(read_json(args.gt_2d)["boxes"])
    det = pd.DataFrame(read_json(args.detections)["detections"])
    gt = gt[gt["scene_id"].isin(["scene_004", "scene_005"])].copy()
    det = det[det["scene_id"].isin(["scene_004", "scene_005"])].copy()
    det = prepare_v9_features(det, manifest, cfg, modifiers=modifiers)
    rows = []
    for scenario in args.scenarios:
        accepted, _ = v9_acceptance(det, scenario, cfg, rf_model)
        row = metric_summary(gt, det, accepted, scenario, len(gt), gt[["scene_id", "frame_id"]].drop_duplicates().shape[0], manifest, 0.001)
        row["condition"] = condition
        rows.append(row)
    return rows


def write_robustness_summary(rows: pd.DataFrame, out: Path) -> None:
    if rows.empty:
        rows.to_csv(out / "robustness_comparison_summary.csv", index=False)
        return
    clean = rows[rows["condition"].isin(["sync_delay_0", "pose_0.0_0", "dropout_0.0"])].groupby("scenario").first()
    summary = []
    for condition, group in rows.groupby("condition"):
        winner_f1 = group.sort_values("F1", ascending=False).iloc[0]["scenario"]
        winner_false = group.sort_values("false_new_tracks", ascending=True).iloc[0]["scenario"]
        for _, row in group.iterrows():
            base = clean.loc[row["scenario"]] if row["scenario"] in clean.index else row
            summary.append(
                {
                    "condition": condition,
                    "scenario": row["scenario"],
                    "FP": row["FP"],
                    "FN": row["FN"],
                    "F1": row["F1"],
                    "false_new_tracks": row["false_new_tracks"],
                    "delta_F1_vs_clean": row["F1"] - base["F1"],
                    "delta_false_new_vs_clean": row["false_new_tracks"] - base["false_new_tracks"],
                    "winner_by_F1": winner_f1,
                    "winner_by_false_new_tracks": winner_false,
                }
            )
    pd.DataFrame(summary).to_csv(out / "robustness_comparison_summary.csv", index=False)


if __name__ == "__main__":
    main()
