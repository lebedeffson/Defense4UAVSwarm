from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from defense4uavswarm.datasets.visdrone import VisDroneDataset
from defense4uavswarm.filtering.tnorms import apply_conf_threshold, apply_tnorm
from defense4uavswarm.metadata import write_metadata
from defense4uavswarm.metrics.asr import attack_success_rate
from defense4uavswarm.metrics.detection import precision_recall_f1
from defense4uavswarm.metrics.tracking import tracking_metrics
from defense4uavswarm.pipeline import build_threshold_selection, evaluate, save
from defense4uavswarm.schema import to_frame


def _write_smoke_dataset(root: Path) -> None:
    seq_dir = root / "VisDrone2019-VID-val" / "sequences" / "smoke_seq_001"
    ann_dir = root / "VisDrone2019-VID-val" / "annotations"
    seq_dir.mkdir(parents=True, exist_ok=True)
    ann_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for frame_id in range(1, 4):
        img = np.zeros((120, 160, 3), dtype=np.uint8)
        x, y, w, h = 20 + frame_id * 5, 30, 30, 20
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255), -1)
        cv2.imwrite(str(seq_dir / f"{frame_id:07d}.jpg"), img)
        rows.append(f"{frame_id},1,{x},{y},{w},{h},1,4,0,0\n")
    (ann_dir / "smoke_seq_001.txt").write_text("".join(rows), encoding="utf-8")


def _synthetic_preds(scenario: str, eps: float) -> pd.DataFrame:
    rows = []
    for frame_id in range(1, 4):
        x, y, w, h = 20 + frame_id * 5, 30, 30, 20
        shift = 0 if scenario == "S0" else frame_id * 4
        conf = 0.9 if scenario == "S0" else 0.55 - frame_id * 0.05
        rows.append(
            {
                "scenario": scenario,
                "model_name": "smoke_model",
                "eps": eps,
                "frame_id": frame_id,
                "sequence_id": "smoke_seq_001",
                "gt_track_id": None,
                "pred_track_id": 1 if frame_id < 3 or scenario == "S0" else 2,
                "class_id": 4,
                "class_name": "car",
                "x1": x + shift,
                "y1": y,
                "x2": x + w + shift,
                "y2": y + h,
                "confidence": conf,
                "c_i": conf,
                "k_i": 1.0 if scenario == "S0" else max(0.2, 1.0 - frame_id * 0.25),
                "s_i": 1.0,
                "x_i": 1.0,
                "Q_i": None,
                "accepted": True,
                "t_norm": None,
                "tau": None,
                "image_path": f"smoke/{frame_id:07d}.jpg",
                "latency_ms": 1.0,
                "xai_triggered": False,
            }
        )
    return to_frame(rows)


def run_smoke_test(cfg: dict) -> None:
    smoke_root = Path("outputs/smoke_visdrone")
    _write_smoke_dataset(smoke_root)
    ds = VisDroneDataset(smoke_root, "val")
    gt = ds.all_annotations()
    results = Path(cfg["outputs"]["results_dir"])

    s0 = _synthetic_preds("S0", 0.0)
    s1 = _synthetic_preds("S1", 0.004)
    tau_conf, tau_q, thresholds = build_threshold_selection(gt, s0, cfg)
    naive = apply_conf_threshold(s1, tau_conf)
    s2_all = [apply_tnorm(s1, name, tau) for name, tau in tau_q.items()]
    s2 = pd.concat(s2_all, ignore_index=True)

    metrics = [evaluate(gt, s0, "S0", 0.0)]
    for frame, name in ((s1, "S1"), (naive, "S_naive")):
        m = evaluate(gt, frame, name, 0.004)
        m["ASR"] = attack_success_rate(gt, s0, frame)
        metrics.append(m)
    for name, tau in tau_q.items():
        frame = s2[s2.t_norm == name]
        m = evaluate(gt, frame, "S2", 0.004, name, tau)
        m["ASR"] = attack_success_rate(gt, s0, frame)
        metrics.append(m)

    save(s0, results / "s0_baseline.csv")
    save(s1, results / "s1_fgsm.csv")
    save(naive, results / "s_naive.csv")
    save(s2, results / "s2_tnorm.csv")
    save(thresholds, results / "threshold_selection.csv")
    save(pd.DataFrame(metrics), results / "summary_metrics.csv")
    write_metadata(
        results / "metadata.json",
        cfg,
        {
            "dataset": "smoke_visdrone",
            "smoke_sequence": "smoke_seq_001",
            "tau_conf_star": tau_conf,
            "tau_Q_star": tau_q,
            "smoke_precision_recall": precision_recall_f1(gt, s0),
            "smoke_tracking": tracking_metrics(gt, s0),
        },
    )
    print(f"Smoke test OK: {results}")
