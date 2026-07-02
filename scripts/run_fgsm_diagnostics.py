#!/usr/bin/env python
from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from defense4uavswarm.attacks.fgsm import fgsm_image_with_diagnostics
from defense4uavswarm.config import ensure_dirs, load_config
from defense4uavswarm.datasets.visdrone import VisDroneDataset
from defense4uavswarm.detectors.yolo import YoloRunner
from defense4uavswarm.matrix import selected_models, task_dataset


def detect_stats(runner: YoloRunner, image_path: Path, cfg: dict) -> tuple[int, float]:
    res = runner.model.predict(
        source=str(image_path),
        imgsz=cfg["model"]["imgsz"],
        conf=cfg["model"]["conf"],
        iou=cfg["model"]["iou"],
        max_det=cfg["model"]["max_det"],
        device=runner.device,
        verbose=False,
    )[0]
    boxes = res.boxes
    if boxes is None or len(boxes) == 0:
        return 0, 0.0
    return len(boxes), float(boxes.conf.mean().item())


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--models", nargs="+", default=["yolov8n", "yolov8s"])
    p.add_argument("--task", default="vid", choices=["vid", "mot"])
    p.add_argument("--eps", nargs="+", type=float, default=[0.004, 0.008, 0.016, 0.032])
    p.add_argument("--limit-sequences", type=int, default=3)
    p.add_argument("--max-frames-per-sequence", type=int, default=10)
    p.add_argument("--loss", default=None, choices=["class_only", "proxy_conf_box_if_available"])
    p.add_argument("--out", default=None)
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    cfg = load_config(args.config)
    ensure_dirs(cfg)
    _, ds = task_dataset(cfg, args.task)
    sequences = ds.sequence_ids()[: args.limit_sequences]
    rows = []
    for model in selected_models(cfg, args.models):
        mc = deepcopy(cfg)
        mc["model"].update(model)
        if args.loss:
            mc["fgsm"]["loss"] = args.loss
        runner = YoloRunner(mc["model"]["weights"], mc["device"])
        model_name = model["name"]
        for seq in sequences:
            ann = ds.annotations(seq)
            frames = list(ds.frames([seq]))[: args.max_frames_per_sequence]
            for rec in tqdm(frames, desc=f"diag {model_name} {seq}"):
                gt_boxes = ann[ann.frame_id == rec.frame_id][["x1", "y1", "x2", "y2"]].to_numpy()
                clean_n, clean_conf = detect_stats(runner, rec.image_path, mc)
                for eps in args.eps:
                    rel = Path(rec.sequence_id) / f"{rec.frame_id:07d}.jpg"
                    out_path = (
                        Path(mc["fgsm"]["cache_dir"])
                        / "diagnostics"
                        / mc["fgsm"]["loss"]
                        / model_name
                        / f"eps_{eps}"
                        / rel
                    )
                    adv_path, diag = fgsm_image_with_diagnostics(
                        rec.image_path,
                        eps,
                        out_path,
                        runner.model.model,
                        mc["model"]["imgsz"],
                        mc["device"],
                        loss_mode=mc["fgsm"]["loss"],
                        gt_boxes=gt_boxes,
                        beta=float(mc["fgsm"].get("proxy_beta", 1.0)),
                        force=args.force,
                    )
                    if not diag:
                        adv_path, diag = fgsm_image_with_diagnostics(
                            rec.image_path,
                            eps,
                            out_path,
                            runner.model.model,
                            mc["model"]["imgsz"],
                            mc["device"],
                            loss_mode=mc["fgsm"]["loss"],
                            gt_boxes=gt_boxes,
                            beta=float(mc["fgsm"].get("proxy_beta", 1.0)),
                            force=True,
                        )
                    adv_n, adv_conf = detect_stats(runner, adv_path, mc)
                    rows.append(
                        {
                            "model_name": model_name,
                            "eps": eps,
                            "sequence_id": rec.sequence_id,
                            "frame_id": rec.frame_id,
                            **diag,
                            "num_detections_clean": clean_n,
                            "num_detections_adv": adv_n,
                            "mean_conf_clean": clean_conf,
                            "mean_conf_adv": adv_conf,
                        }
                    )
    out = Path(args.out) if args.out else Path(cfg["outputs"]["results_dir"]) / "fgsm_diagnostics.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(out)


if __name__ == "__main__":
    main()
