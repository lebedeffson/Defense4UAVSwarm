from __future__ import annotations

from pathlib import Path

import pandas as pd
from tqdm import tqdm

from defense4uavswarm.config import ensure_dirs, require_packages
from defense4uavswarm.class_groups import filter_by_group, load_class_groups
from defense4uavswarm.datasets.visdrone import VisDroneDataset
from defense4uavswarm.filtering.tnorms import apply_conf_threshold, apply_tnorm
from defense4uavswarm.metadata import write_metadata
from defense4uavswarm.metrics.asr import attack_success_rate
from defense4uavswarm.metrics.detection import choose_threshold, precision_recall_f1
from defense4uavswarm.metrics.tracking import tracking_metrics
from defense4uavswarm.tracking.simple import add_simple_tracks


def save(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def run_detection(
    cfg: dict,
    ds: VisDroneDataset,
    scenario: str,
    eps: float | None = None,
    limit_sequences: int | None = None,
    sequence_ids: list[str] | None = None,
) -> pd.DataFrame:
    from defense4uavswarm.attacks.fgsm import fgsm_image_with_diagnostics
    from defense4uavswarm.detectors.yolo import YoloRunner
    from defense4uavswarm.schema import to_frame

    runner = YoloRunner(cfg["model"]["weights"], cfg["device"])
    rows = []
    sequences = sequence_ids or (ds.sequence_ids()[:limit_sequences] if limit_sequences else ds.sequence_ids())
    for seq in tqdm(sequences, desc=f"{scenario} sequences"):
        runner.reset_tracker()
        state = {}
        ann = ds.annotations(seq) if scenario.startswith("S1") else None
        for rec in ds.frames([seq]):
            image_path = rec.image_path
            if scenario.startswith("S1") and eps is not None:
                rel = Path(rec.sequence_id) / f"{rec.frame_id:07d}.jpg"
                model_name = cfg["model"].get("name", Path(cfg["model"]["weights"]).stem)
                loss_mode = cfg.get("fgsm", {}).get("loss", "class_only")
                gt_boxes = None
                if ann is not None and loss_mode == "proxy_conf_box_if_available":
                    gt_boxes = ann[ann.frame_id == rec.frame_id][["x1", "y1", "x2", "y2"]].to_numpy()
                image_path, _ = fgsm_image_with_diagnostics(
                    rec.image_path,
                    eps,
                    Path(cfg["fgsm"]["cache_dir"]) / loss_mode / model_name / f"eps_{eps}" / rel,
                    runner.model.model,
                    cfg["model"]["imgsz"],
                    cfg["device"],
                    loss_mode=loss_mode,
                    gt_boxes=gt_boxes,
                    beta=float(cfg.get("fgsm", {}).get("proxy_beta", 1.0)),
                )
            dets, _ = runner.track_frame(image_path, cfg, scenario, eps or 0.0, rec.sequence_id, rec.frame_id, state)
            rows.extend(dets)
    df = to_frame(rows)
    df["fgsm_loss"] = cfg.get("fgsm", {}).get("loss", "class_only")
    if len(df) and df["pred_track_id"].isna().all():
        df = add_simple_tracks(df)
    return df


def evaluate(
    gt: pd.DataFrame,
    pred: pd.DataFrame,
    scenario: str,
    eps: float | None,
    t_norm: str | None = None,
    tau: float | None = None,
    cfg: dict | None = None,
    class_group: str = "all",
    include_map: bool | None = None,
    include_tracking: bool = True,
) -> dict:
    gt_eval, pred_eval = gt, pred
    if cfg is not None:
        gt_eval = filter_by_group(gt, cfg, class_group)
        pred_eval = filter_by_group(pred, cfg, class_group)
    if include_map is None:
        include_map = class_group == "all"
    groups_cfg = load_class_groups(cfg["class_groups_file"]) if cfg is not None else {}
    d = precision_recall_f1(
        gt_eval,
        pred_eval,
        include_map=include_map,
        class_agnostic=bool(cfg.get("evaluation", {}).get("class_agnostic", True)) if cfg is not None else True,
        aliases=groups_cfg.get("aliases", {}),
    )
    tr = tracking_metrics(gt_eval, pred_eval) if include_tracking else {"MOTA": None, "IDF1": None, "IDSW": None, "tracking_error": None, "track_breaks": None}
    lat = pred["latency_ms"].dropna()
    return {
        "model_name": pred["model_name"].dropna().iloc[0] if "model_name" in pred and len(pred["model_name"].dropna()) else None,
        "scenario": scenario,
        "eps": eps,
        "class_group": class_group,
        "t_norm": t_norm,
        "tau": tau,
        **d,
        **tr,
        "latency_ms": float(lat.mean()) if len(lat) else None,
        "latency_p95_ms": float(lat.quantile(0.95)) if len(lat) else None,
    }


def build_threshold_selection(gt: pd.DataFrame, s0: pd.DataFrame, cfg: dict) -> tuple[float, dict[str, float], pd.DataFrame]:
    rows = []
    groups_cfg = load_class_groups(cfg["class_groups_file"])
    class_agnostic = bool(cfg.get("evaluation", {}).get("class_agnostic", True))
    tau_conf, conf_grid = choose_threshold(gt, s0, cfg["filtering"]["confidence_grid"], "confidence", cfg["filtering"]["tie_eps"], class_agnostic=class_agnostic, aliases=groups_cfg.get("aliases", {}))
    for r in conf_grid.to_dict("records"):
        rows.append({"mode": "S_naive", "t_norm": "confidence", "tau": r["tau"], "F1": r["F1"], "selected": r["tau"] == tau_conf})
    tau_q = {}
    for name in cfg["filtering"]["t_norms"]:
        scored = apply_tnorm(s0, name, 0.0)
        tau, grid = choose_threshold(
            gt,
            scored,
            cfg["filtering"]["tau_grid"],
            "Q_i",
            cfg["filtering"]["tie_eps"],
            class_agnostic=class_agnostic,
            aliases=groups_cfg.get("aliases", {}),
            recall_floor=conf_grid[conf_grid["tau"] == tau_conf]["recall"].iloc[0] * (1.0 - float(cfg["filtering"].get("recall_drop_limit", 1.0))),
        )
        tau_q[name] = tau
        for r in grid.to_dict("records"):
            rows.append({"mode": "S2", "t_norm": name, "tau": r["tau"], "F1": r["F1"], "selected": r["tau"] == tau})
    return tau_conf, tau_q, pd.DataFrame(rows)


def load_existing(path: Path) -> pd.DataFrame | None:
    return pd.read_csv(path) if path.exists() else None


def run_s0_to_s2(cfg: dict, scenario: str = "s0_s2", eps: float | None = None, limit_sequences: int | None = None) -> None:
    require_packages(["cv2", "pandas", "torch", "ultralytics", "tqdm"])
    ensure_dirs(cfg)
    results = Path(cfg["outputs"]["results_dir"])
    ds = VisDroneDataset(cfg["dataset"]["root"], "val")
    sequences = ds.sequence_ids()[:limit_sequences] if limit_sequences else None
    gt = ds.all_annotations(sequences)

    metrics = []
    s0 = load_existing(results / "s0_baseline.csv")
    if scenario in ("s0", "s0_s2") or s0 is None:
        s0 = run_detection(cfg, ds, "S0", 0.0, limit_sequences)
        save(s0, results / "s0_baseline.csv")
    if scenario in ("s0", "s0_s2"):
        metrics.append(evaluate(gt, s0, "S0", 0.0, cfg=cfg))

    tau_conf, tau_q, threshold_df = build_threshold_selection(gt, s0, cfg)
    save(threshold_df, results / "threshold_selection.csv")

    eps_values = [eps] if eps is not None else cfg["fgsm"]["eps_values"]
    all_s1, all_naive, all_s2 = [], [], []
    if scenario in ("s1", "s2", "s0_s2"):
        for current_eps in eps_values:
            s1_path = results / f"s1_fgsm_eps_{current_eps}.csv"
            s1 = load_existing(s1_path)
            if scenario in ("s1", "s0_s2") or s1 is None:
                s1 = run_detection(cfg, ds, "S1", current_eps, limit_sequences)
                save(s1, s1_path)
            all_s1.append(s1)
            if scenario in ("s1", "s0_s2"):
                m = evaluate(gt, s1, "S1", current_eps, cfg=cfg)
                m["ASR"] = attack_success_rate(gt, s0, s1)
                metrics.append(m)

            naive = apply_conf_threshold(s1, tau_conf)
            save(naive, results / f"s_naive_eps_{current_eps}.csv")
            all_naive.append(naive)
            if scenario in ("s2", "s0_s2"):
                m = evaluate(gt, naive, "S_naive", current_eps, "confidence", tau_conf, cfg=cfg)
                m["ASR"] = attack_success_rate(gt, s0, naive)
                metrics.append(m)

            for name, tau in tau_q.items():
                s2 = apply_tnorm(s1, name, tau, mode=cfg["filtering"].get("tnorm_filter_mode", "hard"))
                save(s2, results / f"s2_tnorm_{name}_eps_{current_eps}.csv")
                all_s2.append(s2)
                if scenario in ("s2", "s0_s2"):
                    m = evaluate(gt, s2, "S2", current_eps, name, tau, cfg=cfg)
                    m["ASR"] = attack_success_rate(gt, s0, s2)
                    metrics.append(m)

    if all_s1:
        save(pd.concat(all_s1, ignore_index=True), results / "s1_fgsm.csv")
    if all_naive:
        save(pd.concat(all_naive, ignore_index=True), results / "s_naive.csv")
    if all_s2:
        save(pd.concat(all_s2, ignore_index=True), results / "s2_tnorm.csv")
    if metrics:
        save(pd.DataFrame(metrics), results / "summary_metrics.csv")
    write_metadata(
        results / "metadata.json",
        cfg,
        {"tau_conf_star": tau_conf, "tau_Q_star": tau_q, "xai_frame_fraction": None, "limit_sequences": limit_sequences},
    )
