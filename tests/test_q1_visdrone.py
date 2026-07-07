from __future__ import annotations

import pandas as pd

from defense4uavswarm.q1_visdrone import add_single_camera_features, evaluate_methods, label_detections, make_chunk_split, select_budget_chunks


def test_q1_matching_and_false_new_tracks() -> None:
    gt = pd.DataFrame(
        [
            {"sequence_id": "s1", "frame_id": 1, "object_id": "g1", "class_id": 4, "class_name": "car", "x1": 0, "y1": 0, "x2": 20, "y2": 20},
            {"sequence_id": "s1", "frame_id": 2, "object_id": "g1", "class_id": 4, "class_name": "car", "x1": 1, "y1": 0, "x2": 21, "y2": 20},
        ]
    )
    det = pd.DataFrame(
        [
            {"det_id": "d1", "sequence_id": "s1", "frame_id": 1, "class_id": 2, "class_name": "car", "bbox": [0, 0, 20, 20], "confidence": 0.9, "x1": 0, "y1": 0, "x2": 20, "y2": 20},
            {"det_id": "d2", "sequence_id": "s1", "frame_id": 2, "class_id": 2, "class_name": "car", "bbox": [1, 0, 21, 20], "confidence": 0.8, "x1": 1, "y1": 0, "x2": 21, "y2": 20},
            {"det_id": "f1", "sequence_id": "s1", "frame_id": 1, "class_id": 2, "class_name": "car", "bbox": [50, 50, 70, 70], "confidence": 0.95, "x1": 50, "y1": 50, "x2": 70, "y2": 70},
        ]
    )
    labeled = add_single_camera_features(label_detections(det, gt))
    summary = evaluate_methods(labeled, gt, ["s_naive", "rf_learned_gate"])
    naive = summary[summary["method"].eq("s_naive")].iloc[0]
    rf = summary[summary["method"].eq("rf_learned_gate")].iloc[0]
    assert naive["TP"] == 2
    assert naive["FP"] == 1
    assert naive["false_new_tracks"] == 1
    assert bool(rf["available"]) is False


def test_q1_chunk_budget_split_is_chunk_based() -> None:
    gt = pd.DataFrame(
        [{"sequence_id": "s1", "frame_id": i, "object_id": "g", "class_id": 4, "class_name": "car", "x1": 0, "y1": 0, "x2": 1, "y2": 1} for i in range(1, 301)]
    )
    chunks = make_chunk_split(gt, chunk_size=50)
    selected = select_budget_chunks(chunks, 0.5, seed=11)
    assert selected
    assert all(chunk.startswith("s1_chunk_") for chunk in selected)
    assert not set(chunks[chunks["split"].eq("holdout")]["chunk_id"]).intersection(selected)
