from pathlib import Path

import pandas as pd

from defense4uavswarm.datasets.visdrone import VisDroneDetDataset
from defense4uavswarm.metrics.asr import attack_success_breakdown_det


def test_visdrone_det_dataset_reads_original_annotations(tmp_path: Path):
    root = tmp_path / "visdrone"
    split = root / "VisDrone2019-DET-val"
    (split / "images").mkdir(parents=True)
    (split / "annotations").mkdir()
    (split / "images" / "sample.jpg").write_bytes(b"not-used-by-original-annotation-reader")
    (split / "annotations" / "sample.txt").write_text(
        "10,20,30,40,1,4,0,0\n0,0,10,10,0,0,0,0\n",
        encoding="utf-8",
    )

    ds = VisDroneDetDataset(root, "val")

    assert ds.sequence_ids() == ["sample"]
    frames = list(ds.frames())
    assert frames[0].sequence_id == "sample"
    assert frames[0].frame_id == 1
    ann = ds.annotations("sample")
    assert len(ann) == 1
    assert ann.iloc[0]["class_name"] == "car"
    assert ann.iloc[0]["gt_track_id"] == -1


def test_attack_success_breakdown_det_uses_image_events():
    gt = pd.DataFrame(
        [
            {"sequence_id": "img1", "frame_id": 1, "x1": 0, "y1": 0, "x2": 10, "y2": 10},
            {"sequence_id": "img2", "frame_id": 1, "x1": 0, "y1": 0, "x2": 10, "y2": 10},
        ]
    )
    clean = pd.DataFrame(
        [
            {"sequence_id": "img1", "frame_id": 1, "x1": 0, "y1": 0, "x2": 10, "y2": 10, "accepted": True, "confidence": 0.9},
            {"sequence_id": "img2", "frame_id": 1, "x1": 0, "y1": 0, "x2": 10, "y2": 10, "accepted": True, "confidence": 0.9},
        ]
    )
    attacked = pd.DataFrame(
        [
            {"sequence_id": "img1", "frame_id": 1, "x1": 100, "y1": 100, "x2": 110, "y2": 110, "accepted": True, "confidence": 0.9},
            {"sequence_id": "img2", "frame_id": 1, "x1": 0, "y1": 0, "x2": 10, "y2": 10, "accepted": True, "confidence": 0.9},
            {"sequence_id": "img2", "frame_id": 1, "x1": 20, "y1": 20, "x2": 30, "y2": 30, "accepted": True, "confidence": 0.8},
        ]
    )

    out = attack_success_breakdown_det(gt, clean, attacked)

    assert out["ASR_det"] == 1.0
    assert out["asr_det_miss_images"] == 1
    assert out["asr_det_fp_images"] == 2
    assert out["asr_det_total_images"] == 2
