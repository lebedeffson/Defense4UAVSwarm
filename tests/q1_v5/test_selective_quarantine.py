from __future__ import annotations

import pandas as pd

from defense4uavswarm.q1_v5.selective_quarantine import (
    SelectiveTrustQuarantineConfig,
    selective_quarantine_gate_result,
)


def _row(det_id: str, tracklet_id: str, frame_id: int, confidence: float) -> dict:
    return {
        "det_id": det_id,
        "sequence_id": "s",
        "tracklet_id": tracklet_id,
        "episode_id": 0,
        "frame_id": frame_id,
        "confidence": confidence,
        "class_id": 2,
        "class_name": "car",
        "x1": 0.0,
        "y1": 0.0,
        "x2": 10.0,
        "y2": 10.0,
        "bbox_area": 100.0,
        "image_width": 100,
        "image_height": 100,
    }


def test_selective_quarantine_fast_passes_high_support_first_observation():
    det = pd.DataFrame([_row("d1", "t1", 1, 0.95)])
    cfg = SelectiveTrustQuarantineConfig(
        confidence_floor=0.05,
        fast_pass_support=0.80,
        support_history_min=30,
    )
    gate = selective_quarantine_gate_result(det, cfg)
    assert bool(gate.acceptance_mask.iloc[0])
    meta = gate.episode_metadata.iloc[0]
    assert meta["terminal_status"] == "confirmed"
    assert int(meta["gate_delay_frames"]) == 0


def test_selective_quarantine_backfills_buffer_after_late_confirmation():
    det = pd.DataFrame(
        [
            _row("warmup", "warmup", 1, 0.95),
            _row("low1", "t_low", 2, 0.52),
            _row("low2", "t_low", 3, 0.95),
        ]
    )
    cfg = SelectiveTrustQuarantineConfig(
        confidence_floor=0.05,
        relative_weight=0.0,
        support_history_min=1,
        quarantine_fraction_min=1.0,
        quarantine_fraction_max=1.0,
        fast_pass_support=0.90,
        confirmation_support=0.65,
        maximum_quarantine_frames=3,
        buffer_backfill=True,
    )
    gate = selective_quarantine_gate_result(det, cfg)
    assert gate.acceptance_mask.tolist() == [True, True, True]
    low_meta = gate.episode_metadata[gate.episode_metadata["tracklet_id"].eq("t_low")].iloc[0]
    assert low_meta["terminal_status"] == "confirmed"
    assert int(low_meta["first_candidate_frame_id"]) == 2
    assert int(low_meta["confirmation_frame_id"]) == 3
    assert int(low_meta["gate_delay_frames"]) == 1
