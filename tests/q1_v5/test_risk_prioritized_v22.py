from __future__ import annotations

import pandas as pd

from defense4uavswarm.q1_v5.token_bucket import FrameTokenBucket, assert_bucket_invariant
from defense4uavswarm.q1_v5.two_stage_quarantine import RiskPrioritizedTwoStageConfig, risk_prioritized_two_stage_gate_result


def _row(det_id: str, tracklet_id: str, frame_id: int, conf: float = 0.2) -> dict:
    return {
        "det_id": det_id,
        "sequence_id": "s",
        "tracklet_id": tracklet_id,
        "episode_id": 0,
        "frame_id": frame_id,
        "confidence": conf,
        "class_id": 2,
        "class_name": "car",
        "x1": 0.0,
        "y1": 0.0,
        "x2": 10.0,
        "y2": 10.0,
        "image_width": 100,
        "image_height": 100,
    }


def _manifest(n: int = 6) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["s"] * n,
            "frame_id": list(range(1, n + 1)),
            "image_path": [f"s/{i}.jpg" for i in range(1, n + 1)],
            "image_width": [100] * n,
            "image_height": [100] * n,
            "has_gt": [True] * n,
        }
    )


def test_frame_token_bucket_conservation_and_threshold_skip():
    bucket = FrameTokenBucket(fraction=0.5, capacity=2.0, stage="stage1")
    bucket.begin_frame(eligibility_event_count=2)
    assert bucket.try_admit()
    rec = bucket.record(sequence_id="s", frame_id=1, eligibility_event_count=2, eligible_above_threshold=1, admitted_count=1)
    assert rec.conservation_residual == 0
    assert_bucket_invariant(bucket)


def test_two_stage_temporal_absence_rejects_after_stage1_probe():
    det = pd.DataFrame([_row("a", "t", 1)])
    scores = pd.DataFrame({"sequence_id": ["s"], "tracklet_id": ["t"], "episode_id": [0], "stage1_risk_score": [10.0], "stage2_risk_score": [0.0]})
    gate, ledger, admission = risk_prioritized_two_stage_gate_result(
        det,
        _manifest(3),
        scores,
        tau_stage1=1.0,
        tau_stage2=1.0,
        cfg=RiskPrioritizedTwoStageConfig(stage1_fraction_max=1.0, stage2_fraction_max=0.0),
    )
    assert not bool(gate.acceptance_mask.iloc[0])
    assert gate.episode_metadata.iloc[0]["selective_terminal_status"] == "REJECTED_BY_TEMPORAL_ABSENCE"
    assert ledger["conservation_residual"].abs().max() == 0
    assert admission["admitted"].any()


def test_two_stage_releases_after_stage1_when_stage2_risk_low():
    det = pd.DataFrame([_row("a", "t", 1), _row("b", "t", 2)])
    scores = pd.DataFrame({"sequence_id": ["s"], "tracklet_id": ["t"], "episode_id": [0], "stage1_risk_score": [10.0], "stage2_risk_score": [0.0]})
    gate, _, _ = risk_prioritized_two_stage_gate_result(
        det,
        _manifest(4),
        scores,
        tau_stage1=1.0,
        tau_stage2=1.0,
        cfg=RiskPrioritizedTwoStageConfig(stage1_fraction_max=1.0, stage2_fraction_max=1.0),
    )
    assert gate.acceptance_mask.tolist() == [True, True]
    assert gate.online_acceptance_mask.tolist() == [False, True]
    assert gate.episode_metadata.iloc[0]["selective_terminal_status"] == "RELEASED_AFTER_STAGE1"


def test_two_stage_horizon_release_backfills_buffer():
    det = pd.DataFrame([_row("a", "t", 1), _row("b", "t", 2), _row("c", "t", 5)])
    scores = pd.DataFrame({"sequence_id": ["s"], "tracklet_id": ["t"], "episode_id": [0], "stage1_risk_score": [10.0], "stage2_risk_score": [10.0]})
    gate, _, _ = risk_prioritized_two_stage_gate_result(
        det,
        _manifest(6),
        scores,
        tau_stage1=1.0,
        tau_stage2=1.0,
        cfg=RiskPrioritizedTwoStageConfig(stage1_fraction_max=1.0, stage2_fraction_max=1.0, stage2_horizon_frames=1),
    )
    assert gate.acceptance_mask.tolist() == [True, True, True]
    assert gate.online_acceptance_mask.tolist() == [False, False, True]
    assert gate.episode_metadata.iloc[0]["selective_terminal_status"] == "RELEASED_AFTER_STAGE2_HORIZON"
