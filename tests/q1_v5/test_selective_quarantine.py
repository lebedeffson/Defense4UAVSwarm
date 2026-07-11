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


def _manifest(last_frame: int = 8) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "sequence_id": "s",
                "frame_id": i,
                "image_path": f"s/{i:07d}.jpg",
                "image_width": 100,
                "image_height": 100,
                "has_gt": True,
            }
            for i in range(1, last_frame + 1)
        ]
    )


def test_selective_quarantine_fast_passes_high_support_first_observation():
    det = pd.DataFrame([_row("d1", "t1", 1, 0.95)])
    cfg = SelectiveTrustQuarantineConfig(
        confidence_floor=0.05,
        fast_pass_support=0.80,
        support_history_min=30,
    )
    gate = selective_quarantine_gate_result(det, _manifest(3), cfg)
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
    gate = selective_quarantine_gate_result(det, _manifest(5), cfg)
    assert gate.acceptance_mask.tolist() == [True, True, True]
    assert gate.online_acceptance_mask.tolist() == [True, False, True]
    low_meta = gate.episode_metadata[gate.episode_metadata["tracklet_id"].eq("t_low")].iloc[0]
    assert low_meta["terminal_status"] == "confirmed"
    assert low_meta["selective_terminal_status"] == "CONFIRMED_BY_EVIDENCE"
    assert int(low_meta["first_candidate_frame_id"]) == 2
    assert int(low_meta["confirmation_frame_id"]) == 3
    assert int(low_meta["gate_delay_frames"]) == 1


def test_selective_quarantine_timeout_releases_to_baseline_without_veto():
    det = pd.DataFrame(
        [
            _row("warmup", "warmup", 1, 0.95),
            _row("low1", "t_low", 2, 0.10),
            _row("low2", "t_low", 4, 0.10),
        ]
    )
    cfg = SelectiveTrustQuarantineConfig(
        confidence_floor=0.05,
        critical_confidence_floor=0.02,
        relative_weight=0.0,
        support_history_min=1,
        quarantine_fraction_min=1.0,
        quarantine_fraction_max=1.0,
        fast_pass_support=0.99,
        confirmation_support=0.99,
        maximum_quarantine_frames=1,
        timeout_policy="release_to_baseline",
        reject_requires_hard_veto=True,
        buffer_backfill=True,
    )
    gate = selective_quarantine_gate_result(det, _manifest(6), cfg)
    assert gate.acceptance_mask.tolist() == [True, False, False]
    low_meta = gate.episode_metadata[gate.episode_metadata["tracklet_id"].eq("t_low")].iloc[0]
    assert low_meta["terminal_status"] == "rejected"
    assert low_meta["selective_terminal_status"] == "REJECTED_BY_TEMPORAL_ABSENCE"
    assert low_meta["terminal_reason"] == "temporal_absence_expired"


def test_selective_quarantine_no_reject_without_hard_veto():
    det = pd.DataFrame(
        [
            _row("warmup", "warmup", 1, 0.95),
            _row("low1", "t_low", 2, 0.10),
            _row("low2", "t_low", 4, 0.10),
        ]
    )
    cfg = SelectiveTrustQuarantineConfig(
        confidence_floor=0.05,
        critical_confidence_floor=0.02,
        relative_weight=0.0,
        support_history_min=1,
        quarantine_fraction_min=1.0,
        quarantine_fraction_max=1.0,
        fast_pass_support=0.99,
        confirmation_support=0.99,
        maximum_quarantine_frames=1,
        timeout_policy="reject",
        reject_requires_hard_veto=True,
        buffer_backfill=True,
    )
    gate = selective_quarantine_gate_result(det, _manifest(6), cfg)
    low_meta = gate.episode_metadata[gate.episode_metadata["tracklet_id"].eq("t_low")].iloc[0]
    assert low_meta["selective_terminal_status"] == "REJECTED_BY_TEMPORAL_ABSENCE"
    assert int(low_meta["hard_veto_rejection_count"]) == 0


def test_selective_quarantine_uses_episode_start_history_only():
    det = pd.DataFrame(
        [
            _row("warmup1", "warmup1", 1, 0.95),
            _row("warmup1b", "warmup1", 2, 0.01),
            _row("warmup2", "warmup2", 3, 0.90),
            _row("probe", "probe", 4, 0.925),
        ]
    )
    cfg = SelectiveTrustQuarantineConfig(
        confidence_floor=0.05,
        relative_weight=1.0,
        relative_confidence_min_history=2,
        support_history_min=2,
        quarantine_fraction_min=0.0,
        quarantine_fraction_max=0.0,
        fast_pass_support=1.1,
    )
    gate = selective_quarantine_gate_result(det, _manifest(5), cfg)
    probe_meta = gate.episode_metadata[gate.episode_metadata["tracklet_id"].eq("probe")].iloc[0]
    # Start-support history contains only the two high-confidence episode
    # starts. If the low-confidence continuation leaked, the rank would be > 0.
    assert probe_meta["episode_start_rank"] == 0.0


def test_selective_quarantine_hard_budget_limits_realized_start_fraction():
    det = pd.DataFrame([_row(f"d{i}", f"t{i}", i + 1, 0.10) for i in range(50)])
    cfg = SelectiveTrustQuarantineConfig(
        confidence_floor=0.05,
        relative_weight=0.0,
        support_history_min=1,
        quarantine_fraction_min=0.05,
        quarantine_fraction_max=0.05,
        fast_pass_support=0.99,
        confirmation_support=0.99,
        maximum_quarantine_frames=1,
        timeout_policy="release_to_baseline",
        budget_capacity=2.0,
    )
    gate = selective_quarantine_gate_result(det, _manifest(60), cfg)
    meta = gate.episode_metadata
    quarantined = meta["terminal_reason"].astype(str).str.contains("quarantine|timeout_release", regex=True).sum()
    assert quarantined <= 4
    assert float(meta["realized_quarantine_fraction"].max()) <= 0.05 + 1 / 50


def test_selective_quarantine_releases_two_hits_to_baseline_after_timeout():
    det = pd.DataFrame(
        [
            _row("warmup", "warmup", 1, 0.95),
            _row("low1", "t_low", 2, 0.10),
            _row("low2", "t_low", 3, 0.10),
            _row("future", "t_low", 5, 0.10),
        ]
    )
    cfg = SelectiveTrustQuarantineConfig(
        confidence_floor=0.05,
        critical_confidence_floor=0.02,
        relative_weight=0.0,
        support_history_min=1,
        quarantine_fraction_min=1.0,
        quarantine_fraction_max=1.0,
        fast_pass_support=0.99,
        confirmation_support=0.99,
        maximum_quarantine_frames=1,
        buffer_backfill=True,
    )
    gate = selective_quarantine_gate_result(det, _manifest(6), cfg)
    assert gate.acceptance_mask.tolist() == [True, True, True, True]
    assert gate.online_acceptance_mask.tolist() == [True, False, False, True]
    low_meta = gate.episode_metadata[gate.episode_metadata["tracklet_id"].eq("t_low")].iloc[0]
    assert low_meta["selective_terminal_status"] == "RELEASED_TO_BASELINE"


def test_selective_quarantine_rejects_hard_veto_on_frame_without_detection():
    det = pd.DataFrame([_row("warmup", "warmup", 1, 0.95), _row("veto", "t_veto", 2, 0.01)])
    cfg = SelectiveTrustQuarantineConfig(
        confidence_floor=0.05,
        critical_confidence_floor=0.02,
        relative_weight=0.0,
        support_history_min=1,
        quarantine_fraction_min=1.0,
        quarantine_fraction_max=1.0,
        fast_pass_support=0.99,
        confirmation_support=0.99,
        maximum_quarantine_frames=1,
    )
    gate = selective_quarantine_gate_result(det, _manifest(5), cfg)
    meta = gate.episode_metadata[gate.episode_metadata["tracklet_id"].eq("t_veto")].iloc[0]
    assert meta["selective_terminal_status"] == "REJECTED_BY_HARD_VETO"
    assert meta["negative_evidence_type"] == "hard_veto"
    assert not bool(gate.acceptance_mask.loc[det.index[1]])


def test_selective_quarantine_right_censors_sequence_end():
    det = pd.DataFrame([_row("warmup", "warmup", 1, 0.95), _row("late", "t_late", 3, 0.10)])
    cfg = SelectiveTrustQuarantineConfig(
        confidence_floor=0.05,
        critical_confidence_floor=0.02,
        relative_weight=0.0,
        support_history_min=1,
        quarantine_fraction_min=1.0,
        quarantine_fraction_max=1.0,
        fast_pass_support=0.99,
        confirmation_support=0.99,
        maximum_quarantine_frames=2,
        buffer_backfill=True,
    )
    gate = selective_quarantine_gate_result(det, _manifest(3), cfg)
    meta = gate.episode_metadata[gate.episode_metadata["tracklet_id"].eq("t_late")].iloc[0]
    assert meta["selective_terminal_status"] == "CENSORED_AT_SEQUENCE_END"
    assert bool(meta["right_censored"])
    assert bool(gate.acceptance_mask.loc[det.index[1]])
    assert not bool(gate.online_acceptance_mask.loc[det.index[1]])
    assert not gate.episode_metadata["selective_terminal_status"].astype(str).eq("pending_end").any()
