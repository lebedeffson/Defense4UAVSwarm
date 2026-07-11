from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class MethodMetadata:
    method_id: str
    display_name: str
    requires_labels: bool
    requires_training: bool
    uses_temporal_memory: bool
    uses_inter_agent_geometry: bool
    interpretable_rule: bool
    scope: Literal["visdrone", "controlled_multiagent", "both"]


METHODS: dict[str, MethodMetadata] = {
    "tracker_baseline": MethodMetadata("tracker_baseline", "Tracker baseline", False, False, False, False, True, "both"),
    "confidence_threshold": MethodMetadata("confidence_threshold", "Confidence threshold", False, False, False, False, True, "both"),
    "m_of_n_confirmation": MethodMetadata("m_of_n_confirmation", "M-of-N confirmation", False, False, True, False, True, "both"),
    "trust_strict": MethodMetadata("trust_strict", "Strict trust layer", False, False, False, False, True, "both"),
    "trust_balanced": MethodMetadata("trust_balanced", "Balanced trust layer", False, False, True, False, True, "both"),
    "trust_adaptive": MethodMetadata("trust_adaptive", "Adaptive trust layer", False, False, True, False, True, "visdrone"),
    "bayesian_fixed": MethodMetadata("bayesian_fixed", "Fixed Bayesian existence", False, False, True, False, True, "both"),
    "bayesian_calibrated": MethodMetadata("bayesian_calibrated", "Calibrated Bayesian existence", True, False, True, False, True, "both"),
    "rf_filter": MethodMetadata("rf_filter", "RF learned filter", True, True, True, False, False, "both"),
    "naive_union": MethodMetadata("naive_union", "Naive union", False, False, False, True, True, "controlled_multiagent"),
    "iou_fusion_only": MethodMetadata("iou_fusion_only", "IoU fusion only", False, False, False, True, True, "controlled_multiagent"),
    "mean_confidence_fusion": MethodMetadata("mean_confidence_fusion", "Mean confidence fusion", False, False, False, True, True, "controlled_multiagent"),
    "max_confidence_fusion": MethodMetadata("max_confidence_fusion", "Max confidence fusion", False, False, False, True, True, "controlled_multiagent"),
    "trust_strict_multiagent": MethodMetadata("trust_strict_multiagent", "Strict multi-agent trust", False, False, False, True, True, "controlled_multiagent"),
    "trust_balanced_multiagent": MethodMetadata("trust_balanced_multiagent", "Balanced multi-agent trust", False, False, True, True, True, "controlled_multiagent"),
    "risk_prioritized_two_stage_quarantine": MethodMetadata("risk_prioritized_two_stage_quarantine", "Risk-prioritized two-stage quarantine", True, True, True, False, True, "visdrone"),
}


def get_method(method_id: str) -> MethodMetadata:
    try:
        return METHODS[method_id]
    except KeyError as exc:
        raise KeyError(f"Unknown q1_v5 method: {method_id}") from exc
