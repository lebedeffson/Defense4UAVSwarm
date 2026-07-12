# Simple Fusion Baselines Claim-Safe Notes

Actual controlled input source: `custom_uav_swarm_v8`.

```text
                   method       F1  false_new_tracks
        S2_tnorm_temporal 0.911410             562.4
           S2_v9_selected 0.891547             122.0
bayesian_existence_filter 0.915822              99.0
          iou_fusion_only 0.877839             306.2
    max_confidence_fusion 0.877879             342.0
   mean_confidence_fusion 0.877763             330.0
              naive_union 0.867629            4016.2
```

- Proposed vs naive_union: F1 delta=+0.023918, false_new delta=-3894.2.
- Proposed vs mean_confidence_fusion: F1 delta=+0.013784, false_new delta=-208.0.
- Proposed vs max_confidence_fusion: F1 delta=+0.013668, false_new delta=-220.0.
- Proposed vs bayesian_existence_filter: F1 delta=-0.024275, false_new delta=+23.0.
- Bayesian existence filter dominates proposed point: yes.

Safe claim: the proposed layer is compared with simple fusion baselines and provides an interpretable no-label operating point in this controlled replay.
If the Bayesian existence filter dominates the proposed point, report it as the stronger numerical baseline and keep the trust-layer claim to interpretability/noncompensatory semantics.
Forbidden claim: do not call these baselines cooperative perception SOTA and do not claim replacement for CoBEVFusion/V2X-Real.
