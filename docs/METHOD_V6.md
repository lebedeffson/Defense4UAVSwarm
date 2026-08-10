# METHOD_V6: Temporal-Persistent Trust

The v6 method keeps the selected v5 trust layer unchanged and adds a pending-track confirmation stage.

Base trust:

```text
Q_i = min(c_i, k_i, s_i)
confidence_reweighted = confidence * (0.6 + 0.4 * Q_i)
```

Temporal selected policy:

```text
scenario = S2_tnorm_temporal
temporal_window = 3
temporal_iou_threshold = 0.15
confirmation_threshold = 0.4
min_hits_to_confirm = 2
new_track_threshold = 0.5
```

New candidates are not immediately promoted to confirmed tracks. They enter a pending state and are confirmed only if a compatible detection appears within the temporal window. Existing confirmed tracks are not deleted by the temporal gate.

This implementation is a deterministic replay over the existing `swarm_feature_audit.csv` artifacts. It is suitable for policy evaluation and paper-table generation, but a production tracker should integrate the pending state directly inside the tracker loop.
