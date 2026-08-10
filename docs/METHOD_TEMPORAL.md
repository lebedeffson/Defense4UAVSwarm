# Temporal Trust Method

The v7 selected method is `S2_tnorm_temporal`.

Base score:

```text
Q_i = min(c_i, k_i, s_i)
confidence_reweighted = confidence * (0.6 + 0.4 * Q_i)
```

Temporal confirmation:

```text
new_candidate -> pending_track -> confirmed_track
```

A pending track is confirmed when a compatible detection appears within the configured temporal window:

```text
IoU(pending_bbox, detection_bbox) >= temporal_iou_threshold
confidence_reweighted >= confirmation_threshold
frame_delta <= temporal_window
hits >= min_hits_to_confirm
```

Selected v6/v7 replay parameters are stored in:

```text
configs/selected_s2_temporal.yaml
```

XAI is diagnostic-only and is not part of the selected v7 method.
