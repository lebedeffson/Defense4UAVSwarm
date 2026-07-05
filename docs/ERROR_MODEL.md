# Controlled Pseudo-Swarm Error Model

This project uses a controlled detection-level pseudo attack for synthetic pseudo-swarm evaluation. It is not a physical adversarial attack and not a real multi-UAV sensor fault model.

Configuration: `configs/pseudo_attack_v2.yaml`.

Runtime seed: `attack_model.seed` or CLI `--pseudo-attack-seed`.

Pipeline position:

```text
clean pseudo detections -> controlled error injection -> c_i/k_i/s_i -> trust aggregation -> tracking gate
```

GT is used only to generate controlled pseudo-errors and evaluate metrics, not as a runtime feature for the selected method.

## high_conf_fp

Rule: for each `(sequence_id, frame_id, agent_id)`, add a false detection with probability `per_frame_rate`.

Parameters:

- confidence: uniform in `confidence_range`
- size source: based on matched GT-like/pseudo detection distribution
- placement: near existing objects
- target constraint: low overlap with real objects, configured by `max_iou_with_gt`

Runtime effect:

- adds a new detection
- sets `track_id = -1`
- sets `eval_is_tp = false`
- marks event as `high_conf_fp`

## bbox_shift_tp

Rule: sample a fraction of TP detections and shift their boxes.

Formula:

```text
dx = sign_x * width(B) * U(shift_ratio_min, shift_ratio_max)
dy = sign_y * height(B) * U(shift_ratio_min, shift_ratio_max)
B' = B + (dx, dy)
confidence' = confidence * U(conf_min, conf_max)
```

Runtime effect:

- changes bbox coordinates
- keeps class identity
- may turn `eval_is_tp` false if shifted IoU falls below the TP threshold

## agent_specific_drop

Rule: sample objects visible in multiple pseudo-agents and drop the detection in one random agent.

Runtime effect:

- removes one agent observation
- keeps other agents unchanged
- marks `agent_specific_drop`

## class_confusion

Rule: sample TP detections and replace class with a compatible class.

Implemented compatible swaps:

```text
pedestrian <-> people
bicycle <-> motor
```

Runtime effect:

- changes `class_id` and `class_name`
- keeps bbox approximately unchanged
- confidence is multiplied by configured range
- marks `eval_is_tp = false` for class-aware analysis

## inter_agent_misalignment

Rule: sample one-agent observations and shift their bbox coordinates relative to other agents.

Formula:

```text
dx = sign_x * width(B) * U(r_min, r_max)
dy = sign_y * height(B) * U(r_min, r_max)
B' = B + (dx, dy)
```

Runtime effect:

- changes bbox coordinates for a selected agent only
- decreases inter-agent consistency `s_i`
- does not edit camera transforms

## Calibration-noise diagnostic

The v4.2.1 calibration-noise diagnostic is separate from `inter_agent_misalignment`.

It applies a static projected-bbox offset after pseudo-attack injection and before `compute_inter_agent_consistency()`:

```text
for each (sequence_id, agent_id):
  dx, dy ~ Uniform(-p, p)
  x1, x2 <- x1 + dx, x2 + dx
  y1, y2 <- y1 + dy, y2 + dy
```

This tests sensitivity of inter-agent matching and `s_i` to synthetic static projected-coordinate offsets. It does not prove robustness to real extrinsic calibration errors, perspective effects, or time synchronization errors.
