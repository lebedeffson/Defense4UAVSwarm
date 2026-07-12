# Trust-Initiation Algorithm Contract

This contract is the code-aligned wording source for the paper algorithm block.

## Inputs

Each detector/tracker candidate row provides:

- detector confidence `c_i`;
- temporal age / persistence evidence;
- optional kinematic consistency `k_i`;
- optional geometric or inter-agent support `s_i`;
- tracklet identity and frame index.

VisDrone is a single-camera protocol. In VisDrone, geometric/inter-agent support
must be described as a temporal/geometric proxy, not true inter-agent
consistency.

## Strict Mode

Strict mode computes component scores in `[0, 1]` and applies a
noncompensatory rule:

```text
Q_i = min(c_i, k_i, t_i, s_i)
accept_i = Q_i >= tau
```

In this mode, a weak component cannot be hidden by a strong component. This is
the only mode for which the noncompensation guarantee should be claimed.

## Balanced Operational Mode

Balanced mode may include temporal recovery:

```text
accept_i =
  (Q_i >= tau)
  OR (temporal_age_i >= recovery_min_age AND c_i >= recovery_min_confidence)
```

This mode is operationally useful for recall preservation, but it is not fully
noncompensatory because the recovery branch can accept candidates that fail the
strict min rule.

Safe wording:

> Strict mode is noncompensatory. Balanced mode relaxes this property through
> temporal recovery to recover recall.

Forbidden wording:

> The full operational decision rule is noncompensatory.

## Confidence Correction

When a corrected confidence is reported, it should be described as a trust-gated
operating point, not a calibrated detection probability. The rule is a screening
decision for track initiation, not an object existence posterior.

## Channel Claims

The current ablation supports a limited claim:

- temporal/geometric support drive most of the observed reduction in false
  initiations;
- `k_i` is not claimed as an independently dominant channel in the current
  settings;
- controlled multi-agent replay and VisDrone single-camera validation must not
  be merged into one real-swarm claim.

## Pseudocode Template

```text
for each candidate i:
    c_i <- detector confidence
    t_i <- temporal evidence from tracklet age
    k_i <- kinematic consistency if available else neutral value
    s_i <- geometric/inter-agent support if available else neutral value

    Q_i <- min(c_i, k_i, t_i, s_i)

    if mode == strict:
        accept_i <- Q_i >= tau
    else if mode == balanced:
        accept_i <- (Q_i >= tau) OR temporal_recovery(i)

    if accepted:
        emit candidate to map/tracker output
    else:
        suppress or delay candidate depending on experiment protocol
```

Any paper version should state all thresholds, selected mode, calibration split,
holdout split, and whether the result is strict or balanced.
