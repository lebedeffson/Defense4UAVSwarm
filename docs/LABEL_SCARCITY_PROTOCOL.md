# Label-Scarcity Protocol

The label-scarcity experiment uses sequence/chunk-level calibration splits, not random frame sampling.

Budgets:

- `0.00`
- `0.01`
- `0.05`
- `0.10`
- `0.25`
- `1.00`

Rules:

- RF is unavailable at `0%` labels.
- RF is trained only on selected calibration chunks.
- Holdout chunks are disjoint from calibration chunks.
- Training-free methods use fixed parameters and do not tune on VisDrone holdout labels.

This protocol is intended to show the practical niche of training-free trust layers under no-label or low-label calibration.
