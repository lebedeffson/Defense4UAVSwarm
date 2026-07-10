# Defense4UAVSwarm q1_v5 reproducibility package

No new datasets are required. The package uses existing VisDrone annotations,
saved YOLO detections/features, and frozen v9 tracker-comparison summaries.

Run `bash reproducibility/q1_v5/commands.sh` from the repository root.

Important scope notes:
- VisDrone is single-camera UAV validation, not real multi-UAV validation.
- q1_v5 operating curves are candidate-space analyses from `feature_audit.csv`.
- frozen v9 tracker-comparison numbers are kept separate in `legacy_freeze`.
- TrustGuard v5.2 ablation and confidence-shift checks are candidate-space
  diagnostics; use them for architecture selection/limitations, not as new
  tracker-level VisDrone claims.
- Baseline-dominance and calibration-staleness reports are claim-boundary
  checks: keep Bayesian/RF results even when they limit the trust-layer claim.
