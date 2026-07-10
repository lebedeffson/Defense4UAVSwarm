# Defense4UAVSwarm q1_v5 reproducibility package

No new datasets are required. The package uses existing VisDrone annotations,
saved YOLO detections/features, and frozen v9 tracker-comparison summaries.

Run `bash reproducibility/q1_v5/commands.sh` from the repository root.

Important scope notes:
- VisDrone is single-camera UAV validation, not real multi-UAV validation.
- q1_v5 operating curves are candidate-space analyses from `feature_audit.csv`.
- frozen v9 tracker-comparison numbers are kept separate in `legacy_freeze`.
