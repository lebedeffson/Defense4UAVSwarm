# Q1 Upgrade Limitations

- VisDrone is single-camera UAV video, not a multi-UAV swarm dataset.
- This experiment validates real-detector noise handling, not inter-agent geometry.
- The corruption robustness script applies deterministic detector-output perturbations unless a full image-level detector rerun is added.
- RF remains a supervised upper baseline when enough labeled calibration data is available.
- In the controlled replay benchmark, the Bayesian existence filter is numerically stronger than the proposed trust layer; do not claim universal dominance over simple fusion baselines.
- The confidence-threshold baseline can reduce false-new tracks, but at the tested operating points it trades away F1 faster than the trust layer.
- Simple fusion baselines are reference rules for track-initiation confirmation, not analogues of CoBEVFusion, V2X-Real, or full cooperative perception systems.
- Failure-case bins are candidate-level diagnostics. Do not report them as standard object-level F1, and do not infer motion robustness from tracklet-length bins.
- Do not claim Q1 acceptance, real swarm validation, or RF superiority without table support.
- The safest article wording is that the trust layer provides an interpretable, no-label operating point rather than a globally best tracker.
