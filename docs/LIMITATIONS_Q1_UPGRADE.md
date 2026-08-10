# Q1 Upgrade Limitations

- VisDrone is single-camera UAV video, not a multi-UAV swarm dataset.
- This experiment validates real-detector noise handling, not inter-agent geometry.
- The corruption robustness script applies deterministic detector-output perturbations unless a full image-level detector rerun is added.
- RF remains a supervised upper baseline when enough labeled calibration data is available.
- In the controlled replay benchmark, the Bayesian existence filter is numerically stronger than the proposed trust layer; do not claim universal dominance over simple fusion baselines.
- The confidence-threshold baseline can reduce false-new tracks, but at the tested operating points it trades away F1 faster than the trust layer.
- The safest article wording is that the trust layer provides an interpretable, no-label operating point rather than a globally best tracker.
- `q1_final_plus` is a legacy protocol and must not be mixed with Selective Trust Quarantine v2.1 confirmatory tables.
- Do not claim Q1 acceptance, real swarm validation, or RF superiority without table support.
