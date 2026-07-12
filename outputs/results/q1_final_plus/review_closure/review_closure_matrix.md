# Q1 Review Closure Matrix

Statuses: closed/supported indicate evidence exists; partial/limitation indicate claim narrowing is required.

| ID | Status | Evidence | Reviewer issue | Allowed wording | Forbidden wording |
| --- | --- | --- | --- | --- | --- |
| R01 | limitation | complete | Pseudo-swarm is not real swarm validation | VisDrone is single-camera real-detector validation; controlled replay is a track-initiation extension. | Validated real UAV swarm; replacement for synchronized multi-UAV datasets. |
| R02 | supported | complete | F1 gain is small; task should focus on false-new tracks | The trust layer reduces false-new tracks with bounded F1 loss; matched-cleanliness comparisons are primary. | The method improves overall tracking F1. |
| R03 | supported | complete | Confidence threshold may explain the effect | At comparable false-new counts, trust preserves more F1 than the tested fixed confidence threshold. | Trust universally dominates all confidence thresholds. |
| R04 | limitation | complete | Bayesian filter dominates the selected trust point | Bayesian existence is the stronger numerical baseline in controlled replay; trust remains interpretable/no-label. | Bayesian requires a learned model; trust beats simple fusion baselines. |
| R05 | closed | complete | Simple fusion baselines are not cooperative perception SOTA | Simple fusion rules are track-confirmation references, not CoBEVFusion/V2X-Real analogues. | Simple fusion baselines represent cooperative perception SOTA. |
| R06 | partial | complete | Min-rule guarantee does not cover temporal recovery | Strict mode is noncompensatory; balanced mode uses temporal recovery and weakens the guarantee. | The complete deployed decision rule is fully noncompensatory. |
| R07 | supported | complete | Kinematic feature may be redundant | Geometry/temporal channels are the main practical contributors; kinematic support is not claimed as dominant. | Each channel independently improves the result. |
| R08 | partial | complete | Label scarcity does not prove the niche by F1 alone | Low-label/stale-calibration niche is reported with caveats and supporting diagnostics. | Trust is better than RF under all label budgets. |
| R09 | supported | complete | Need map-contamination metrics, not only false-new count | Report observed false-track occupancy and initiation precision as map-contamination proxies. | Full tracker lifecycle occupancy is measured if only observed-row proxy exists. |
| R10 | closed | complete | Failure-case subgroup F1 is not standard object-level F1 | Failure-case tables are candidate-level diagnostics. | Subgroup diagnostics are standard object-level F1 or speed robustness. |
| R11 | partial | complete | Five seeds are not independent scene repeats | Repeated seeds are deterministic confidence-perturbation trials, not independent scenes. | Five seeds prove independent simulation reproducibility. |
| R12 | partial | complete | Algorithm must be self-contained | Use the code-level contract and strict/balanced terminology for the algorithm description. | Q=min(c,k,s) alone describes the full implementation. |
| R13 | partial | complete | Runtime claims need decomposition | Runtime is implementation-dependent; vectorized aggregation is a microbenchmark over prepared features. | 5 ms/frame is generally acceptable for UAV deployment. |
| R14 | supported | complete | Reproducibility bundle must be auditable | The bundle provides scripts, selected outputs, manifest/checksum references, and claim-safe notes. | The bundle alone contains raw images, detector JSON, and all external dependencies. |
