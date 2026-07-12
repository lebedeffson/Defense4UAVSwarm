# Final Q1 Practice Handoff

## Current Commit

- Commit: `a6757cd q1 add final practice handoff`
- Branch: `feature/q1-selective-v22-risk-prioritized`

## Final Bundles

- Slim review/practice bundle:
  `outputs/bundles/Defense4UAVSwarm_q1_final_plus_v18_practice_bundle.zip`
  - Size: `410883` bytes
  - SHA256: `483385572e4618c36b4be694b0df5ec7c3b71cb4773f9d0758ee6ed796b066d2`
- Refreshed all-data bundle:
  `outputs/bundles/Defense4UAVSwarm_q1_all_data_selective_v2_final.zip`
  - Size: `151005765` bytes
  - SHA256: `5dc41af0a73a2fc29159cb79647b44f830ed8b67e40c1655d05500309a570e90`

The all-data bundle includes Q1 outputs, detector JSON, configs, scripts, source,
tests, docs, and reproducibility metadata. It excludes raw VisDrone images,
external repositories, model weights, and nested zip archives.

## Primary Claim Boundary

The claim is not that the trust layer is globally best. The claim is:

> The method provides an interpretable, label-free track-initiation operating
> point that reduces false track initiations under bounded F1 loss.

Use matched-cleanliness comparison against confidence thresholding as the
strongest VisDrone result. Keep Bayesian/RF results in the paper when they are
stronger.

## Evidence Outputs

- Claim closure:
  `outputs/results/q1_final_plus/review_closure/review_closure_matrix.csv`
- Algorithm contract:
  `docs/TRUST_INITIATION_ALGORITHM_CONTRACT.md`
- Reviewer response wording:
  `docs/REVIEWER_RESPONSE_CLAIM_SAFE_V18.md`
- Bayesian/simple-fusion claim-safe note:
  `outputs/results/q1_final_plus/simple_fusion_baselines/simple_fusion_claim_safe.md`
- Failure diagnostics:
  `outputs/results/q1_final_plus/failure_case_mining/failure_case_claim_safe.md`
- Tracklet-persistence replacement for invalid motion proxy:
  `outputs/results/q1_final_plus/failure_case_mining/failure_by_tracklet_persistence.csv`

## Forbidden Article Claims

- Real UAV swarm validation.
- Universal superiority over Bayesian/RF/confidence threshold baselines.
- Simple fusion baselines as CoBEVFusion/V2X-Real analogues.
- Complete operational rule is fully noncompensatory.
- Subgroup diagnostics are standard object-level F1.
- Tracklet-length bins prove speed/motion robustness.
- 5 ms/frame is generally acceptable for UAV deployment.
- The paper is ready for Q1/Q2 solely on the current evidence.

## Checks

- `pytest -q tests/q1_v5`: passed.
- `compileall` on changed scripts: passed.
- Slim bundle integrity: passed.
- All-data bundle integrity: passed.
