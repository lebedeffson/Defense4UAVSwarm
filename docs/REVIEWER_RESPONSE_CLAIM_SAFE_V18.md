# Claim-Safe Reviewer Response Notes for v18

This file is the safe wording source for replies to the methodological review.
It intentionally avoids journal-rank promises and overclaims.

## Scope

We agree that the current study does not validate a real UAV swarm. The
multi-agent part is a controlled replay/extension for track-initiation logic,
not a substitute for synchronized multi-UAV data. The real-detector evidence is
VisDrone single-camera video.

## Main Positioning

The method should be described as an interpretable, label-free track-initiation
filter. It is not a globally best tracker and it is not a replacement for
trained filters when fresh labeled calibration data are available.

Safe wording:

> The proposed trust layer targets a constrained operating point: reducing
> false track initiations while keeping the F1 loss small. It is intended for
> low-label or stale-calibration settings where an interpretable rule is useful.

Avoid:

> The method improves overall tracking quality.
> The method is better than all baselines.
> The paper is ready for Q1/Q2 submission.

## Confidence Threshold Baseline

The strongest VisDrone claim is the matched-cleanliness comparison:

- ByteTrack: F1 = 0.453987, false-new tracks = 6838.
- ByteTrack + trust: F1 = 0.450787, false-new tracks = 4799.
- Confidence threshold 0.2: F1 = 0.435418, false-new tracks = 4581.

Safe wording:

> At comparable false-new track counts, the trust layer preserves more F1 than
> a fixed confidence-threshold baseline in the tested YOLOv8s VisDrone sweep.

Avoid:

> The trust layer universally dominates confidence thresholding.

## Bayesian Baseline

The Bayesian existence filter is a simple, transparent, no-label baseline in
the controlled replay. In the v18 controlled replay it is numerically stronger
than the selected trust point:

- Bayesian existence filter: F1 = 0.915822, false-new tracks = 99.0.
- Selected trust point: F1 = 0.891547, false-new tracks = 122.0.

Safe wording:

> In the controlled replay, the Bayesian existence filter is the stronger
> numerical baseline. The trust layer remains useful as an interpretable
> noncompensatory rule and as a constrained operating-point demonstration, not
> as the best numerical method in that table.

Avoid:

> The Bayesian filter requires a model.
> The Bayesian filter is less interpretable.
> The trust layer beats simple fusion baselines.

## Simple Fusion Baselines

The simple fusion baselines are reference rules for track-initiation
confirmation. They are not analogues of CoBEVFusion, V2X-Real, OPV2V, or other
full cooperative perception systems.

Safe wording:

> We compare against simple track-confirmation and fusion-style reference rules.
> Full cooperative perception baselines are outside the current data modality
> and remain future work.

Avoid:

> These baselines represent cooperative perception SOTA.
> The experiment replaces CoBEVFusion/V2X-Real comparison.

## Min Rule and Recovery

The min rule is noncompensatory only for the strict score. Any temporal recovery
branch creates a balanced operational mode and weakens the formal
noncompensation guarantee.

Safe wording:

> We distinguish strict noncompensatory scoring from the balanced operational
> mode that includes temporal recovery for recall preservation.

Avoid:

> The complete deployed decision rule is fully noncompensatory.

## Failure-Case Diagnostics

The failure-case bins are candidate-level diagnostics from saved detector
candidates. They do not include missed ground-truth objects that never became
detector candidates.

Tracklet length is not object speed. It should be called tracklet persistence,
not motion.

Safe wording:

> Failure-case tables are candidate-level diagnostics. Tracklet-persistence
> bins summarize candidate longevity, not object speed.

Avoid:

> The method is robust to object speed.
> The subgroup F1 values are standard object-level F1.

## Runtime

The 5 ms/frame value is an implementation-level measurement that includes more
than final aggregation. The vectorized microbenchmark only measures aggregation
over prepared features.

Safe wording:

> The measured runtime is implementation-dependent. Vectorization shows that
> the aggregation component is not the bottleneck, but full embedded runtime
> requires a separate end-to-end deployment benchmark.

Avoid:

> 5 ms is generally acceptable for UAV deployment.
> The method has negligible runtime overhead.

## Final Reply Tone

Use:

> We narrowed the scope, changed the primary claim to false-initiation
> reduction under bounded F1 loss, and report stronger baselines where they
> dominate.

Do not use:

> All remarks are resolved.
> The article is ready for Q2/Q1.
