# Real-Detector VisDrone Validation

This experiment evaluates the trust layer on real YOLO detections from VisDrone2019-VID-val.

Scope:

- It is a single-UAV real-detector validation.
- It checks behavior under real neural detector errors.
- It is not a real multi-UAV or swarm validation.

Primary methods:

- `s_naive`
- `persistence_gate`
- `bayesian_existence_filter`
- `ema_confidence_gate`
- `bytetrack`
- `s2_logodds_temporal`
- `geometry_dynamic_no_multiagent`
- `rf_learned_gate`

`geometry_dynamic_no_multiagent` disables inter-agent and epipolar support. It uses confidence, temporal linking, kinematic consistency, and log-odds style persistence.

Claim-safe reading:

- Compare the trust layer only against the simple confidence-threshold and simple fusion baselines implemented in this protocol.
- These results do not prove superiority over ByteTrack, the Bayesian existence filter, learned classifiers, modern trackers, or cooperative perception methods.
- The defensible claim is that the trust layer provides an interpretable, conservative operating point under the current real-detector protocol.
