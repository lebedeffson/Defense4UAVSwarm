# Failure Case Mining Claim-Safe Notes

Diagnostics are candidate-level bins from saved VisDrone detections. They do not reassign absent GT objects to bins.
Most negative F1 delta by size: `medium`.
Most negative F1 delta by confidence: `medium_confidence`.
Most negative F1 delta by density: `high_density`.
Motion analysis: proxy only. Tracklet length is used as a motion/persistence proxy; true FP objects do not have GT motion.

Safe claim: use these as failure diagnostics, not as formal robustness proof.
Forbidden claim: do not claim speed robustness from this proxy analysis.
