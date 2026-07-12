# Failure Case Mining Claim-Safe Notes

Diagnostics are candidate-level bins from saved VisDrone detections. They do not reassign absent GT objects to bins.
Most negative F1 delta by size: `medium`.
Most negative F1 delta by confidence: `medium_confidence`.
Most negative F1 delta by density: `high_density`.
Most negative F1 delta by tracklet persistence: `short_tracklet`.
Tracklet persistence is not object speed. It is a diagnostic grouping by candidate tracklet length.

Safe claim: use these as failure diagnostics, not as formal robustness proof.
Forbidden claim: do not claim speed or motion robustness from tracklet-length bins.
