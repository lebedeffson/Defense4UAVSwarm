# METHOD V9 GEOMDYN

V9 adds a geometry-dynamic temporal trust layer on top of the v8 controlled multi-UAV simulation.

The selected candidate combines:

- T-norm confidence reweighting;
- log-odds pending track memory;
- Mahalanobis-style kinematic consistency;
- bottom-center projection to the world ground plane;
- soft cross-agent geometric support;
- optional soft epipolar support.

V9 is a training-free method. `S2_learned_fp_gate` remains a supervised RF baseline trained on calibration data only.

Runtime gates must not use GT labels. GT/source labels are used only for evaluation and supervised RF training.
