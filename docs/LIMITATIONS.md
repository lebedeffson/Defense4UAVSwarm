# Limitations

1. The benchmark is a synthetic pseudo-swarm built from transformed VisDrone views, not a real multi-UAV dataset.
2. The controlled attack model is detection-level and diagnostic. It does not fully model physical adversarial perturbations, camera desynchronization, or real extrinsic calibration drift.
3. The selected method is intentionally lightweight and interpretable. Its main benefit is reducing false positives and false new-track initiations, not producing a large F1 jump.
4. XAI is implemented and evaluated only as a diagnostic module. It is not part of the selected method.
5. The learned FP-classifier baseline uses calibration labels and is a training-based comparator. It should be interpreted separately from the training-free T-norm method.
6. Runtime depends strongly on hardware, YOLO backend, tracker implementation, and whether detections are cached.
7. AirSim or real multi-UAV validation remains future work unless `outputs/results/v6_airsim` is explicitly provided.
