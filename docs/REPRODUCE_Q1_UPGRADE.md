# Reproduce Q1 Upgrade

Run YOLO detections:

```bash
/home/lebedeffson/Code/venv/bin/python scripts/run_real_detector_visdrone.py \
  --dataset-root data/visdrone/VisDrone2019-VID-val \
  --model yolov8n \
  --device cuda \
  --conf 0.05 \
  --iou 0.7 \
  --output outputs/results/q1_real_detector/detections/yolov8n_visdrone.json \
  --runtime-output outputs/results/q1_real_detector/detections/yolov8n_runtime.csv
```

Run main trust comparison:

```bash
/home/lebedeffson/Code/venv/bin/python scripts/run_q1_real_detector_trust_experiment.py \
  --dataset-root data/visdrone/VisDrone2019-VID-val \
  --detections outputs/results/q1_real_detector/detections/yolov8n_visdrone.json \
  --methods s_naive persistence_gate bayesian_existence_filter ema_confidence_gate bytetrack s2_logodds_temporal geometry_dynamic_no_multiagent rf_learned_gate \
  --output-dir outputs/results/q1_real_detector/yolov8n_main
```

Run label scarcity:

```bash
/home/lebedeffson/Code/venv/bin/python scripts/run_q1_label_scarcity.py \
  --dataset-root data/visdrone/VisDrone2019-VID-val \
  --detections outputs/results/q1_real_detector/detections/yolov8n_visdrone.json \
  --label-budgets 0 0.01 0.05 0.10 0.25 1.0 \
  --seeds 11 22 33 44 55 \
  --split-mode sequence_chunks \
  --methods s_naive persistence_gate bayesian_existence_filter ema_confidence_gate bytetrack s2_logodds_temporal geometry_dynamic_no_multiagent rf_learned_gate \
  --output-dir outputs/results/q1_label_scarcity/yolov8n
```

Current curated v18 practice bundle:

- `outputs/bundles/Defense4UAVSwarm_q1_final_plus_v18_practice_bundle.zip`

Current claim-safe result folders:

- `outputs/results/q1_final_plus/confidence_threshold_baseline_yolov8s/`
- `outputs/results/q1_final_plus/tradeoff_yolov8s/`
- `outputs/results/q1_final_plus/simple_fusion_baselines/`
- `outputs/results/q1_final_plus/feature_ablation_final_controlled/`
- `outputs/results/q1_final_plus/failure_case_mining/`
- `outputs/results/q1_final_plus/runtime_vectorization/`
- `outputs/results/q1_final_plus/review_closure/`

Review-closure artifacts:

```bash
/home/lebedeffson/Code/venv/bin/python scripts/build_q1_review_closure_matrix.py \
  --output-dir outputs/results/q1_final_plus/review_closure
```
