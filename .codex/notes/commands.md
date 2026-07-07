# Commands

Add only verified project-specific commands.

When checking U2UData integration, run:

```bash
/home/lebedeffson/Code/venv/bin/python scripts/inventory_hf_repo.py \
  --repo-id fengtt42/U2UData-2 \
  --repo-type dataset \
  --output outputs/results/v7_2_data_search/u2udata2_hf_inventory.csv \
  --report outputs/results/v7_2_data_search/u2udata2_hf_inventory_report.md

/home/lebedeffson/Code/venv/bin/python scripts/download_u2u_minimal_subset.py \
  --repo-id fengtt42/U2UData-2 \
  --include-archives \
  --max-files 3 \
  --max-file-mb 5000 \
  --max-total-gb 12 \
  --min-free-disk-gb 25 \
  --retries 10 \
  --output-root data/U2UData

/home/lebedeffson/Code/venv/bin/python scripts/extract_u2u_rar_subset.py \
  --archives data/U2UData/Scene_Sunny_Rain_drone_1.rar \
             data/U2UData/Scene_Sunny_Rain_drone_2.rar \
             data/U2UData/Scene_Sunny_Rain_drone_3.rar \
  --output-root data/U2UData/extracted \
  --stream-regex 'front_center_0|airsim_rec' \
  --max-frames 300 \
  --min-free-disk-gb 15

/home/lebedeffson/Code/venv/bin/python scripts/check_u2u_dataset.py \
  --dataset-root data/U2UData \
  --output outputs/results/v7_u2u/dataset_check/u2u_dataset_report.json
```

When building a U2U manifest, run:

```bash
/home/lebedeffson/Code/venv/bin/python scripts/build_u2u_manifest.py \
  --dataset-root data/U2UData \
  --split validation \
  --agents 3 \
  --min-frames 300 \
  --output-root data/manifests/u2u_v7
```

For the current U2UData-2 `Scene_Sunny_Rain` subset, only 139 synchronized 3-agent frames are available and no GT boxes are present. Use `--min-frames 139` only for structure/manifest validation, not for the final Q1 experiment.

When looking for the original U2UData benchmark package, first inventory the HF repo. As of v7.2, `fengtt42/U2UData` is not available as a HF dataset repo and `fengtt42/U2UData-2` exposes recorder `.rar` archives only, with no labels/annotations/OpenCOOD benchmark files.

When checking the 2D projection route, run:

```bash
/home/lebedeffson/Code/venv/bin/python scripts/project_u2u_3d_boxes_to_2d.py \
  --manifest data/manifests/u2u_v7/manifest.json \
  --output data/manifests/u2u_v7/gt_2d_projected.json \
  --min-visible-corners 4 \
  --clip-to-image \
  --output-report outputs/results/v7_u2u_projection/projection_report.csv
```

When packaging the v7 Q1 closure bundle, run:

```bash
/home/lebedeffson/Code/venv/bin/python scripts/package_v7_q1_bundle.py
```

When running the controlled v8 custom UAV swarm validation, run:

```bash
/home/lebedeffson/Code/venv/bin/python scripts/generate_custom_uav_swarm_v8.py \
  --output-root data/custom_uav_swarm_v8 \
  --num-scenes 5 \
  --num-agents 3 \
  --frames-per-scene 300 \
  --objects-per-scene 25 \
  --image-width 1280 \
  --image-height 720 \
  --seed 2026 \
  --write-images \
  --write-gt-2d \
  --write-gt-3d \
  --write-poses \
  --write-calibration

/home/lebedeffson/Code/venv/bin/python scripts/generate_synthetic_detections_v8.py \
  --manifest data/custom_uav_swarm_v8/manifest.json \
  --gt-2d data/custom_uav_swarm_v8/gt_2d_boxes.json \
  --scenario combined_stress \
  --tp-detection-prob 0.85 \
  --bbox-jitter-px 5 \
  --fp-rate-per-frame 2.0 \
  --false-burst-prob 0.08 \
  --false-burst-min-duration 1 \
  --false-burst-max-duration 3 \
  --seed 2026 \
  --output outputs/results/v8_custom_swarm/detections/combined_stress_detections.json

/home/lebedeffson/Code/venv/bin/python scripts/run_v8_custom_swarm_experiment.py \
  --manifest data/custom_uav_swarm_v8/manifest.json \
  --gt-2d data/custom_uav_swarm_v8/gt_2d_boxes.json \
  --gt-3d data/custom_uav_swarm_v8/gt_3d_boxes.json \
  --detections outputs/results/v8_custom_swarm/detections/combined_stress_detections.json \
  --scenarios s_naive s2_tnorm_soft s2_tnorm_temporal s2_ema_confidence_gate s2_support_count_gate s2_learned_fp_gate \
  --selected-params configs/selected_s2_temporal.yaml \
  --split holdout \
  --output-dir outputs/results/v8_custom_swarm/main_holdout

/home/lebedeffson/Code/venv/bin/python scripts/package_v8_custom_swarm_bundle.py \
  --bundle-path outputs/bundles/Defense4UAVSwarm_v8_custom_uav_swarm_bundle.zip
```

When running the v9 geometry-dynamic temporal trust validation, run:

```bash
/home/lebedeffson/Code/venv/bin/python scripts/run_v9_geomdyn_experiment.py \
  --manifest data/custom_uav_swarm_v8/manifest.json \
  --gt-2d data/custom_uav_swarm_v8/gt_2d_boxes.json \
  --gt-3d data/custom_uav_swarm_v8/gt_3d_boxes.json \
  --detections outputs/results/v8_custom_swarm/detections/combined_stress_detections.json \
  --scenarios s_naive s2_tnorm_soft s2_tnorm_temporal s2_tnorm_temporal_logodds s2_tnorm_temporal_logodds_maha s2_tnorm_temporal_logodds_maha_world s2_tnorm_temporal_logodds_maha_world_epi s2_support_count_gate s2_ema_confidence_gate s2_learned_fp_gate s2_v9_selected \
  --calibration-scenes scene_001 scene_002 scene_003 \
  --holdout-scenes scene_004 scene_005 \
  --output-dir outputs/results/v9_geomdyn/main

/home/lebedeffson/Code/venv/bin/python scripts/run_v9_robustness_sweep.py \
  --manifest data/custom_uav_swarm_v8/manifest.json \
  --gt-2d data/custom_uav_swarm_v8/gt_2d_boxes.json \
  --gt-3d data/custom_uav_swarm_v8/gt_3d_boxes.json \
  --detections outputs/results/v8_custom_swarm/detections/combined_stress_detections.json \
  --scenarios s2_tnorm_temporal s2_v9_selected s2_learned_fp_gate \
  --selected-params outputs/results/v9_geomdyn/main/v9_selected_params.yaml \
  --sync-delay-frames 0 1 2 3 \
  --pose-noise-translation-m 0 0.5 1.0 2.0 \
  --pose-noise-yaw-deg 0 1 3 5 \
  --agent-dropout-prob 0.0 0.1 0.3 0.5 \
  --combined-stress \
  --output-dir outputs/results/v9_geomdyn/robustness

/home/lebedeffson/Code/venv/bin/python scripts/run_v9_runtime_benchmark.py \
  --manifest data/custom_uav_swarm_v8/manifest.json \
  --detections outputs/results/v8_custom_swarm/detections/combined_stress_detections.json \
  --scenarios s2_tnorm_temporal s2_v9_selected s2_learned_fp_gate \
  --selected-params outputs/results/v9_geomdyn/main/v9_selected_params.yaml \
  --output-dir outputs/results/v9_geomdyn/runtime
```

When running the Q1 VisDrone real-detector/label-scarcity smoke path, run:

```bash
/home/lebedeffson/Code/venv/bin/python scripts/run_real_detector_visdrone.py \
  --dataset-root data/visdrone/VisDrone2019-VID-val \
  --model yolov8n.pt \
  --device auto \
  --conf 0.05 \
  --iou 0.7 \
  --limit-frames 20 \
  --output outputs/results/q1_real_detector/detections/yolov8n_visdrone_smoke.json \
  --runtime-output outputs/results/q1_real_detector/detections/yolov8n_runtime_smoke.csv

/home/lebedeffson/Code/venv/bin/python scripts/run_q1_real_detector_trust_experiment.py \
  --dataset-root data/visdrone/VisDrone2019-VID-val \
  --detections outputs/results/q1_real_detector/detections/yolov8n_visdrone_smoke.json \
  --methods s_naive persistence_gate bayesian_existence_filter ema_confidence_gate bytetrack s2_logodds_temporal geometry_dynamic_no_multiagent rf_learned_gate \
  --output-dir outputs/results/q1_real_detector/yolov8n_smoke_main

/home/lebedeffson/Code/venv/bin/python scripts/run_q1_label_scarcity.py \
  --dataset-root data/visdrone/VisDrone2019-VID-val \
  --detections outputs/results/q1_real_detector/detections/yolov8n_visdrone_smoke.json \
  --label-budgets 0 0.01 0.05 \
  --seeds 11 22 \
  --split-mode sequence_chunks \
  --methods s_naive persistence_gate bayesian_existence_filter ema_confidence_gate bytetrack s2_logodds_temporal geometry_dynamic_no_multiagent rf_learned_gate \
  --output-dir outputs/results/q1_label_scarcity/yolov8n_smoke
```
