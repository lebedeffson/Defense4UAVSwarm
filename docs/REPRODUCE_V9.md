# REPRODUCE V9

Run after v8 data/detections exist:

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
```
