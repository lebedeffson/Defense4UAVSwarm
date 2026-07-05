# Reproducing Defense4UAVSwarm

## Install

```bash
python -m venv /home/lebedeffson/Code/venv
/home/lebedeffson/Code/venv/bin/pip install -r requirements.txt
/home/lebedeffson/Code/venv/bin/pip install -e .
```

## Data

Place VisDrone2019-VID-val under:

```text
data/visdrone/
```

Raw images are not included in this repository.

## Build manifest-only pseudo-swarm holdout

```bash
/home/lebedeffson/Code/venv/bin/python scripts/build_pseudo_swarm.py \
  --config configs/pseudo_swarm_stress.yaml \
  --split-config configs/vid_split.yaml \
  --split holdout \
  --materialization-mode manifest_only \
  --output data/swarm/pseudo_visdrone_stress_manifest/holdout
```

## Check pseudo-swarm

```bash
/home/lebedeffson/Code/venv/bin/python scripts/check_swarm_dataset.py \
  --root data/swarm/pseudo_visdrone_stress_manifest/holdout
```

## Single-seed holdout

```bash
/home/lebedeffson/Code/venv/bin/python scripts/run_experiment_matrix.py \
  --config configs/default.yaml \
  --swarm-config configs/pseudo_swarm_stress.yaml \
  --pseudo-attack-config configs/pseudo_attack_v2.yaml \
  --pseudo-attack-seed 11 \
  --split-config configs/vid_split.yaml \
  --split holdout \
  --tasks swarm_vid \
  --models yolov8n \
  --eps 0.008 \
  --scenarios s_naive s2_tnorm_soft \
  --use-selected-params configs/selected_s2_tnorm_soft_v30.yaml \
  --swarm-root data/swarm/pseudo_visdrone_stress_manifest/holdout \
  --swarm-generation-mode on_the_fly \
  --sequence-batch-size 1 \
  --class-groups all \
  --output-dir outputs/results/reproduce_holdout_seed11
```

## Multi-seed table

```bash
/home/lebedeffson/Code/venv/bin/python scripts/run_v4_2_multiseed.py \
  --seeds 11 22 33 44 55 \
  --selected-params configs/selected_s2_tnorm_soft_v30.yaml \
  --output-root outputs/results/v4_2_review_response/seeds \
  --summary-output outputs/results/v4_2_review_response/seed_summary.csv \
  --mean-std-output outputs/results/v4_2_review_response/mean_std_summary.csv
```

## Learned FP gate baseline

```bash
/home/lebedeffson/Code/venv/bin/python scripts/train_fp_classifier_baseline.py \
  --train-audit outputs/results/swarm_v3_newgate_calibration_selected/swarm_feature_audit.csv \
  --test-audit outputs/results/swarm_v3_newgate_holdout/swarm_feature_audit.csv \
  --models logistic random_forest gradient_boosting \
  --threshold-grid 0.50 0.55 0.60 0.65 0.70 0.75 0.80 0.85 0.90 \
  --selection-objective constrained_fp_reduction \
  --output-dir outputs/results/v5_1_fp_classifier/train
```

## Full runtime

```bash
/home/lebedeffson/Code/venv/bin/python scripts/run_full_runtime_benchmark.py \
  --config configs/default.yaml \
  --swarm-config configs/pseudo_swarm_stress.yaml \
  --pseudo-attack-config configs/pseudo_attack_v2.yaml \
  --split-config configs/vid_split.yaml \
  --split holdout \
  --models yolov8n \
  --eps 0.008 \
  --scenarios s_naive s2_tnorm_soft \
  --runtime-mode full_yolo \
  --warmup-frames 10 \
  --sample-frames 50 \
  --device cuda \
  --output-dir outputs/results/v5_1_full_runtime
```
