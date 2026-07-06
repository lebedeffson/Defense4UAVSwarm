# Reproduce U2UData v7 Experiment

Check dataset:

```bash
export HF_TOKEN=...

python scripts/download_u2u_minimal_subset.py \
  --repo-id fengtt42/U2UData-2 \
  --include-archives \
  --max-files 3 \
  --max-file-mb 5000 \
  --max-total-gb 12 \
  --min-free-disk-gb 25 \
  --retries 10 \
  --output-root data/U2UData

python scripts/extract_u2u_rar_subset.py \
  --archive-root data/U2UData \
  --output-root data/U2UData_extracted_minimal \
  --max-frames 300

python scripts/check_u2u_dataset.py \
  --dataset-root data/U2UData \
  --output outputs/results/v7_u2u/dataset_check/u2u_dataset_report.json
```

Build manifest:

```bash
python scripts/build_u2u_manifest.py \
  --dataset-root data/U2UData \
  --split validation \
  --agents 3 \
  --min-frames 300 \
  --output-root data/manifests/u2u_v7
```

Run v7 experiment:

```bash
python scripts/run_u2u_temporal_experiment.py \
  --manifest data/manifests/u2u_v7/manifest.json \
  --detections outputs/results/v7_u2u/detections/yolov8n_detections.json \
  --scenarios s_naive s2_tnorm_soft s2_tnorm_temporal s2_learned_fp_gate s2_ema_confidence_gate s2_bayesian_persistence_gate s2_support_count_gate \
  --agents 3 \
  --split holdout \
  --selected-params configs/selected_s2_temporal.yaml \
  --output-dir outputs/results/v7_u2u/main_holdout
```

Package:

```bash
python scripts/package_v7_q1_bundle.py
```

If U2UData or detections are missing, the outputs are intentionally marked as blocked.
