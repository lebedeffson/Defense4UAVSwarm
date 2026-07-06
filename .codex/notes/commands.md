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
