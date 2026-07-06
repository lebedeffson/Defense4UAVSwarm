# Commands

Add only verified project-specific commands.

When checking U2UData integration, run:

```bash
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

When packaging the v7 Q1 closure bundle, run:

```bash
/home/lebedeffson/Code/venv/bin/python scripts/package_v7_q1_bundle.py
```
