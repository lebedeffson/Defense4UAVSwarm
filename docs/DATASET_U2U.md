# U2UData Integration

U2UData is the preferred v7 dataset path because it targets UAV-to-UAV cooperative perception for swarm UAV autonomous flight.

This repository does not vendor U2UData. Download it separately and place it under:

```text
data/U2UData
```

Then run:

```bash
python scripts/check_u2u_dataset.py \
  --dataset-root data/U2UData \
  --output outputs/results/v7_u2u/dataset_check/u2u_dataset_report.json

python scripts/build_u2u_manifest.py \
  --dataset-root data/U2UData \
  --split validation \
  --agents 3 \
  --min-frames 300 \
  --output-root data/manifests/u2u_v7
```

The adapter exports a common multi-agent schema with scenes, synchronized frames, agents, poses, modalities, and GT summary fields. If the local dataset layout is not recognized, the scripts produce a blocked report instead of fabricating metrics.

Sources used for integration notes:

* U2UData GitHub: https://github.com/fengtt42/U2UData
* U2UData paper: https://arxiv.org/abs/2408.00606
* U2UData+ project: https://github.com/fengtt42/U2UData-2
