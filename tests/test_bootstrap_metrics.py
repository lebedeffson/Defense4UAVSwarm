from pathlib import Path
import subprocess

import pandas as pd


def test_bootstrap_outputs_ci(tmp_path: Path):
    inp = tmp_path / "seq.csv"
    rows = []
    for seed in [1, 2]:
        for seq in ["a", "b"]:
            rows.append({"seed": seed, "sequence_id": seq, "scenario": "S_naive", "num_frames": 10, "TP": 10, "FP": 5, "FN": 1, "F1": 0.7, "IDF1": 0.7, "track_breaks": 1, "false_new_tracks": 3})
            rows.append({"seed": seed, "sequence_id": seq, "scenario": "S2_tnorm_soft", "num_frames": 10, "TP": 10, "FP": 4, "FN": 1, "F1": 0.72, "IDF1": 0.72, "track_breaks": 1, "false_new_tracks": 2})
    pd.DataFrame(rows).to_csv(inp, index=False)
    out = tmp_path / "ci.csv"
    subprocess.check_call(["/home/lebedeffson/Code/venv/bin/python", "scripts/bootstrap_metrics.py", "--input", str(inp), "--n-bootstrap", "20", "--output", str(out)])
    ci = pd.read_csv(out)
    assert len(ci) > 0
    assert (ci["ci95_low"] <= ci["mean"] + 1e-9).all()
    assert (ci["mean"] <= ci["ci95_high"] + 1e-9).all()
    assert "FP_delta" in set(ci["metric"])
