from pathlib import Path
import subprocess

import pandas as pd


def test_false_track_summary_counts(tmp_path: Path):
    inp = tmp_path / "events.csv"
    pd.DataFrame(
        [
            {"sequence_id": "s", "frame_id": 1, "scenario": "S_naive", "is_created_track": True, "is_suppressed": False, "is_tp_detection": False, "is_high_conf_fp": True},
            {"sequence_id": "s", "frame_id": 2, "scenario": "S_naive", "is_created_track": True, "is_suppressed": False, "is_tp_detection": True, "is_high_conf_fp": False},
            {"sequence_id": "s", "frame_id": 1, "scenario": "S2_tnorm_soft", "is_created_track": False, "is_suppressed": True, "is_tp_detection": False, "is_high_conf_fp": True},
        ]
    ).to_csv(inp, index=False)
    out = tmp_path / "summary.csv"
    subprocess.check_call(["/home/lebedeffson/Code/venv/bin/python", "scripts/build_false_track_summary.py", "--track-events", str(inp), "--output", str(out)])
    df = pd.read_csv(out)
    naive = df[df.scenario == "S_naive"].iloc[0]
    assert naive["false_new_tracks"] == 1
    assert naive["true_new_tracks"] == 1
