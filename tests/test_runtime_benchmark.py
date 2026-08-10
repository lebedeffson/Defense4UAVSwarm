from pathlib import Path


def test_runtime_script_exists() -> None:
    assert Path("scripts/benchmark_q1_runtime_final.py").exists()
