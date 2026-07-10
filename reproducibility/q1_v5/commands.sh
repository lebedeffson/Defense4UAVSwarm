#!/usr/bin/env bash
set -euo pipefail
/home/lebedeffson/Code/venv/bin/python scripts/q1_v5/freeze_legacy_baseline.py --output-dir outputs/results/q1_v5/legacy_freeze
/home/lebedeffson/Code/venv/bin/python scripts/q1_v5/run_operating_curves.py --config configs/q1_v5/visdrone_bytetrack.yaml --output-dir outputs/results/q1_v5/operating_curves/bytetrack --overwrite
/home/lebedeffson/Code/venv/bin/python scripts/q1_v5/run_matched_point_analysis.py --operating-points outputs/results/q1_v5/operating_curves/bytetrack/operating_points_raw.csv --trust-method legacy_geometry_dynamic_no_multiagent --trust-parameter 0 --output-dir outputs/results/q1_v5/matched_points/bytetrack
/home/lebedeffson/Code/venv/bin/python scripts/q1_v5/run_statistical_analysis.py --by-sequence outputs/results/q1_v5/operating_curves/bytetrack/operating_points_by_sequence.csv --baseline-method tracker_baseline --method legacy_geometry_dynamic_no_multiagent --primary-metric false_new_tracks_per_100_frames --output-dir outputs/results/q1_v5/statistics/bytetrack
/home/lebedeffson/Code/venv/bin/python scripts/q1_v5/run_trust_guard_ablation.py --config configs/q1_v5/trust_guard_v52.yaml --output-dir outputs/results/q1_v5/trust_guard_ablation/bytetrack
/home/lebedeffson/Code/venv/bin/python scripts/q1_v5/run_confidence_shift_eval.py --config configs/q1_v5/trust_guard_v52.yaml --output-dir outputs/results/q1_v5/confidence_shift/bytetrack
/home/lebedeffson/Code/venv/bin/python scripts/q1_v5/run_baseline_dominance_audit.py --operating-points outputs/results/q1_v5/operating_curves/bytetrack/operating_points_raw.csv --output-dir outputs/results/q1_v5/baseline_dominance/bytetrack
/home/lebedeffson/Code/venv/bin/python scripts/q1_v5/run_calibration_staleness.py --output-dir outputs/results/q1_v5/calibration_staleness
/home/lebedeffson/Code/venv/bin/python scripts/q1_v5/build_methodology_closure_report.py --output-dir outputs/results/q1_v5/methodology_closure
