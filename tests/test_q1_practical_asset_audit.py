from __future__ import annotations

from pathlib import Path

from scripts.audit_q1_practical_assets import AssetProbe, probe_assets, summarize_path


def test_probe_assets_marks_missing_dataset_blocked() -> None:
    rows = probe_assets("dataset", [AssetProbe("V2U4Real", ("definitely_missing_dataset",))])
    assert rows[0]["status"] == "blocked_missing_local_asset"
    assert rows[0]["claim_scope"] == "blocker_only"


def test_probe_assets_does_not_treat_adapter_as_result() -> None:
    rows = probe_assets("tracker", [AssetProbe("DeepSORT", (), adapter_paths=("pyproject.toml",))])
    assert rows[0]["status"] == "implemented_needs_unified_result_or_blocker"
    assert rows[0]["claim_scope"] == "not_claim_eligible_until_unified_v21_evidence"


def test_probe_assets_marks_unavailable_tracker_files_as_blocker(tmp_path: Path) -> None:
    blocker = tmp_path / "strongsort_unavailable.md"
    blocker.write_text("blocked", encoding="utf-8")
    rows = probe_assets(
        "tracker",
        [
            AssetProbe(
                "StrongSORT",
                (),
                adapter_paths=("pyproject.toml",),
                result_globs=(blocker.as_posix(),),
            )
        ],
    )
    assert rows[0]["status"] == "implemented_with_blocker_evidence"
    assert rows[0]["claim_scope"] == "blocker_only"


def test_summarize_path_counts_file() -> None:
    info = summarize_path(Path("pyproject.toml"))
    assert info["exists"] is True
    assert info["is_dir"] is False
    assert info["file_count"] == 1
