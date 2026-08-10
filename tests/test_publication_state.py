from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from scripts.verify_publication_state import REQUIRED_FILES, verify_repository


def copy_publication_state(source: Path, destination: Path) -> None:
    for relative in REQUIRED_FILES:
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / relative, target)
    verifier = destination / "scripts/verify_publication_state.py"
    if not verifier.exists():
        verifier.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / "scripts/verify_publication_state.py", verifier)


def test_publication_state_passes_for_repository() -> None:
    report = verify_repository(run_tests=False)
    assert report["status"] == "PASS", report


def test_publication_state_rejects_local_public_path(tmp_path: Path) -> None:
    source = Path(__file__).resolve().parents[1]
    copy_publication_state(source, tmp_path)
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    readme = tmp_path / "README.md"
    readme.write_text(readme.read_text(encoding="utf-8") + "\n/home/example/private\n", encoding="utf-8")

    report = verify_repository(tmp_path, run_tests=False)
    hygiene = next(item for item in report["checks"] if item["check"] == "public_surface_hygiene")
    assert hygiene["pass"] is False
