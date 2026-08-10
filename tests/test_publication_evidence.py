from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

from scripts.verify_publication_evidence import ARCHIVE_ROOT, KEY_MEMBERS, MANIFEST_MEMBER, verify_bundle


def test_publication_verifier_reports_missing_bundle(tmp_path: Path) -> None:
    report = verify_bundle(tmp_path / "missing.zip")
    assert report["status"] == "FAIL"
    assert report["checks"][0]["check"] == "bundle_exists"


def test_publication_verifier_accepts_valid_fixture(tmp_path: Path) -> None:
    bundle = tmp_path / "evidence.zip"
    payloads = {member: member.encode("utf-8") for member in KEY_MEMBERS}
    manifest = {
        "status": "success",
        "checks": [{"check": f"C{i:02d}", "pass": True} for i in range(1, 51)],
        "files": [
            {
                "path": str(Path(member).relative_to(str(ARCHIVE_ROOT))),
                "sha256": hashlib.sha256(data).hexdigest(),
                "bytes": len(data),
            }
            for member, data in sorted(payloads.items())
        ],
    }
    with zipfile.ZipFile(bundle, "w") as archive:
        for member, data in payloads.items():
            archive.writestr(member, data)
        archive.writestr(MANIFEST_MEMBER, json.dumps(manifest))

    report = verify_bundle(bundle, hashlib.sha256(bundle.read_bytes()).hexdigest())
    assert report["status"] == "PASS"
    assert report["manifest_check_count"] == 50
