from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable

import cv2
import pandas as pd

from defense4uavswarm.datasets.visdrone import VisDroneDataset


def build_visdrone_frame_manifest(dataset_root: str | Path, sequence_ids: Iterable[str] | None = None) -> pd.DataFrame:
    ds = VisDroneDataset(dataset_root, "val", subset_hint="VID")
    rows: list[dict[str, object]] = []
    allowed = list(sequence_ids) if sequence_ids is not None else None
    for rec in ds.frames(allowed):
        img = cv2.imread(str(rec.image_path))
        if img is None:
            raise RuntimeError(f"Cannot read image dimensions: {rec.image_path}")
        height, width = img.shape[:2]
        rows.append(
            {
                "sequence_id": rec.sequence_id,
                "frame_id": int(rec.frame_id),
                "image_path": str(rec.image_path),
                "image_width": int(width),
                "image_height": int(height),
            }
        )
    if not rows:
        raise RuntimeError(f"No VisDrone frames found under {dataset_root}")
    manifest = pd.DataFrame(rows).sort_values(["sequence_id", "frame_id"]).reset_index(drop=True)
    manifest["has_gt"] = False
    return manifest


def add_gt_presence(manifest: pd.DataFrame, gt: pd.DataFrame) -> pd.DataFrame:
    out = manifest.copy()
    if gt is None or gt.empty:
        out["has_gt"] = False
        return out
    keys = gt[["sequence_id", "frame_id"]].drop_duplicates()
    out = out.merge(keys.assign(has_gt=True), on=["sequence_id", "frame_id"], how="left", suffixes=("", "_gt"))
    out["has_gt"] = out["has_gt_gt"].fillna(False).astype(bool)
    return out.drop(columns=[c for c in ["has_gt_gt"] if c in out])


def manifest_sha256(manifest: pd.DataFrame) -> str:
    payload = manifest.sort_values(["sequence_id", "frame_id"]).to_csv(index=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def merge_manifest_dimensions(candidates: pd.DataFrame, manifest: pd.DataFrame) -> pd.DataFrame:
    required = {"sequence_id", "frame_id", "image_width", "image_height"}
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"Frame manifest missing columns: {sorted(missing)}")
    keep = manifest[["sequence_id", "frame_id", "image_width", "image_height"]].drop_duplicates()
    drop_cols = [c for c in ["image_width", "image_height"] if c in candidates]
    out = candidates.drop(columns=drop_cols).merge(keep, on=["sequence_id", "frame_id"], how="left")
    if out[["image_width", "image_height"]].isna().any().any():
        bad = out[out[["image_width", "image_height"]].isna().any(axis=1)][["sequence_id", "frame_id"]].drop_duplicates().head(10)
        raise ValueError(f"Candidate frames missing from manifest: {bad.to_dict(orient='records')}")
    return out
