from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import pandas as pd
import yaml


VISDRONE_CLASSES = {
    0: "ignored",
    1: "pedestrian",
    2: "people",
    3: "bicycle",
    4: "car",
    5: "van",
    6: "truck",
    7: "tricycle",
    8: "awning-tricycle",
    9: "bus",
    10: "motor",
    11: "others",
}


@dataclass(frozen=True)
class FrameRecord:
    sequence_id: str
    frame_id: int
    image_path: Path


class VisDroneDataset:
    def __init__(self, root: str | Path, split: str = "val", subset_hint: str | None = None) -> None:
        self.root = Path(root)
        self.split = split
        self.subset_hint = subset_hint
        self.split_root = self._find_split_root(split, subset_hint)
        self.sequences_dir = self.split_root / "sequences"
        self.annotations_dir = self.split_root / "annotations"
        if not self.sequences_dir.exists():
            raise FileNotFoundError(f"No sequences dir: {self.sequences_dir}")
        if not self.annotations_dir.exists():
            raise FileNotFoundError(f"No annotations dir: {self.annotations_dir}")

    def _find_split_root(self, split: str, subset_hint: str | None = None) -> Path:
        suffixes = {
            "train": ("train", "trainset"),
            "val": ("val", "valset"),
            "test": ("test-dev", "testset-dev"),
        }[split]
        if not self.root.exists():
            raise FileNotFoundError(f"VisDrone root does not exist: {self.root}")
        candidates = [
            p for p in self.root.iterdir()
            if p.is_dir()
            and any(s in p.name.lower() for s in suffixes)
            and (subset_hint is None or subset_hint.lower() in p.name.lower())
            and (p / "sequences").exists()
            and (p / "annotations").exists()
        ]
        if not candidates and (self.root / "sequences").exists():
            return self.root
        if not candidates:
            hint = f" subset={subset_hint}" if subset_hint else ""
            raise FileNotFoundError(f"Cannot find VisDrone {split}{hint} under {self.root}")
        return sorted(candidates)[0]

    def sequence_ids(self) -> list[str]:
        return sorted(p.name for p in self.sequences_dir.iterdir() if p.is_dir())

    def frames(self, sequence_ids: list[str] | None = None) -> Iterator[FrameRecord]:
        allowed = set(sequence_ids or self.sequence_ids())
        for seq in self.sequence_ids():
            if seq not in allowed:
                continue
            for image_path in sorted((self.sequences_dir / seq).glob("*.jpg")):
                yield FrameRecord(seq, int(image_path.stem), image_path)

    def annotations(self, sequence_id: str) -> pd.DataFrame:
        path = self.annotations_dir / f"{sequence_id}.txt"
        if not path.exists():
            raise FileNotFoundError(path)
        rows = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                parts = [x.strip() for x in line.strip().split(",")]
                if len(parts) < 8:
                    continue
                vals = [float(x) for x in parts[:10]]
                if len(vals) >= 10:
                    frame_id, track_id, x, y, w, h, score, cls, trunc, occ = vals[:10]
                else:
                    frame_id, x, y, w, h, score, cls, trunc = vals[:8]
                    track_id, occ = -1, 0
                if int(cls) <= 0 or int(cls) >= 11 or score <= 0:
                    continue
                rows.append(
                    {
                        "sequence_id": sequence_id,
                        "frame_id": int(frame_id),
                        "gt_track_id": int(track_id),
                        "class_id": int(cls),
                        "class_name": VISDRONE_CLASSES.get(int(cls), "unknown"),
                        "x1": x,
                        "y1": y,
                        "x2": x + w,
                        "y2": y + h,
                        "score": score,
                        "truncation": trunc,
                        "occlusion": occ,
                    }
                )
        return pd.DataFrame(rows)

    def all_annotations(self, sequence_ids: list[str] | None = None) -> pd.DataFrame:
        frames = [self.annotations(seq) for seq in (sequence_ids or self.sequence_ids())]
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def verify(self, max_frames: int = 30, out_dir: str | Path = "outputs/figures/gt_check") -> dict:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        count = 0
        for rec in self.frames():
            img = cv2.imread(str(rec.image_path))
            if img is None:
                raise RuntimeError(f"Cannot read image: {rec.image_path}")
            ann = self.annotations(rec.sequence_id)
            ann = ann[ann.frame_id == rec.frame_id].head(50)
            for r in ann.itertuples():
                cv2.rectangle(img, (int(r.x1), int(r.y1)), (int(r.x2), int(r.y2)), (0, 255, 0), 2)
                cv2.putText(img, f"{r.gt_track_id}:{r.class_id}", (int(r.x1), int(r.y1) - 3), 0, 0.45, (0, 255, 0), 1)
            cv2.imwrite(str(out / f"{rec.sequence_id}_{rec.frame_id:07d}.jpg"), img)
            count += 1
            if count >= max_frames:
                break
        return {"checked_frames": count, "visualizations": str(out)}


def write_custom_split(root: str | Path, split_file: str | Path, val_ratio: float) -> dict:
    ds = VisDroneDataset(root, "train")
    seqs = ds.sequence_ids()
    n_val = max(1, int(round(len(seqs) * val_ratio)))
    val = seqs[-n_val:]
    train = seqs[:-n_val]
    payload = {"dataset": str(root), "mode": "custom_train_scene_split", "train_sequences": train, "val_sequences": val}
    with open(split_file, "w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, allow_unicode=True, sort_keys=False)
    return payload


def discover_visdrone_subsets(root: str | Path) -> list[dict]:
    root = Path(root)
    if not root.exists():
        return []
    rows = []
    for p in sorted(root.iterdir()):
        if not p.is_dir():
            continue
        seq_dir = p / "sequences"
        ann_dir = p / "annotations"
        if not seq_dir.exists() or not ann_dir.exists():
            continue
        name = p.name
        subset = "VID" if "VID" in name.upper() else "MOT" if "MOT" in name.upper() else "UNKNOWN"
        split = "val" if "val" in name.lower() else "train" if "train" in name.lower() else "test-dev" if "test" in name.lower() else "unknown"
        rows.append({"path": p, "dataset_subset": f"{subset}-{split}", "subset": subset, "split": split})
    return rows
