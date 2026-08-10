from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment

from defense4uavswarm.v8_sim import vector_iou

from .base import TrackerAdapter


def _bbox_to_z(box: np.ndarray) -> np.ndarray:
    w = max(1e-6, float(box[2] - box[0]))
    h = max(1e-6, float(box[3] - box[1]))
    return np.asarray([float(box[0] + w / 2.0), float(box[1] + h / 2.0), w * h, w / h], dtype=float)


def _x_to_bbox(x: np.ndarray) -> np.ndarray:
    area = max(1e-6, float(x[2]))
    ratio = max(1e-6, float(x[3]))
    w = np.sqrt(area * ratio)
    h = area / max(w, 1e-6)
    return np.asarray([float(x[0] - w / 2.0), float(x[1] - h / 2.0), float(x[0] + w / 2.0), float(x[1] + h / 2.0)], dtype=float)


@dataclass
class KalmanBoxTrack:
    bbox: np.ndarray
    track_id: int
    confidence: float
    class_id: float

    def __post_init__(self) -> None:
        self.x = np.zeros((7, 1), dtype=float)
        self.x[:4, 0] = _bbox_to_z(self.bbox)
        self.P = np.eye(7, dtype=float)
        self.P[4:, 4:] *= 1000.0
        self.P *= 10.0
        self.F = np.eye(7, dtype=float)
        self.F[0, 4] = 1.0
        self.F[1, 5] = 1.0
        self.F[2, 6] = 1.0
        self.H = np.zeros((4, 7), dtype=float)
        self.H[:4, :4] = np.eye(4, dtype=float)
        self.R = np.eye(4, dtype=float)
        self.R[2:, 2:] *= 10.0
        self.Q = np.eye(7, dtype=float)
        self.Q[4:, 4:] *= 0.01
        self.time_since_update = 0
        self.hits = 1
        self.hit_streak = 1
        self.age = 0

    def predict(self) -> np.ndarray:
        if (self.x[2, 0] + self.x[6, 0]) <= 0:
            self.x[6, 0] = 0.0
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        return self.bbox_state()

    def update(self, detection: np.ndarray) -> None:
        z = _bbox_to_z(detection[:4]).reshape(4, 1)
        y = z - self.H @ self.x
        s = self.H @ self.P @ self.H.T + self.R
        k = self.P @ self.H.T @ np.linalg.inv(s)
        self.x = self.x + k @ y
        self.P = (np.eye(7) - k @ self.H) @ self.P
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1
        self.confidence = float(detection[4])
        self.class_id = float(detection[5])

    def bbox_state(self) -> np.ndarray:
        return _x_to_bbox(self.x[:, 0])

    def output_row(self) -> np.ndarray:
        return np.asarray([*self.bbox_state(), float(self.track_id), float(self.confidence), float(self.class_id)], dtype=float)


class SortAdapter(TrackerAdapter):
    """Simple SORT tracker: Kalman prediction plus Hungarian IoU assignment."""

    def __init__(self, max_age: int = 30, min_hits: int = 3, iou_threshold: float = 0.3) -> None:
        self.max_age = int(max_age)
        self.min_hits = int(min_hits)
        self.iou_threshold = float(iou_threshold)
        self.reset()

    @property
    def name(self) -> str:
        return "sort"

    def reset(self) -> None:
        self.trackers: list[KalmanBoxTrack] = []
        self.frame_count = 0
        self._next_id = 1

    def update(self, detections: Any, frame_id: int, image: Any | None = None) -> np.ndarray:
        del image
        self.frame_count += 1
        dets = np.asarray(detections, dtype=float)
        if dets.size == 0:
            dets = np.empty((0, 6), dtype=float)
        dets = dets.reshape((-1, dets.shape[-1]))
        if dets.shape[1] < 6:
            raise ValueError("SORT detections must have columns x1,y1,x2,y2,confidence,class_id")

        predictions = np.asarray([trk.predict() for trk in self.trackers], dtype=float) if self.trackers else np.empty((0, 4), dtype=float)
        matched, unmatched_dets, unmatched_trks = _associate(dets[:, :4], predictions, self.iou_threshold)
        for det_idx, trk_idx in matched:
            self.trackers[int(trk_idx)].update(dets[int(det_idx), :6])
        for det_idx in unmatched_dets:
            self.trackers.append(KalmanBoxTrack(dets[int(det_idx), :4], self._next_id, float(dets[int(det_idx), 4]), float(dets[int(det_idx), 5])))
            self._next_id += 1

        outputs = []
        alive = []
        for trk in self.trackers:
            if trk.time_since_update < 1 and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                outputs.append(trk.output_row())
            if trk.time_since_update <= self.max_age:
                alive.append(trk)
        self.trackers = alive
        return np.asarray(outputs, dtype=float).reshape((-1, 7)) if outputs else np.empty((0, 7), dtype=float)


def _associate(detections: np.ndarray, trackers: np.ndarray, iou_threshold: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(trackers) == 0:
        return np.empty((0, 2), dtype=int), np.arange(len(detections), dtype=int), np.empty((0,), dtype=int)
    if len(detections) == 0:
        return np.empty((0, 2), dtype=int), np.empty((0,), dtype=int), np.arange(len(trackers), dtype=int)
    iou = np.asarray([vector_iou(det, trackers) for det in detections], dtype=float)
    rows, cols = linear_sum_assignment(-iou)
    matches = []
    unmatched_dets = set(range(len(detections)))
    unmatched_trks = set(range(len(trackers)))
    for det_idx, trk_idx in zip(rows, cols):
        if iou[det_idx, trk_idx] < float(iou_threshold):
            continue
        matches.append([int(det_idx), int(trk_idx)])
        unmatched_dets.discard(int(det_idx))
        unmatched_trks.discard(int(trk_idx))
    return (
        np.asarray(matches, dtype=int).reshape((-1, 2)),
        np.asarray(sorted(unmatched_dets), dtype=int),
        np.asarray(sorted(unmatched_trks), dtype=int),
    )
