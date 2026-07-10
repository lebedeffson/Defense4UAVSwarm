from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def bbox_center_area(box: tuple[float, float, float, float] | list[float] | np.ndarray) -> tuple[np.ndarray, float, float, float]:
    x1, y1, x2, y2 = [float(x) for x in box]
    w = max(1e-6, x2 - x1)
    h = max(1e-6, y2 - y1)
    center = np.asarray([(x1 + x2) * 0.5, (y1 + y2) * 0.5], dtype=float)
    return center, w * h, w, h


def kinematic_support(
    current_box: tuple[float, float, float, float] | list[float] | np.ndarray,
    previous_box: tuple[float, float, float, float] | list[float] | np.ndarray,
    previous_previous_box: tuple[float, float, float, float] | list[float] | np.ndarray,
    current_frame_id: int | None = None,
    previous_frame_id: int | None = None,
    previous_previous_frame_id: int | None = None,
    position_scale: float = 0.75,
    area_scale: float = 0.60,
) -> float:
    cur_center, cur_area, _, _ = bbox_center_area(current_box)
    prev_center, prev_area, prev_w, prev_h = bbox_center_area(previous_box)
    prev_prev_center, _, _, _ = bbox_center_area(previous_previous_box)
    if current_frame_id is not None and previous_frame_id is not None and previous_previous_frame_id is not None:
        dt_prev = max(1, int(previous_frame_id) - int(previous_previous_frame_id))
        dt_cur = max(1, int(current_frame_id) - int(previous_frame_id))
        pred_center = prev_center + (prev_center - prev_prev_center) / float(dt_prev) * float(dt_cur)
    else:
        pred_center = prev_center + (prev_center - prev_prev_center)
    denom = max(1e-6, float(np.sqrt(max(1e-6, prev_w * prev_h))))
    d_pos = float(np.linalg.norm(cur_center - pred_center) / denom)
    d_area = abs(float(np.log((cur_area + 1e-6) / (prev_area + 1e-6))))
    return float(np.exp(-0.5 * (d_pos / float(position_scale)) ** 2 - 0.5 * (d_area / float(area_scale)) ** 2))


@dataclass
class KinematicTrackState:
    position_scale: float = 0.75
    area_scale: float = 0.60
    observations: list[tuple[int, tuple[float, float, float, float]]] = field(default_factory=list)

    def score(self, box: tuple[float, float, float, float] | list[float] | np.ndarray, frame_id: int | None = None) -> tuple[float | None, bool]:
        if len(self.observations) < 2:
            return None, False
        prev_prev_frame, prev_prev_box = self.observations[-2]
        prev_frame, prev_box = self.observations[-1]
        cur_frame = int(frame_id) if frame_id is not None else prev_frame + 1
        if cur_frame <= prev_frame:
            return 1.0, False
        return (
            kinematic_support(
                box,
                prev_box,
                prev_prev_box,
                current_frame_id=cur_frame,
                previous_frame_id=prev_frame,
                previous_previous_frame_id=prev_prev_frame,
                position_scale=self.position_scale,
                area_scale=self.area_scale,
            ),
            True,
        )

    def update(self, box: tuple[float, float, float, float] | list[float] | np.ndarray, frame_id: int | None = None) -> None:
        x1, y1, x2, y2 = [float(x) for x in box]
        fid = int(frame_id) if frame_id is not None else (self.observations[-1][0] + 1 if self.observations else 0)
        self.observations.append((fid, (x1, y1, x2, y2)))
        if len(self.observations) > 3:
            self.observations = self.observations[-3:]
