from __future__ import annotations

from typing import Any

from .base import TrackerAdapter


class ByteTrackLikeAdapter(TrackerAdapter):
    @property
    def name(self) -> str:
        return "bytetrack"

    def reset(self) -> None:
        return None

    def update(self, detections: Any, frame_id: int, image: Any | None = None) -> Any:
        return detections
