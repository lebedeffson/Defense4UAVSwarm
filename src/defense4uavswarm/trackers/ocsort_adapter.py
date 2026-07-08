from __future__ import annotations

from typing import Any

from .base import TrackerAdapter


class OCSortAdapter(TrackerAdapter):
    def __init__(self) -> None:
        try:
            from boxmot import OCSort  # type: ignore
        except Exception as exc:
            raise ImportError(f"OC-SORT unavailable: cannot import boxmot.OCSort ({exc})") from exc
        self._tracker = OCSort()

    @property
    def name(self) -> str:
        return "ocsort"

    def reset(self) -> None:
        self.__init__()

    def update(self, detections: Any, frame_id: int, image: Any | None = None) -> Any:
        return self._tracker.update(detections, image)
