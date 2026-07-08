from __future__ import annotations

from typing import Any

from .base import TrackerAdapter


class StrongSORTAdapter(TrackerAdapter):
    def __init__(self) -> None:
        try:
            from boxmot import StrongSort  # type: ignore
        except Exception as exc:
            raise ImportError(f"StrongSORT unavailable: cannot import boxmot.StrongSort ({exc})") from exc
        self._tracker = StrongSort()

    @property
    def name(self) -> str:
        return "strongsort"

    def reset(self) -> None:
        self.__init__()

    def update(self, detections: Any, frame_id: int, image: Any | None = None) -> Any:
        return self._tracker.update(detections, image)
