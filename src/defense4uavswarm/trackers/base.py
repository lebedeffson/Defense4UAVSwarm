from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class TrackerAdapter(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def reset(self) -> None:
        ...

    @abstractmethod
    def update(self, detections: Any, frame_id: int, image: Any | None = None) -> Any:
        ...
