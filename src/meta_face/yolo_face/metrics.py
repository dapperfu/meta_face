"""Per-stage timing and rolling FPS."""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field

DEFAULT_FPS_WINDOW = 30


class RollingFps:
    """Average frame rate over the last ``window`` frame intervals."""

    def __init__(self, window: int = DEFAULT_FPS_WINDOW) -> None:
        self._durations: deque[float] = deque(maxlen=window)

    def add(self, seconds: float) -> None:
        if seconds > 0:
            self._durations.append(seconds)

    @property
    def fps(self) -> float:
        total = sum(self._durations)
        return len(self._durations) / total if total > 0 else 0.0


@dataclass
class FrameTimer:
    """Collects millisecond timings for named stages of one frame."""

    stages: dict[str, float] = field(default_factory=dict)

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        start = time.perf_counter()
        try:
            yield
        finally:
            self.stages[name] = (time.perf_counter() - start) * 1000.0

    def get(self, name: str) -> float:
        return self.stages.get(name, 0.0)
