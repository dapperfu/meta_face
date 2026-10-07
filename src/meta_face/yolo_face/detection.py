"""Backend-independent face detection result type."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Detection:
    """One detected face in pixel coordinates of the original frame (x2/y2 exclusive)."""

    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float
    class_id: int = 0

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1

    @property
    def center_x(self) -> float:
        return (self.x1 + self.x2) / 2

    @property
    def center_y(self) -> float:
        return (self.y1 + self.y2) / 2

    @property
    def area(self) -> int:
        return self.width * self.height

    @property
    def xyxy(self) -> list[int]:
        return [self.x1, self.y1, self.x2, self.y2]

    @property
    def xywh(self) -> list[int]:
        """Upper-left corner plus size."""
        return [self.x1, self.y1, self.width, self.height]

    def normalized(self, image_width: int, image_height: int) -> dict[str, float]:
        """Coordinates scaled to ``[0.0, 1.0]`` relative to the image size."""
        if image_width <= 0 or image_height <= 0:
            raise ValueError("Image dimensions must be positive")
        return {
            "x1": _unit(self.x1 / image_width),
            "y1": _unit(self.y1 / image_height),
            "x2": _unit(self.x2 / image_width),
            "y2": _unit(self.y2 / image_height),
        }

    def to_dict(
        self,
        image_width: int | None = None,
        image_height: int | None = None,
        *,
        normalized: bool = False,
    ) -> dict[str, Any]:
        """JSON-friendly representation; normalized coords need the image size."""
        out: dict[str, Any] = {
            "x1": self.x1,
            "y1": self.y1,
            "x2": self.x2,
            "y2": self.y2,
            "width": self.width,
            "height": self.height,
            "confidence": round(self.confidence, 4),
        }
        if normalized:
            if image_width is None or image_height is None:
                raise ValueError("normalized output requires image_width and image_height")
            out["normalized"] = {
                k: round(v, 4) for k, v in self.normalized(image_width, image_height).items()
            }
        return out


def _unit(value: float) -> float:
    return min(max(value, 0.0), 1.0)


def make_detection(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    confidence: float,
    image_width: int,
    image_height: int,
    class_id: int = 0,
) -> Detection | None:
    """
    Build a :class:`Detection` from float box coordinates, clipped to the image.

    Returns ``None`` for boxes that are non-finite or empty after clipping, so callers
    never see coordinates that violate the output contract.
    """
    values = (x1, y1, x2, y2, confidence)
    if not all(math.isfinite(v) for v in values):
        return None
    ix1 = min(max(math.floor(x1), 0), image_width)
    iy1 = min(max(math.floor(y1), 0), image_height)
    ix2 = min(max(math.ceil(x2), 0), image_width)
    iy2 = min(max(math.ceil(y2), 0), image_height)
    if ix2 <= ix1 or iy2 <= iy1:
        return None
    return Detection(ix1, iy1, ix2, iy2, _unit(float(confidence)), class_id)


def clip_detection(det: Detection, image_width: int, image_height: int) -> Detection | None:
    """Clip ``det`` to the image bounds, or ``None`` if nothing valid remains."""
    return make_detection(
        det.x1, det.y1, det.x2, det.y2, det.confidence, image_width, image_height, det.class_id
    )
