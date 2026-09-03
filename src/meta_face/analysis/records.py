"""Data records for sidecar meta-analysis."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

_YEAR_RE = re.compile(r"^20\d{2}$")


@dataclass
class ImageSummary:
    """Per-image row extracted from a .scar sidecar."""

    media_path: Path
    sidecar_path: Path | None
    parent_name: str
    year: int | None
    has_sidecar: bool
    tools_present: list[str] = field(default_factory=list)
    scrfd_face_count: int | None = None
    dlib_face_count: int | None = None
    scrfd_avg_det_score: float | None = None
    cluster_labels: list[int] | None = None
    cluster_dlib_labels: list[int] | None = None
    cluster_unique: int | None = None
    cluster_noise: int | None = None
    cluster_dlib_unique: int | None = None
    cluster_dlib_noise: int | None = None

    @staticmethod
    def year_from_parent(parent_name: str) -> int | None:
        if _YEAR_RE.match(parent_name):
            return int(parent_name)
        return None
