"""Stable RQ job ids."""

from __future__ import annotations

import hashlib
from pathlib import Path


def job_id_for_path(prefix: str, path: Path) -> str:
    digest = hashlib.sha256(str(path.resolve()).encode()).hexdigest()[:24]
    return f"{prefix}-{digest}"


def job_id_for_key(prefix: str, key: str) -> str:
    digest = hashlib.sha256(key.encode()).hexdigest()[:24]
    return f"{prefix}-{digest}"
