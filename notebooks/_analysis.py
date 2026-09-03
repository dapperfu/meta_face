"""Shared setup for meta-analysis notebooks (20_* series)."""

from __future__ import annotations

import sys
from pathlib import Path

NOTEBOOKS_DIR = Path(__file__).resolve().parent
REPO_ROOT = NOTEBOOKS_DIR.parent

for path in (REPO_ROOT, NOTEBOOKS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def default_root() -> Path:
    """Default photo collection root for 20XX year folders."""
    return Path("/tun/steph_pictures")
