"""Bulk discovery and parallel sidecar collection."""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from meta_face.imaging import is_image_path
from meta_face.sidecar import media_path_for_sidecar

from .load import summarize_sidecar, summarize_sidecar_str
from .records import ImageSummary


def discover_images(root: Path, *, recursive: bool = True) -> list[Path]:
    """Find image files under root."""
    root = Path(root).resolve()
    if root.is_file() and is_image_path(root):
        return [root]
    if not root.is_dir():
        return []
    if recursive:
        return sorted(p for p in root.rglob("*") if p.is_file() and is_image_path(p))
    return sorted(p for p in root.iterdir() if p.is_file() and is_image_path(p))


def discover_sidecars(root: Path, *, recursive: bool = True) -> list[Path]:
    """Find .scar files under root and resolve to media paths where possible."""
    root = Path(root).resolve()
    scar_paths: list[Path]
    if root.is_file() and root.suffix.lower() == ".scar":
        scar_paths = [root]
    elif not root.is_dir():
        return []
    elif recursive:
        scar_paths = sorted(p for p in root.rglob("*.scar") if p.is_file())
    else:
        scar_paths = sorted(p for p in root.iterdir() if p.is_file() and p.suffix.lower() == ".scar")

    media_paths: list[Path] = []
    for scar in scar_paths:
        media = media_path_for_sidecar(scar)
        media_paths.append(media if media is not None else scar.with_suffix(""))
    return media_paths


def collect_summaries(
    paths: list[Path],
    *,
    workers: int | None = None,
    show_progress: bool = True,
) -> list[ImageSummary]:
    """Load ImageSummary rows in parallel."""
    if not paths:
        return []

    resolved = [str(Path(p).resolve()) for p in paths]
    worker_count = workers if workers is not None else (os.cpu_count() or 4)

    if worker_count <= 1 or len(resolved) == 1:
        return _collect_sequential(resolved, show_progress=show_progress)

    return _collect_parallel(resolved, worker_count, show_progress=show_progress)


def _collect_sequential(paths: list[str], *, show_progress: bool) -> list[ImageSummary]:
    iterator = paths
    if show_progress:
        try:
            from tqdm import tqdm

            iterator = tqdm(paths, desc="sidecars", unit="file")
        except ImportError:
            pass
    return [summarize_sidecar_str(p) for p in iterator]


def _collect_parallel(
    paths: list[str],
    workers: int,
    *,
    show_progress: bool,
) -> list[ImageSummary]:
    results: list[ImageSummary] = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(summarize_sidecar_str, p): p for p in paths}
        iterator = as_completed(futures)
        if show_progress:
            try:
                from tqdm import tqdm

                iterator = tqdm(iterator, total=len(futures), desc="sidecars", unit="file")
            except ImportError:
                pass
        for future in iterator:
            results.append(future.result())
    return results
