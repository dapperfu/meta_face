"""Directory listing for scan jobs."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from meta_face.scanner import scan_directory_level


def test_scan_directory_level_enqueues_images_without_reading_sidecars(tmp_path: Path) -> None:
    photos = tmp_path / "album"
    nested = photos / "nested"
    nested.mkdir(parents=True)
    (photos / "keep.jpg").write_bytes(b"\xff\xd8\xff\xd9")
    (photos / "notes.txt").write_text("skip")
    (photos / "keep.scar").write_text("already processed")
    (nested / "child.jpg").write_bytes(b"\xff\xd8\xff\xd9")

    with patch("meta_face.scanner.needs_processing") as skip_check:
        stats, images, subdirs = scan_directory_level(photos, ["scrfd"], force=False)

    skip_check.assert_not_called()
    assert stats.discovered == 1
    assert stats.enqueued == 1
    assert stats.skipped == 0
    assert images == [photos / "keep.jpg"]
    assert subdirs == [nested]


def test_scan_directory_level_lists_a_single_image(tmp_path: Path) -> None:
    image = tmp_path / "solo.png"
    image.write_bytes(b"png")
    stats, images, subdirs = scan_directory_level(image, ["scrfd"])
    assert stats.enqueued == 1
    assert images == [image]
    assert subdirs == []
