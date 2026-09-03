"""Tests for mf annotate rendering helpers."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
from click.testing import CliRunner

from meta_face.annotate import (
    annotated_output_path,
    draw_annotations,
)
from meta_face.cli import main


def test_annotated_output_path_jpg() -> None:
    assert annotated_output_path(Path("/photos/images.jpg")) == Path("/photos/images_scrfd.jpg")


def test_annotated_output_path_heic() -> None:
    assert annotated_output_path(Path("/photos/images.heic")) == Path("/photos/images_scrfd.jpg")


def test_draw_annotations_smoke() -> None:
    image = np.zeros((200, 300, 3), dtype=np.uint8)
    records = [
        {
            "bbox": [50.0, 40.0, 150.0, 180.0],
            "det_score": 0.99,
            "kps": [
                [70.0, 80.0],
                [120.0, 82.0],
                [95.0, 110.0],
                [75.0, 140.0],
                [115.0, 142.0],
            ],
            "pose": [1.5, -2.0, 0.5],
            "gender": 1,
            "age": 34,
            "sex": "M",
        }
    ]
    out = draw_annotations(image, records, cluster_labels=[3])
    assert out.shape == image.shape
    assert not np.array_equal(out, image)


def test_annotate_cli_enqueues_by_default(tmp_path: Path) -> None:
    image = tmp_path / "photo.jpg"
    image.write_bytes(b"\xff\xd8\xff\xd9")
    runner = CliRunner()
    with (
        patch("meta_face.cli.require_insightface_runtime"),
        patch("meta_face.queue.enqueue_annotate", return_value="ann-1") as enqueue,
    ):
        result = runner.invoke(main, ["annotate", str(image)])
    assert result.exit_code == 0, result.output
    assert "enqueued 1 annotate jobs" in result.output
    enqueue.assert_called_once()
