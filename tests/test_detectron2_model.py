"""Tests for detectron2 model zoo resolution."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from meta_face.detectron2_model import (
    Detectron2ModelPaths,
    is_detectron2_available,
    uses_custom_detectron2_paths,
)


def test_is_detectron2_available_with_package_only() -> None:
    with (
        patch("meta_face.detectron2_model.detectron2_package_installed", return_value=True),
        patch("meta_face.detectron2_model.uses_custom_detectron2_paths", return_value=False),
    ):
        assert is_detectron2_available() is True


def test_resolve_detectron2_model_uses_model_zoo() -> None:
    pytest.importorskip("detectron2")
    from meta_face.detectron2_model import resolve_detectron2_model

    with (
        patch("meta_face.detectron2_model.uses_custom_detectron2_paths", return_value=False),
        patch(
            "detectron2.model_zoo.get_config_file",
            return_value="/cfg/retinanet.yaml",
        ),
        patch(
            "detectron2.model_zoo.get_checkpoint_url",
            return_value="https://example.com/model.pkl",
        ),
        patch(
            "meta_face.detectron2_model.cached_model_zoo_weights_path",
            return_value=__import__("pathlib").Path("/cache/model.pkl"),
        ),
    ):
        paths = resolve_detectron2_model()
    assert paths == Detectron2ModelPaths(
        config="/cfg/retinanet.yaml",
        weights="https://example.com/model.pkl",
        model_zoo="COCO-Detection/retinanet_R_50_FPN_3x.yaml",
    )
