"""Tests for Detectron2 device resolution."""

from __future__ import annotations

from unittest.mock import patch

import meta_face.config as config


def test_resolve_detectron2_device_honors_cpu() -> None:
    with patch.object(config, "DETECTRON2_DEVICE", "cpu"):
        assert config.resolve_detectron2_device() == "cpu"


def test_resolve_detectron2_device_falls_back_when_cuda_unavailable() -> None:
    with (
        patch.object(config, "DETECTRON2_DEVICE", "cuda:0"),
        patch("torch.cuda.is_available", return_value=False),
    ):
        assert config.resolve_detectron2_device() == "cpu"
