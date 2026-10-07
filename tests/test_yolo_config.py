"""Configuration precedence tests: defaults < TOML < CLI."""

from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner

from meta_face.yolo_face import config as cfg
from meta_face.yolo_face.config import AppConfig, load_config


@pytest.fixture(autouse=True)
def _no_repo_config(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(cfg, "DEFAULT_CONFIG_PATH", tmp_path / "absent.toml")


def _write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "c.toml"
    path.write_text(text)
    return path


def test_defaults() -> None:
    config = load_config()
    assert config == AppConfig()
    assert config.model.confidence == 0.40
    assert config.model.iou == 0.45
    assert config.model.image_size == 640
    assert config.model.device == "auto"
    assert config.performance.frame_skip == 0
    assert config.filters.min_face_width == 0


def test_file_overrides(tmp_path: Path) -> None:
    path = _write(
        tmp_path, '[model]\nconfidence = 0.6\npath = "m/x.pt"\n[performance]\nframe_skip = 2\n'
    )
    config = load_config(path)
    assert config.model.confidence == 0.6
    assert config.model.path == Path("m/x.pt")
    assert config.model.iou == 0.45
    assert config.performance.frame_skip == 2


def test_cli_overrides_file(tmp_path: Path) -> None:
    path = _write(tmp_path, "[model]\nconfidence = 0.6\nimage_size = 960\n")
    config = load_config(path, {"model": {"confidence": 0.3, "image_size": None}})
    assert config.model.confidence == 0.3
    assert config.model.image_size == 960


def test_unknown_key(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="model.bogus"):
        load_config(_write(tmp_path, "[model]\nbogus = 1\n"))


def test_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "nope.toml")


def test_repo_default_config_parses() -> None:
    repo_config = Path(__file__).resolve().parents[1] / "config" / "yolo_face.toml"
    config = load_config(repo_config)
    assert config.model.path.name == "yolov8n-face-lindevs.pt"


def test_cli_precedence_reaches_detector(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from meta_face.yolo_face import cli

    seen: list[AppConfig] = []

    def fake_make_detector(config: AppConfig) -> None:
        seen.append(config)
        raise SystemExit(0)

    monkeypatch.setattr(cli, "_make_detector", fake_make_detector)
    image = tmp_path / "x.png"
    import cv2
    import numpy as np

    cv2.imwrite(str(image), np.zeros((8, 8, 3), np.uint8))
    path = _write(tmp_path, "[model]\nconfidence = 0.6\niou = 0.3\n")
    result = CliRunner().invoke(
        cli.yolo, ["image", str(image), "--config", str(path), "--confidence", "0.2"]
    )
    assert result.exit_code == 0, result.output
    assert seen[0].model.confidence == 0.2
    assert seen[0].model.iou == 0.3
