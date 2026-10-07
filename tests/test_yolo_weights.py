"""Tests for YOLOv8-Face weight download helpers and `mf download` wiring."""

from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner

from meta_face.yolo_face import weights


@pytest.fixture
def fake_fetch(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    calls: list[str] = []

    def fetch(url: str, dest: Path) -> None:
        calls.append(url)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"weights")

    monkeypatch.setattr(weights, "_fetch", fetch)
    return calls


def test_paths(tmp_path: Path) -> None:
    assert weights.yolo_model_path("s", tmp_path) == tmp_path / "yolov8s-face-lindevs.pt"
    assert weights.YOLO_FACE_WEIGHTS["n"].endswith("/yolov8n-face-lindevs.pt")
    with pytest.raises(ValueError):
        weights.yolo_weight_filename("q")


def test_env_model_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("META_FACE_YOLO_MODEL_DIR", str(tmp_path))
    assert weights.yolo_model_path() == tmp_path / "yolov8n-face-lindevs.pt"


def test_download_skips_when_present(tmp_path: Path, fake_fetch: list[str]) -> None:
    weights.download_yolo_weights("n", model_dir=tmp_path)
    weights.download_yolo_weights("n", model_dir=tmp_path)
    assert len(fake_fetch) == 1
    assert weights.is_yolo_available("n", tmp_path)


def test_download_force(tmp_path: Path, fake_fetch: list[str]) -> None:
    weights.download_yolo_weights("n", model_dir=tmp_path)
    weights.download_yolo_weights("n", model_dir=tmp_path, force=True)
    assert len(fake_fetch) == 2


def test_cli_backend_yolo(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, fake_fetch: list[str]
) -> None:
    import meta_face.config
    from meta_face.cli import main

    monkeypatch.setattr(meta_face.config, "YOLO_FACE_MODEL_DIR", tmp_path)
    result = CliRunner().invoke(main, ["download", "--backend", "yolo", "--yolo-variant", "all"])
    assert result.exit_code == 0, result.output
    assert len(fake_fetch) == 5


def test_cli_backend_all_includes_yolo(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, fake_fetch: list[str]
) -> None:
    import meta_face.analysis_models
    import meta_face.config
    import meta_face.models
    from meta_face.cli import main

    monkeypatch.setattr(meta_face.config, "YOLO_FACE_MODEL_DIR", tmp_path)
    monkeypatch.setattr(meta_face.models, "download_all", lambda **_: {})
    monkeypatch.setattr(meta_face.analysis_models, "download_all_analysis_models", lambda **_: {})
    result = CliRunner().invoke(main, ["download"])
    assert result.exit_code == 0, result.output
    assert "yolo/yolov8n-face-lindevs.pt" in result.output
    assert fake_fetch == [weights.YOLO_FACE_WEIGHTS["n"]]
