"""Sidecar edits hold the lock file only inside the sidecar-rs edit context."""

from __future__ import annotations

import threading
from pathlib import Path

import pytest
from click.testing import CliRunner
from sidecar_rs import LockTimeout, SidecarDocument

from meta_face.sidecar import update_sidecar, update_sidecar_path


def _set_value(value: int):
    return lambda doc: doc.set("face.test.value", value)


def test_lock_file_exists_only_inside_the_edit_context(tmp_path: Path) -> None:
    image = tmp_path / "photo.jpg"
    image.write_bytes(b"x")
    scar = image.with_suffix(".scar")
    seen: list[bool] = []

    def patch(doc: SidecarDocument) -> None:
        seen.append((scar.name + ".lock") in {p.name for p in tmp_path.iterdir()})
        doc.set("face.test.value", 1)

    update_sidecar(image, patch)

    assert seen == [True]
    assert SidecarDocument.from_path(scar)["face.test.value"] == 1
    assert sorted(p.name for p in tmp_path.iterdir()) == ["photo.jpg", "photo.scar"]


def test_failed_patch_releases_lock_without_writing(tmp_path: Path) -> None:
    image = tmp_path / "photo.jpg"
    image.write_bytes(b"x")
    scar = image.with_suffix(".scar")
    update_sidecar(image, _set_value(1))

    def broken(doc: SidecarDocument) -> None:
        doc.set("face.test.value", 2)
        raise ValueError("boom")

    with pytest.raises(ValueError, match="boom"):
        update_sidecar(image, broken)

    assert SidecarDocument.from_path(scar)["face.test.value"] == 1
    assert sorted(p.name for p in tmp_path.iterdir()) == ["photo.jpg", "photo.scar"]


def test_lockfile_timeout_is_a_failure(tmp_path: Path) -> None:
    scar = tmp_path / "photo.scar"
    holding = threading.Event()
    release = threading.Event()

    def holder() -> None:
        with SidecarDocument.edit(scar) as doc:
            doc.set("face.test.value", 1)
            holding.set()
            assert release.wait(timeout=2.0)

    thread = threading.Thread(target=holder)
    thread.start()
    assert holding.wait(timeout=2.0)
    try:
        with pytest.raises(LockTimeout, match="lockfile timeout"):
            update_sidecar_path(scar, _set_value(2), lock_timeout_s=0.05)
    finally:
        release.set()
        thread.join(timeout=3.0)

    assert SidecarDocument.from_path(scar)["face.test.value"] == 1
    assert not (tmp_path / "photo.scar.lock").exists()


def test_clean_locks_command_reports_removed_and_in_use(tmp_path: Path) -> None:
    from meta_face.cli import main

    nested = tmp_path / "album"
    nested.mkdir()
    (tmp_path / "a.scar.lock").touch()
    (nested / "b.scar.lock").touch()
    holding = threading.Event()
    release = threading.Event()

    def holder() -> None:
        with SidecarDocument.edit(nested / "c.scar"):
            holding.set()
            assert release.wait(timeout=2.0)

    thread = threading.Thread(target=holder)
    thread.start()
    assert holding.wait(timeout=2.0)
    try:
        result = CliRunner().invoke(main, ["clean-locks", str(tmp_path)])
        assert result.exit_code == 0, result.output
        assert "Removed 2 stale lock file(s); 1 in use." in result.output
        assert sorted(p.name for p in tmp_path.rglob("*.lock")) == ["c.scar.lock"]
    finally:
        release.set()
        thread.join(timeout=3.0)
