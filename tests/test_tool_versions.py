"""Skip markers must follow current tool versions, not mere key presence."""

from __future__ import annotations

from pathlib import Path

from meta_face.config import TOOL_VERSIONS
from meta_face.jobs import _tools_to_run
from meta_face.scanner import needs_processing
from meta_face.sidecar import SidecarDocument, has_tool, load_or_create, tool_is_current, update_sidecar


def test_stale_version_is_not_current(tmp_path: Path) -> None:
    image = tmp_path / "photo.jpg"
    image.write_bytes(b"x")

    def apply(doc: SidecarDocument) -> None:
        doc.set("face.opencv_fer.version", "1.0.0")
        doc.set("face.opencv_fer.faces", [])

    update_sidecar(image, apply)
    doc, _ = load_or_create(image)
    assert has_tool(doc, "opencv_fer")
    assert not tool_is_current(doc, "opencv_fer")
    assert TOOL_VERSIONS["opencv_fer"] != "1.0.0"
    assert needs_processing(image, ["opencv_fer"], force=False)
    assert _tools_to_run(doc, ["opencv_fer"], force=False) == ["opencv_fer"]
