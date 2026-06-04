"""Sidecar document helpers wrapping sidecar_rs."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sidecar_rs import SidecarDocument

from meta_face.config import (
    FACE_KEY_PREFIX,
    TOOL_VERSIONS,
    tool_data_key,
    tool_processed_at_key,
    tool_version_key,
)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def sidecar_path_for_media(media_path: Path) -> Path:
    """Resolve photo.jpg -> photo.scar in the same directory."""
    if media_path.suffix.lower() == ".scar":
        return media_path
    return media_path.with_suffix(".scar")


def media_path_for_sidecar(sidecar_path: Path) -> Path | None:
    """Best-effort media path from _media.basename or by swapping extension."""
    if not sidecar_path.exists():
        return None
    doc = SidecarDocument.from_path(str(sidecar_path))
    basename = doc["_media.basename"] if "_media.basename" in doc else None
    if isinstance(basename, str) and basename:
        candidate = sidecar_path.parent / basename
        if candidate.exists():
            return candidate
    stem = sidecar_path.with_suffix("")
    for ext in (".jpg", ".jpeg", ".png", ".webp", ".heic", ".heif", ".tif", ".tiff", ".bmp"):
        candidate = stem.with_suffix(ext)
        if candidate.exists():
            return candidate
    return None


def load_or_create(media_path: Path) -> tuple[SidecarDocument, Path]:
    scar_path = sidecar_path_for_media(media_path)
    if scar_path.exists():
        doc = SidecarDocument.from_path(str(scar_path))
    else:
        doc = SidecarDocument()
        doc.set_media_basename(media_path.name)
    return doc, scar_path


def save(doc: SidecarDocument, scar_path: Path) -> None:
    scar_path.parent.mkdir(parents=True, exist_ok=True)
    doc.to_path(str(scar_path))


def has_tool(doc: SidecarDocument, tool: str) -> bool:
    """Return True when face.<tool>.version is present."""
    key = tool_version_key(tool)
    return key in doc


def list_face_tools(doc: SidecarDocument) -> list[str]:
    tools: set[str] = set()
    prefix = FACE_KEY_PREFIX
    for key in doc.entries():
        if not key.startswith(prefix):
            continue
        remainder = key[len(prefix) :]
        if "." not in remainder:
            continue
        tool, _field = remainder.split(".", 1)
        tools.add(tool)
    return sorted(tools)


def get_face_section(doc: SidecarDocument, tool: str) -> dict[str, Any]:
    """Return all face.<tool>.* keys as a nested dict."""
    prefix = f"{FACE_KEY_PREFIX}{tool}."
    section: dict[str, Any] = {}
    for key, value in doc.entries().items():
        if key.startswith(prefix):
            section[key[len(prefix) :]] = value
    return section


def write_tool_result(
    doc: SidecarDocument,
    tool: str,
    data: dict[str, Any],
    *,
    version: str | None = None,
) -> None:
    """Write face.<tool>.version, processed_at, and data fields."""
    doc.set(tool_version_key(tool), version or TOOL_VERSIONS.get(tool, "1.0.0"))
    doc.set(tool_processed_at_key(tool), utc_now_iso())
    for field, value in data.items():
        doc.set(tool_data_key(tool, field), value)


def read_tool_data(doc: SidecarDocument, tool: str, field: str) -> Any:
    key = tool_data_key(tool, field)
    return doc[key]
