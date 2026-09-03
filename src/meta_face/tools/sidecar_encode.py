"""JSON-safe encoding of tool outputs for sidecar persistence."""

from __future__ import annotations

from typing import Any


def json_safe(value: Any) -> Any:
    """Recursively convert numpy/scalar values to CBOR-friendly Python types."""
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, (int, float)):
        return value
    if hasattr(value, "tolist"):
        return json_safe(value.tolist())
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    return str(value)


def passthrough_face_record(det: dict[str, Any]) -> dict[str, Any]:
    """Copy all keys from a detection dict into a sidecar face record."""
    return json_safe(det)
