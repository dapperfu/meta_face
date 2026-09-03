"""Image positions stored as normalized fractions, clamped to [0, 1].

Inference and rendering use pixels. Confidence, angles and native depth retain
their units; explicitly projected depth is scaled by width but remains signed.
"""

from __future__ import annotations

import copy
import math
import re
from typing import Any

COORDINATES = {"schema": 2, "unit": "normalized", "space": "image",
               "x_reference": "width", "y_reference": "height",
               "projected_depth_reference": "width"}
RELATIVE_UNITS = {"normalized", "percent"}  # Read schema 1 sidecars as well.


def _size(image_size: Any) -> tuple[float, float]:
    w, h = map(float, image_size)
    if not all(math.isfinite(v) and v > 0 for v in (w, h)):
        raise ValueError("Image dimensions must be finite and positive")
    return w, h


def _position(value: Any, factor: float | None, clamp: bool) -> Any:
    if value is None:
        return None
    if factor is None:
        raise ValueError("Original image dimensions are required for pixel coordinates")
    result = value * factor
    if not math.isfinite(result):
        raise ValueError("Image coordinates must be finite")
    return max(0.0, min(1.0, result)) if clamp else result


def _points(value: Any, fx: float | None, fy: float | None, *,
            clamp: bool, depth: bool = False) -> Any:
    if isinstance(value, dict):
        return {k: _position(v, fx if k == "x" else fy, clamp) if k in {"x", "y"}
                else _position(v, fx, False) if k == "z" and depth
                else _points(v, fx, fy, clamp=clamp, depth=depth)
                for k, v in value.items()}
    if not isinstance(value, list) or not value:
        return value
    if all(v is None or isinstance(v, (int, float)) for v in value):
        return [_position(v, fx if i == 0 else fy, clamp) if i < 2
                else _position(v, fx, False) if i == 2 and depth else v
                for i, v in enumerate(value)]
    return [_points(v, fx, fy, clamp=clamp, depth=depth) for v in value]


def _span(start: float, extent: float, factor: float | None, clamp: bool) -> tuple[float, float]:
    """Clip an origin/extent pair together so the far edge stays in frame."""
    start = _position(start, factor, False)
    extent = _position(extent, factor, False)
    if not clamp:
        return start, extent
    clipped_start = max(0.0, min(1.0, start))
    clipped_extent = min(max(0.0, extent + min(start, 0.0)), 1.0 - clipped_start)
    return clipped_start, clipped_extent


def _transform(value: Any, fx: float | None, fy: float | None, *,
               normalize: bool, path: str = "", pixel_size: Any = None) -> Any:
    """Transform known image fields only; honor each explicitly tagged frame."""
    if isinstance(value, list):
        return [_transform(v, fx, fy, normalize=normalize, path=path, pixel_size=pixel_size) for v in value]
    if not isinstance(value, dict):
        return value
    coords = value.get("coordinates", {})
    if not normalize and coords.get("space") == "aligned_crop":
        return copy.deepcopy(value)
    if normalize and coords:
        unit = coords.get("unit")
        if unit in RELATIVE_UNITS:
            fx = fy = 1.0 if unit == "normalized" else 0.01
        elif unit in {None, "pixels"}:
            size = value.get("source_image_size") or value.get("image_size")
            if size is None and coords.get("space") != "aligned_crop":
                size = pixel_size
            fx, fy = (None, None) if size is None else tuple(1 / v for v in _size(size))
        else:
            raise ValueError(f"Unknown image coordinate unit: {unit}")
    result = {}
    for key, item in value.items():
        child_path = f"{path}.{key}"
        if key in {"bbox", "location"} and isinstance(item, list) and len(item) == 4:
            factors = (fy, fx, fy, fx) if key == "location" else (fx, fy, fx, fy)
            result[key] = [_position(v, f, normalize) for v, f in zip(item, factors)]
        elif key in {"bbox_width", "bbox_height", "FaceRectX", "FaceRectY",
                     "FaceRectWidth", "FaceRectHeight"} and isinstance(item, (int, float)):
            factor = fy if key in {"bbox_height", "FaceRectY", "FaceRectHeight"} else fx
            result[key] = _position(item, factor, normalize)
        elif re.fullmatch(r"(?:mesh_)?[xy]_\d+", key) and isinstance(item, (int, float)):
            result[key] = _position(item, fy if key.startswith(("y_", "mesh_y_")) else fx, normalize)
        elif key in {"landmarks", "kps", "keypoints", "landmarks_named", "Landmark106", "PIPNet"} or key.startswith(("landmark_2d_", "landmark_3d_")):
            if isinstance(item, dict) and any(re.fullmatch(r"(?:mesh_)?[xy]_\d+", k) for k in item):
                result[key] = _transform(item, fx, fy, normalize=normalize, path=child_path, pixel_size=pixel_size)
            else:
                depth = key.startswith("landmark_3d_") or ".FaceMesh." in child_path
                result[key] = _points(item, fx, fy, clamp=normalize, depth=depth)
        elif key in {"facial_area", "region"} and isinstance(item, dict):
            area = copy.deepcopy(item)
            for start, extent, factor in (("x", "w", fx), ("y", "h", fy)):
                if all(isinstance(area.get(k), (int, float)) for k in (start, extent)):
                    area[start], area[extent] = _span(item[start], item[extent], factor, normalize)
                else:
                    for field in (start, extent):
                        if isinstance(area.get(field), (int, float)):
                            area[field] = _position(item[field], factor, normalize)
            for field in ("left_eye", "right_eye", "nose", "mouth_left", "mouth_right"):
                if field in area:
                    area[field] = _points(area[field], fx, fy, clamp=normalize)
            result[key] = area
        elif key == "coordinates":
            result[key] = {**item, **COORDINATES, "space": item.get("space", "image")} if normalize else {**item, "schema": 2, "unit": "pixels"}
        else:
            result[key] = _transform(item, fx, fy, normalize=normalize, path=child_path, pixel_size=pixel_size)
    for start, extent, factor in (("FaceRectX", "FaceRectWidth", fx), ("FaceRectY", "FaceRectHeight", fy)):
        if all(isinstance(value.get(k), (int, float)) for k in (start, extent)):
            result[start], result[extent] = _span(value[start], value[extent], factor, normalize)
    bbox = result.get("bbox")
    if isinstance(bbox, list) and len(bbox) == 4 and all(isinstance(v, (int, float)) for v in bbox):
        for field, start, end in (("bbox_width", 0, 2), ("bbox_height", 1, 3)):
            if field in result:
                result[field] = max(0.0, bbox[end] - bbox[start])
    return result


def to_normalized(data: dict[str, Any], image_size: Any = None) -> dict[str, Any]:
    """Normalize and clamp positions; also upgrade schema 1 relative coordinates.

    Relative payloads do not need original dimensions. Repeated normalization
    preserves their original dimension metadata and repairs out-of-range values.
    """
    relative = data.get("coordinates", {}).get("unit") in RELATIVE_UNITS
    size = data.get("image_size") if relative else image_size or data.get("image_size")
    if size is None:
        size = image_size
    dims = _size(size) if size is not None else None
    factors = tuple(1 / v for v in dims) if dims else (None, None)
    result = _transform(data, *factors, normalize=True, pixel_size=dims)
    result["coordinates"] = {**COORDINATES, **result.get("coordinates", {})}
    if dims is not None:
        result["image_size"] = [int(v) for v in dims]
    for face in result.get("faces", []):
        if isinstance(face, dict):
            face.setdefault("coordinates", dict(COORDINATES))
            if dims is not None:
                face.setdefault("source_image_size", [int(v) for v in dims])
    return result


def record_to_pixels(record: dict[str, Any], image_size: Any,
                     *, source_image_size: Any = None) -> dict[str, Any]:
    """Resolve relative records, or rescale legacy pixels using their saved size."""
    w, h = _size(image_size)
    source = record.get("source_image_size") or source_image_size
    if record.get("coordinates", {}).get("unit") not in RELATIVE_UNITS and source is None:
        return copy.deepcopy(record)
    normalized = to_normalized(record, source)
    result = _transform(normalized, w, h, normalize=False)
    result["coordinates"] = {"schema": 2, "unit": "pixels", "space": "image"}
    result.pop("image_size", None)  # Records use source_image_size, not section dimensions.
    result["source_image_size"] = [int(w), int(h)]
    return result


def section_records_in_pixels(section: dict[str, Any], image_size: Any) -> list[dict[str, Any]]:
    records = []
    for face in section.get("faces", []):
        record = dict(face)
        if "coordinates" not in record and "coordinates" in section:
            record["coordinates"] = section["coordinates"]
        records.append(record_to_pixels(record, image_size, source_image_size=section.get("image_size")))
    return records
