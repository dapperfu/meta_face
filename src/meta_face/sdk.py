"""Lazy access to the complete public APIs of the optional face SDKs.

Recipes retain native objects between calls, so constructors, tracking, vector
stores, Fex analysis and visualization are usable without flattening their APIs.
No provider is imported, initialized or downloaded merely to list the catalog.
"""

from __future__ import annotations

import dataclasses
import importlib
import inspect
import json
import math
import os
from enum import Enum
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np

PROVIDERS = {
    "deepface": ("deepface.DeepFace", "deepface"),
    "uniface": ("uniface", "uniface"),
    "py_feat": ("feat", "py-feat"),
}

# Discovery hints, not an execution allowlist: newer public APIs are callable too.
CATALOG = {
    "deepface": {
        "detection": ["extract_faces"],
        "recognition": ["represent", "verify", "find", "register", "search", "build_index"],
        "attributes": ["analyze"],
        "video": ["stream"],
        "models": ["build_model"],
        "compatibility": ["detectFace", "cli"],
    },
    "uniface": {
        "pipeline": ["FaceAnalyzer"],
        "detection": ["BlazeFace", "CenterFace", "RetinaFace", "SCRFD", "YOLOv5Face", "YOLOv8Face"],
        "recognition": ["AdaFace", "ArcFace", "EdgeFace", "MobileFace", "SphereFace"],
        "landmarks": ["Landmark106", "PIPNet", "FaceMesh"],
        "attributes": ["AgeGender", "FairFace", "Emotion", "FaceAttribNet"],
        "gaze_pose": ["MobileGaze", "HeadPose"],
        "segmentation": ["BiSeNet", "XSeg", "MODNet"],
        "quality_liveness": ["EDifFIQA", "MiniFASNet"],
        "tracking_privacy_search": ["BYTETracker", "BlurFace", "FAISS"],
        "utilities": ["compute_similarity", "face_alignment", "download_models",
                      "get_cache_dir", "set_cache_dir", "verify_model_weights", "enable_logging"],
    },
    "py_feat": {
        "detectors": ["Detector", "Detectorv1", "Detectorv2"],
        "results": ["Fex"],
        "utilities": ["detector_capabilities", "data", "utils", "plotting", "transforms"],
    },
}


def provider_version(name: str) -> str | None:
    try:
        return version(PROVIDERS[name][1])
    except PackageNotFoundError:
        return None


def provider_issue(name: str) -> str | None:
    if provider_version(name) is None:
        return f"{name} is not installed. Install: pip install -e '.[sdk-tools]'"
    return None


def provider_options(name: str) -> dict[str, Any]:
    """Read worker-local options; fail visibly on misspelled/non-object JSON."""
    key = f"META_FACE_{name.upper()}_OPTIONS"
    value = json.loads(os.environ.get(key, "{}"))
    if not isinstance(value, dict):
        raise ValueError(f"{key} must be a JSON object")
    return value


def encode_result(value: Any) -> Any:
    """Preserve SDK result fields, arrays, dataclasses, tensors and Fex columns."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Enum):
        return encode_result(value.value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): encode_result(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [encode_result(v) for v in value]
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {f.name: encode_result(getattr(value, f.name)) for f in dataclasses.fields(value)}
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        return encode_result(value.detach().cpu().tolist())
    if hasattr(value, "columns") and hasattr(value, "to_dict"):
        return encode_result(value.to_dict(orient="records"))
    if hasattr(value, "tolist"):
        return encode_result(value.tolist())
    if hasattr(value, "to_dict"):
        return encode_result(value.to_dict())
    raise TypeError(
        f"{type(value).__name__} is a native SDK object. Keep it in a recipe step, "
        "call a public method, or select a serializable output."
    )


def _public_attr(obj: Any, name: str) -> Any:
    if not name.isidentifier() or name.startswith("_"):
        raise ValueError(f"Only public SDK attributes are supported: {name!r}")
    try:
        return getattr(obj, name)
    except AttributeError:
        if isinstance(obj, ModuleType) and hasattr(obj, "__path__"):
            return importlib.import_module(f"{obj.__name__}.{name}")
        raise


class SDKSession:
    """One provider and native result/object references shared across calls."""

    def __init__(self, provider: str):
        if provider not in PROVIDERS:
            raise ValueError(f"Unknown SDK {provider!r}; choose {', '.join(PROVIDERS)}")
        self.provider = provider
        self.objects: dict[str, Any] = {}
        self._module: ModuleType | None = None

    @property
    def module(self) -> ModuleType:
        if self._module is None:
            try:
                self._module = importlib.import_module(PROVIDERS[self.provider][0])
            except ImportError as exc:
                raise ImportError(
                    f"Cannot import {self.provider}: {exc}. "
                    "Install its dependencies with pip install -e '.[sdk-tools]'."
                ) from exc
        return self._module

    def get(self, target: str) -> Any:
        """Resolve a public provider symbol or $step.public_method (no eval)."""
        parts = target.split(".")
        if parts[0].startswith("$"):
            value = self.objects[parts.pop(0)[1:]]
        else:
            value = self.module
        for part in parts:
            value = _public_attr(value, part)
        return value

    def resolve(self, value: Any) -> Any:
        """Resolve JSON inputs to native references, images, arrays or tensors."""
        if isinstance(value, list):
            return [self.resolve(v) for v in value]
        if not isinstance(value, dict):
            return value
        if "$ref" in value:
            result = self.objects[value["$ref"]]
            for part in value.get("path", []):
                if isinstance(part, int) or isinstance(result, dict):
                    result = result[part]
                else:
                    result = _public_attr(result, part)
            return result
        if "$symbol" in value:
            return self.get(value["$symbol"])
        if "$image" in value:
            from meta_face.imaging import load_image

            result = load_image(Path(value["$image"]))
            color = value.get("color", "bgr")
            if color not in {"bgr", "rgb"}:
                raise ValueError("$image color must be bgr or rgb")
            if color == "rgb":
                result = result[:, :, ::-1].copy()
            layout = value.get("layout", "HWC")
            if layout == "CHW":
                result = result.transpose(2, 0, 1).copy()
            elif layout != "HWC":
                raise ValueError("$image layout must be HWC or CHW")
            return result[None] if value.get("batch", False) else result
        if "$array" in value:
            return np.asarray(self.resolve(value["$array"]), dtype=value.get("dtype"))
        if "$tensor" in value:
            import torch

            result = torch.as_tensor(self.resolve(value["$tensor"]))
            if "dtype" in value:
                dtype = _public_attr(torch, value["dtype"])
                if not isinstance(dtype, torch.dtype):
                    raise ValueError("$tensor dtype must name a torch dtype")
                result = result.to(dtype=dtype)
            return result.to(value.get("device", "cpu"))
        if "$tuple" in value:
            return tuple(self.resolve(v) for v in value["$tuple"])
        return {k: self.resolve(v) for k, v in value.items()}

    def call(self, target: str, *args: Any, **kwargs: Any) -> Any:
        """Call any public SDK operation with its original arguments and result."""
        operation = self.get(target)
        if not callable(operation):
            raise TypeError(f"SDK target {target!r} is not callable")
        return operation(*args, **kwargs)

    def run(self, recipe: dict[str, Any]) -> Any:
        """Execute ordered native calls, retaining objects for later recipe steps."""
        steps = recipe.get("steps")
        if not isinstance(steps, list) or not steps:
            raise ValueError("A recipe requires a nonempty steps list")
        result: Any = None
        for step in steps:
            name = step["id"]
            if not isinstance(name, str) or not name.isidentifier() or name.startswith("_"):
                raise ValueError(f"Invalid step id: {name!r}")
            if name in self.objects:
                raise ValueError(f"Duplicate step id: {name}")
            result = self.call(
                step["call"],
                *self.resolve(step.get("args", [])),
                **self.resolve(step.get("kwargs", {})),
            )
            self.objects[name] = result
        return self.resolve(recipe["output"]) if "output" in recipe else result


def describe_sdk(provider: str, target: str | None = None) -> dict[str, Any]:
    """List the catalog offline, or inspect an installed public object without constructing it."""
    if provider not in PROVIDERS:
        raise ValueError(f"Unknown SDK: {provider}")
    info: dict[str, Any] = {
        "provider": provider, "version": provider_version(provider), "catalog": CATALOG[provider],
    }
    if target is not None:
        session = SDKSession(provider)
        obj = session.get(target) if target else session.module
        members: dict[str, Any] = {}
        for name in dir(obj):
            if name.startswith("_"):
                continue
            value = getattr(obj, name)
            if not callable(value):
                continue
            try:
                signature = str(inspect.signature(value))
            except (TypeError, ValueError):
                signature = None
            members[name] = {"signature": signature, "doc": inspect.getdoc(value)}
        info["public_api"] = members
        info["doc"] = inspect.getdoc(obj)
    return info
