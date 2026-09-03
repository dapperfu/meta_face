"""UniFace 4 detection, recognition and all static facial analysis heads.

Tracking, stores, matting and anonymization are available through mf sdk.
"""

from __future__ import annotations

import json
from functools import lru_cache
from typing import Any

from meta_face.bbox import crop_face
from meta_face.sdk import SDKSession, encode_result, provider_issue, provider_options, provider_version
from meta_face.tools.analysis.base import FaceContext, face_results_payload

TOOL_NAME = "uniface"
TOOL_VERSION = "2.0.0"
MODEL_NAME = "uniface"
PHOTO_ANALYSES = (
    "AgeGender", "FairFace", "Emotion", "FaceAttribNet", "Landmark106", "PIPNet",
    "FaceMesh", "MobileGaze", "HeadPose", "EDifFIQA", "MiniFASNet", "BiSeNet", "XSeg",
)


@lru_cache(maxsize=1)
def _get_models(config_json: str) -> tuple[Any, Any, dict[str, Any]]:
    options = json.loads(config_json)
    sdk = SDKSession(TOOL_NAME)

    def build(spec: str | dict[str, Any]) -> Any:
        if isinstance(spec, str):
            return sdk.call(spec)
        return sdk.call(spec["class"], **sdk.resolve(spec.get("kwargs", {})))

    detector = build(options.get("detector", "SCRFD"))
    spec = options.get("recognizer", "ArcFace")
    recognizer = None if spec is None else build(spec)
    names = options.get("analyses", list(PHOTO_ANALYSES))
    if set(names) - set(PHOTO_ANALYSES):
        raise ValueError(f"Unknown UniFace photo analysis: {set(names) - set(PHOTO_ANALYSES)}")
    configs = options.get("models", {})
    models = {name: build({"class": name, "kwargs": configs.get(name, {})}) for name in names}
    return detector, recognizer, models


def availability() -> str | None:
    issue = provider_issue(TOOL_NAME)
    installed = provider_version(TOOL_NAME)
    if issue is None and installed and int(installed.split(".")[0]) < 4:
        return "UniFace 4 is required. Install: pip install -e '.[uniface]'"
    return issue


def analyze_faces(image_bgr: Any, faces: list[FaceContext]) -> dict[str, Any]:
    options = provider_options(TOOL_NAME)
    unknown = set(options) - {"detector", "recognizer", "analyses", "models", "calls"}
    if unknown:
        raise ValueError(f"Unknown UniFace options: {sorted(unknown)}")
    detector, recognizer, models = _get_models(json.dumps(options, sort_keys=True))
    calls = options.get("calls", {})
    detections = detector.detect(image_bgr, **calls.get("detect", {}))
    records = []
    for idx, face in enumerate(detections):
        embedding = None
        if recognizer is not None:
            embedding = recognizer.get_normalized_embedding(image_bgr, face.landmarks)
        results = {}
        for name, model in models.items():
            kwargs = calls.get(name, {})
            if name in {"AgeGender", "FairFace", "Emotion", "FaceAttribNet"}:
                result = model.predict(image_bgr, face, **kwargs)
            elif name in {"Landmark106", "PIPNet"}:
                result = model.get_landmarks(image_bgr, face.bbox, **kwargs)
            elif name == "FaceMesh":
                result = model.predict(image_bgr, faces=[face], **kwargs)
            elif name in {"MobileGaze", "HeadPose"}:
                crop = crop_face(image_bgr, face.bbox, buffer_pct=0)
                result = model.estimate(crop, **kwargs)
            elif name == "EDifFIQA":
                result = model.predict(image_bgr, face.landmarks, **kwargs)
            elif name == "MiniFASNet":
                result = model.predict(image_bgr, face.bbox, **kwargs)
            else:
                result = model.parse(image_bgr, landmarks=face.landmarks, **kwargs)
            results[name] = encode_result(result)
        record = encode_result(face)
        record.update(face_index=idx, analyses=results)
        if embedding is not None:
            record["embedding"] = encode_result(embedding)
        records.append(record)
    return face_results_payload(records, model=MODEL_NAME, extra={
        "sdk_version": provider_version(TOOL_NAME), "options": options,
        "face_index_source": TOOL_NAME, "analyses": list(models),
        "detector_class": type(detector).__name__,
        "recognizer_class": type(recognizer).__name__ if recognizer is not None else None,
    })
