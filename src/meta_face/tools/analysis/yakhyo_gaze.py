"""yakhyo/gaze-estimation gaze direction (optional dependency)."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from meta_face.config import yakhyo_gaze_model_path
from meta_face.tools.analysis.base import FaceContext, face_results_payload
from meta_face.tools.analysis.decode import GAZE_INPUT, gaze_angles_degrees, rgb_normalized_tensor

TOOL_NAME = "yakhyo_gaze"
TOOL_VERSION = "1.1.0"
MODEL_NAME = "yakhyo_gaze"


@lru_cache(maxsize=1)
def _get_session():
    import onnxruntime as ort

    model_path = yakhyo_gaze_model_path()
    if not model_path.is_file():
        raise FileNotFoundError(
            f"yakhyo gaze ONNX model missing at {model_path}. "
            "Run: mf download --backend yakhyo_gaze"
        )
    return ort.InferenceSession(
        str(model_path),
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )


def availability() -> str | None:
    try:
        import onnxruntime  # noqa: F401
    except ImportError:
        return "onnxruntime is required for yakhyo_gaze."
    try:
        _get_session()
    except FileNotFoundError as exc:
        return str(exc)
    except Exception as exc:
        return f"yakhyo_gaze failed to load: {exc}"
    return None


def analyze_faces(
    image_bgr: Any,
    faces: list[FaceContext],
) -> dict[str, Any]:
    del image_bgr
    session = _get_session()
    input_name = session.get_inputs()[0].name
    per_face: list[dict[str, Any]] = []
    for ctx in faces:
        outputs = session.run(None, {input_name: rgb_normalized_tensor(ctx.crop_bgr, GAZE_INPUT, imagenet=True)})
        yaw, pitch = gaze_angles_degrees(outputs)
        per_face.append(
            {
                "face_index": ctx.face_index,
                "gaze": {"yaw": yaw, "pitch": pitch, "units": "degrees"},
            }
        )
    return face_results_payload(per_face, model=MODEL_NAME)
