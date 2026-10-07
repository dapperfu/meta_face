"""Export weights to ONNX and check the export against the PyTorch model."""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from meta_face.yolo_face.detection import Detection
from meta_face.yolo_face.detector import FaceDetector
from meta_face.yolo_face.errors import MissingDependencyError, ModelLoadError, ModelNotFoundError
from meta_face.yolo_face.sources import load_image

logger = logging.getLogger(__name__)

DEFAULT_MIN_IOU = 0.9
DEFAULT_CONF_TOLERANCE = 0.05


def export_onnx(
    model_path: Path,
    *,
    image_size: int = 640,
    half: bool = False,
    dynamic: bool = False,
    simplify: bool = False,
    opset: int | None = None,
) -> Path:
    """Write ``<model>.onnx`` next to the weights via Ultralytics and return its path."""
    if not model_path.is_file():
        raise ModelNotFoundError(f"Model file does not exist: {model_path}")
    try:
        from ultralytics import YOLO
    except ImportError:
        raise MissingDependencyError(
            "ultralytics is required for export: uv pip install -e '.[yolo]'"
        ) from None
    try:
        exported = YOLO(str(model_path), task="detect").export(
            format="onnx",
            imgsz=image_size,
            half=half,
            dynamic=dynamic,
            simplify=simplify,
            opset=opset,
            device="0" if half else "cpu",
        )
    except Exception as exc:
        raise ModelLoadError(f"ONNX export failed for '{model_path}': {exc}") from exc
    return Path(exported)


def iou(a: Detection, b: Detection) -> float:
    ix = max(0, min(a.x2, b.x2) - max(a.x1, b.x1))
    iy = max(0, min(a.y2, b.y2) - max(a.y1, b.y1))
    inter = ix * iy
    union = a.area + b.area - inter
    return inter / union if union > 0 else 0.0


@dataclass
class ImageComparison:
    image: Path
    reference: int
    candidate: int
    matched: int
    worst_iou: float
    worst_conf_delta: float
    problems: list[str] = field(default_factory=list)


@dataclass
class ComparisonReport:
    images: list[ImageComparison]

    @property
    def ok(self) -> bool:
        return all(not img.problems for img in self.images)

    def summary(self) -> str:
        lines = []
        for img in self.images:
            status = "OK" if not img.problems else "MISMATCH"
            lines.append(
                f"{status} {img.image.name}: ref={img.reference} onnx={img.candidate} "
                f"matched={img.matched} worst_iou={img.worst_iou:.3f} "
                f"worst_conf_delta={img.worst_conf_delta:.3f}"
            )
            lines.extend(f"    {p}" for p in img.problems)
        return "\n".join(lines)


def _compare(
    path: Path,
    ref: Sequence[Detection],
    cand: Sequence[Detection],
    threshold: float,
    min_iou: float,
    conf_tol: float,
) -> ImageComparison:
    unmatched = list(cand)
    matched = 0
    worst_iou, worst_conf = 1.0, 0.0
    problems: list[str] = []
    for r in ref:
        best = max(unmatched, key=lambda c: iou(r, c), default=None)
        score = iou(r, best) if best is not None else 0.0
        if best is None or score < 0.5:
            if r.confidence >= threshold + conf_tol:
                problems.append(f"missing box {r.xyxy} conf={r.confidence:.3f}")
            continue
        unmatched.remove(best)
        matched += 1
        worst_iou = min(worst_iou, score)
        delta = abs(r.confidence - best.confidence)
        worst_conf = max(worst_conf, delta)
        if score < min_iou:
            problems.append(f"box {r.xyxy} vs {best.xyxy} IoU {score:.3f} < {min_iou}")
        if delta > conf_tol:
            problems.append(f"box {r.xyxy} confidence differs by {delta:.3f}")
    for c in unmatched:
        if c.confidence >= threshold + conf_tol:
            problems.append(f"extra box {c.xyxy} conf={c.confidence:.3f}")
    return ImageComparison(path, len(ref), len(cand), matched, worst_iou, worst_conf, problems)


def compare_detectors(
    reference: FaceDetector,
    candidate: FaceDetector,
    images: Iterable[Path],
    *,
    min_iou: float = DEFAULT_MIN_IOU,
    conf_tolerance: float = DEFAULT_CONF_TOLERANCE,
) -> ComparisonReport:
    """
    Match boxes between two detectors on each image.

    Boxes whose confidence sits within ``conf_tolerance`` of the threshold may appear in
    only one output without counting as a mismatch.
    """
    threshold = max(reference.confidence_threshold, candidate.confidence_threshold)
    results = []
    for path in images:
        frame = load_image(path)
        results.append(
            _compare(
                path,
                reference.detect(frame),
                candidate.detect(frame),
                threshold,
                min_iou,
                conf_tolerance,
            )
        )
    return ComparisonReport(results)
