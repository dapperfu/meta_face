"""Render face detection overlays to *_scrfd.* images."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from meta_face.config import ANNOTATE_OUTPUT_SUFFIX
from meta_face.imaging import is_image_path, load_image, save_image
from meta_face.sidecar import get_face_section, load_or_create, sidecar_path_for_media
from meta_face.tools.face_record import faces_to_annotation_records
from meta_face.tools.scrfd import detect_faces

_HEIC_SUFFIXES = {".heic", ".heif"}

# Distinct BGR colors per face index.
_FACE_COLORS: tuple[tuple[int, int, int], ...] = (
    (0, 0, 255),
    (0, 255, 0),
    (255, 0, 0),
    (0, 255, 255),
    (255, 0, 255),
    (255, 255, 0),
)

_FONT = cv2.FONT_HERSHEY_SIMPLEX
_FONT_SCALE = 0.45
_FONT_THICKNESS = 1
_LINE_HEIGHT = 16
_PANEL_PAD = 4


@dataclass
class AnnotateStats:
    discovered: int = 0
    written: int = 0
    skipped: int = 0
    errors: int = 0


def annotated_output_path(media_path: Path) -> Path:
    """Resolve output path: photo.jpg -> photo_scrfd.jpg; HEIC -> *_scrfd.jpg."""
    suffix = media_path.suffix.lower()
    out_suffix = ".jpg" if suffix in _HEIC_SUFFIXES else media_path.suffix
    return media_path.with_name(f"{media_path.stem}_{ANNOTATE_OUTPUT_SUFFIX}{out_suffix}")


def annotated_output_exists(media_path: Path) -> bool:
    """True when the canonical annotated output file already exists."""
    return annotated_output_path(media_path).exists()


def _cluster_labels_for_media(media_path: Path) -> list[int] | None:
    scar_path = sidecar_path_for_media(media_path)
    if not scar_path.exists():
        return None
    doc, _ = load_or_create(media_path)
    section = get_face_section(doc, "cluster")
    labels = section.get("labels")
    if not isinstance(labels, list):
        return None
    return [int(x) for x in labels]


def _sex_label(record: dict[str, Any]) -> str | None:
    if record.get("sex"):
        return str(record["sex"])
    gender = record.get("gender")
    if gender is None:
        return None
    return "M" if int(gender) == 1 else "F"


def _panel_lines(record: dict[str, Any], cluster_label: int | None) -> list[str]:
    lines: list[str] = []
    sex = _sex_label(record)
    age = record.get("age")
    if sex is not None and age is not None:
        lines.append(f"{sex} {int(age)}")
    elif sex is not None:
        lines.append(sex)
    elif age is not None:
        lines.append(f"age {int(age)}")

    det = record.get("det_score")
    if det is not None:
        lines.append(f"det {float(det):.2f}")

    pose = record.get("pose")
    if isinstance(pose, list) and len(pose) >= 3:
        p, y, r = float(pose[0]), float(pose[1]), float(pose[2])
        lines.append(f"pose P:{p:+.1f} Y:{y:+.1f} R:{r:+.1f}")

    if cluster_label is not None:
        if cluster_label < 0:
            lines.append("cluster noise")
        else:
            lines.append(f"cluster {cluster_label}")

    return lines


def _draw_panel(
    canvas: np.ndarray,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    lines: list[str],
    color: tuple[int, int, int],
) -> None:
    if not lines:
        return
    h_img, w_img = canvas.shape[:2]
    max_width = 0
    text_heights: list[tuple[str, tuple[int, int]]] = []
    for line in lines:
        (tw, th), _ = cv2.getTextSize(line, _FONT, _FONT_SCALE, _FONT_THICKNESS)
        max_width = max(max_width, tw)
        text_heights.append((line, (tw, th)))

    panel_h = _PANEL_PAD * 2 + _LINE_HEIGHT * len(lines)
    panel_w = _PANEL_PAD * 2 + max_width

    px1 = max(0, min(x1, w_img - 1))
    py2 = max(0, min(y2, h_img - 1))
    py1 = max(0, py2 - panel_h + 1)
    px2 = min(w_img - 1, px1 + panel_w - 1)
    if py1 > py2 or px1 > px2:
        return

    overlay = canvas.copy()
    cv2.rectangle(overlay, (px1, py1), (px2, py2), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, canvas, 0.45, 0, canvas)

    y_text = py1 + _PANEL_PAD + text_heights[0][1][1]
    for line, _ in text_heights:
        cv2.putText(
            canvas,
            line,
            (px1 + _PANEL_PAD, y_text),
            _FONT,
            _FONT_SCALE,
            color,
            _FONT_THICKNESS,
            cv2.LINE_AA,
        )
        y_text += _LINE_HEIGHT


def draw_annotations(
    image: np.ndarray,
    records: list[dict[str, Any]],
    *,
    cluster_labels: list[int] | None = None,
    dense_landmarks: bool = True,
) -> np.ndarray:
    """Draw bboxes, landmarks, and per-face info panels on a copy of image."""
    canvas = image.copy()
    for idx, record in enumerate(records):
        color = _FACE_COLORS[idx % len(_FACE_COLORS)]
        bbox = record["bbox"]
        x1, y1, x2, y2 = (int(round(v)) for v in bbox[:4])
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)

        kps = record.get("kps")
        if kps:
            for li, (kx, ky) in enumerate(kps):
                pt_color = (0, 255, 0) if li in (0, 3) else (0, 0, 255)
                cv2.circle(canvas, (int(round(kx)), int(round(ky))), 2, pt_color, 2)

        if dense_landmarks:
            dense = record.get("landmark_2d_106")
            if dense:
                for px, py in dense:
                    cv2.circle(
                        canvas,
                        (int(round(px)), int(round(py))),
                        1,
                        (180, 180, 180),
                        -1,
                    )

        cluster_label: int | None = None
        if cluster_labels is not None and idx < len(cluster_labels):
            cluster_label = cluster_labels[idx]

        lines = _panel_lines(record, cluster_label)
        _draw_panel(canvas, x1, y1, x2, y2, lines, color)

    return canvas


def annotate_image(
    media_path: Path,
    *,
    force: bool = False,
    dense_landmarks: bool = True,
) -> Path | None:
    """
    Run inference, draw overlays, write *_scrfd.* next to the source image.

    Returns output path when written, None when skipped.
    """
    media_path = media_path.resolve()
    out_path = annotated_output_path(media_path)
    if out_path.exists() and not force:
        return None

    image = load_image(media_path)
    faces = detect_faces(image)
    records = faces_to_annotation_records(faces)
    cluster_labels = _cluster_labels_for_media(media_path)
    annotated = draw_annotations(
        image,
        records,
        cluster_labels=cluster_labels,
        dense_landmarks=dense_landmarks,
    )
    save_image(out_path, annotated)
    return out_path


def iter_annotate_paths(root: Path, *, recursive: bool = True) -> list[Path]:
    """Collect image paths under root (file or directory)."""
    root = root.resolve()
    if root.is_file():
        return [root] if is_image_path(root) else []
    if not root.is_dir():
        return []
    if recursive:
        candidates = root.rglob("*")
    else:
        candidates = root.iterdir()
    return sorted(p for p in candidates if p.is_file() and is_image_path(p))


def annotate_path(
    root: Path,
    *,
    recursive: bool = True,
    force: bool = False,
    dense_landmarks: bool = True,
) -> AnnotateStats:
    """Annotate all images under root; return aggregate stats."""
    stats = AnnotateStats()
    paths = iter_annotate_paths(root, recursive=recursive)
    stats.discovered = len(paths)
    for media_path in paths:
        try:
            out_path = annotated_output_path(media_path)
            if out_path.exists() and not force:
                stats.skipped += 1
                continue
            result = annotate_image(
                media_path,
                force=force,
                dense_landmarks=dense_landmarks,
            )
            if result is None:
                stats.skipped += 1
            else:
                stats.written += 1
        except Exception:
            stats.errors += 1
    return stats
