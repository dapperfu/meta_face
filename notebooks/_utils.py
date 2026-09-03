"""Matplotlib display helpers for annotation notebooks."""

from __future__ import annotations

from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np

from meta_face.bbox import crop_face, expand_bbox
from meta_face.coordinates import record_to_pixels

def _record_in_pixels(record: dict[str, Any], image: np.ndarray) -> dict[str, Any]:
    """Resolve stored fractions to pixels so crops contain the face, not a 1-px box."""
    h_img, w_img = image.shape[:2]
    return record_to_pixels(record, (w_img, h_img))


_FACE_BBOX_COLORS: tuple[tuple[int, int, int], ...] = (
    (0, 0, 255),
    (0, 255, 0),
    (255, 0, 0),
    (0, 255, 255),
    (255, 0, 255),
    (255, 255, 0),
)


def bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    """Convert OpenCV BGR uint8 image to RGB for matplotlib."""
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def show_original_and_annotated(
    original: np.ndarray,
    annotated: np.ndarray,
    *,
    title: str = "Original vs annotated",
) -> None:
    """Display unannotated and annotated images side by side."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    axes[0].imshow(bgr_to_rgb(original))
    axes[0].set_title("Original")
    axes[0].axis("off")
    axes[1].imshow(bgr_to_rgb(annotated))
    axes[1].set_title("Annotated")
    axes[1].axis("off")
    fig.suptitle(title)
    plt.tight_layout()
    plt.show()


def show_bbox_comparison(
    image: np.ndarray,
    records: list[dict[str, Any]],
    buffer_pct: float,
    *,
    title: str = "Tight (solid) vs buffered (dashed) bboxes",
) -> None:
    """Draw tight bboxes (solid) and buffered bboxes (dashed) on the image."""
    canvas = image.copy()
    h_img, w_img = canvas.shape[:2]
    for idx, record in enumerate(records):
        color = _FACE_BBOX_COLORS[idx % len(_FACE_BBOX_COLORS)]
        bbox = _record_in_pixels(record, image)["bbox"]
        x1, y1, x2, y2 = (int(round(v)) for v in bbox[:4])
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)

        bx1, by1, bx2, by2 = (
            int(round(v)) for v in expand_bbox(bbox, buffer_pct, image_size=(w_img, h_img))
        )
        _draw_dashed_rect(canvas, bx1, by1, bx2, by2, color, thickness=2)

    plt.figure(figsize=(10, 8))
    plt.imshow(bgr_to_rgb(canvas))
    plt.title(f"{title} (buffer={buffer_pct}%)")
    plt.axis("off")
    plt.show()


def show_face_crop_grid(
    image: np.ndarray,
    records: list[dict[str, Any]],
    buffer_pct: float,
    *,
    title: str = "Face crops: tight vs buffered",
) -> None:
    """Grid of per-face crops: column 0 tight, column 1 buffered."""
    n_faces = len(records)
    if n_faces == 0:
        print("No faces to display.")
        return

    fig, axes = plt.subplots(n_faces, 2, figsize=(8, 4 * n_faces), squeeze=False)
    for row, record in enumerate(records):
        bbox = _record_in_pixels(record, image)["bbox"]
        tight = crop_face(image, bbox, buffer_pct=0.0)
        buffered = crop_face(image, bbox, buffer_pct=buffer_pct)

        axes[row, 0].imshow(bgr_to_rgb(tight))
        axes[row, 0].set_title(f"Face {row + 1} — tight")
        axes[row, 0].axis("off")

        axes[row, 1].imshow(bgr_to_rgb(buffered))
        axes[row, 1].set_title(f"Face {row + 1} — buffer {buffer_pct}%")
        axes[row, 1].axis("off")

    fig.suptitle(title)
    plt.tight_layout()
    plt.show()


def _draw_numbered_face_bboxes(
    image: np.ndarray,
    records: list[dict[str, Any]],
) -> np.ndarray:
    """Draw colored bboxes with face index labels for side-by-side comparison."""
    canvas = image.copy()
    for idx, record in enumerate(records):
        color = _FACE_BBOX_COLORS[idx % len(_FACE_BBOX_COLORS)]
        bbox = _record_in_pixels(record, image)["bbox"]
        x1, y1, x2, y2 = (int(round(v)) for v in bbox[:4])
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        label = f"Face {idx + 1}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        ty = max(y1 - 6, th + 4)
        cv2.rectangle(canvas, (x1, ty - th - 4), (x1 + tw + 4, ty + 4), color, -1)
        cv2.putText(
            canvas,
            label,
            (x1 + 2, ty),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    return canvas


def show_photo_with_face_attributes(
    image: np.ndarray,
    records: list[dict[str, Any]],
    *,
    cluster_labels: list[int] | None = None,
    title: str = "Photo with face attributes",
    annotate_faces: bool = True,
) -> None:
    """Display full photo beside stacked face attribute text for comparison."""
    from _face_info import format_face_attributes

    n_faces = len(records)
    display = _draw_numbered_face_bboxes(image, records) if annotate_faces and n_faces else image

    h_img, w_img = image.shape[:2]
    img_aspect = w_img / h_img if h_img else 1.0
    fig_h = max(8.0, min(16.0, 6.0 + n_faces * 0.75))
    fig_w = fig_h * img_aspect + 6.0

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(fig_w, fig_h),
        gridspec_kw={"width_ratios": [img_aspect, 0.45]},
    )

    img_ax = axes[0]
    img_ax.imshow(bgr_to_rgb(display))
    img_ax.set_title("Annotated photo" if annotate_faces and n_faces else "Photo")
    img_ax.axis("off")

    text_ax = axes[1]
    text_ax.axis("off")
    if n_faces == 0:
        text_ax.text(
            0,
            1,
            "No faces detected.",
            transform=text_ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            fontfamily="monospace",
        )
    else:
        blocks: list[str] = []
        for idx, record in enumerate(records):
            cluster: int | None = None
            if cluster_labels is not None and idx < len(cluster_labels):
                cluster = cluster_labels[idx]
            blocks.append(format_face_attributes(record, face_index=idx, cluster_label=cluster))
        text_ax.text(
            0,
            1,
            "\n\n".join(blocks),
            transform=text_ax.transAxes,
            fontsize=9,
            verticalalignment="top",
            fontfamily="monospace",
        )

    fig.suptitle(title)
    plt.tight_layout()
    plt.show()


def show_face_metadata_crop_grid(
    image: np.ndarray,
    records: list[dict[str, Any]],
    buffer_pct: float,
    *,
    cluster_labels: list[int] | None = None,
    title: str = "Buffered face crops with metadata",
    use_buffered: bool = True,
) -> None:
    """Per-face grid: buffered (or tight) crop beside full attribute text."""
    from _face_info import compact_face_label, format_face_attributes

    n_faces = len(records)
    if n_faces == 0:
        print("No faces to display.")
        return

    fig, axes = plt.subplots(
        n_faces,
        2,
        figsize=(14, 4 * n_faces),
        squeeze=False,
        gridspec_kw={"width_ratios": [1, 1.2]},
    )

    for row, record in enumerate(records):
        bbox = _record_in_pixels(record, image)["bbox"]
        buf = buffer_pct if use_buffered else 0.0
        crop = crop_face(image, bbox, buffer_pct=buf)

        cluster: int | None = None
        if cluster_labels is not None and row < len(cluster_labels):
            cluster = cluster_labels[row]

        img_ax = axes[row, 0]
        img_ax.imshow(bgr_to_rgb(crop))
        buf_note = f"buffer {buffer_pct}%" if use_buffered else "tight"
        img_ax.set_title(
            f"{compact_face_label(record, face_index=row, cluster_label=cluster)}\n({buf_note})"
        )
        img_ax.axis("off")

        text_ax = axes[row, 1]
        text_ax.axis("off")
        meta = format_face_attributes(record, face_index=row, cluster_label=cluster)
        text_ax.text(
            0,
            1,
            meta,
            transform=text_ax.transAxes,
            fontsize=9,
            verticalalignment="top",
            fontfamily="monospace",
        )

    fig.suptitle(title)
    plt.tight_layout()
    plt.show()


def show_buffer_comparison_row(
    image: np.ndarray,
    bbox: list[float],
    buffer_values: list[float],
    *,
    title: str = "Buffer comparison for one face",
) -> None:
    """Single-row comparison of one face at multiple buffer percentages."""
    n = len(buffer_values)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4), squeeze=False)
    for col, buf in enumerate(buffer_values):
        crop = crop_face(image, bbox, buffer_pct=buf)
        axes[0, col].imshow(bgr_to_rgb(crop))
        axes[0, col].set_title(f"buffer {buf}%")
        axes[0, col].axis("off")
    fig.suptitle(title)
    plt.tight_layout()
    plt.show()


def _draw_dashed_rect(
    canvas: np.ndarray,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    color: tuple[int, int, int],
    *,
    thickness: int = 2,
    dash_len: int = 8,
) -> None:
    """Draw a dashed rectangle on a BGR image."""
    segments = [
        ((x1, y1), (x2, y1)),
        ((x2, y1), (x2, y2)),
        ((x2, y2), (x1, y2)),
        ((x1, y2), (x1, y1)),
    ]
    for (sx, sy), (ex, ey) in segments:
        _draw_dashed_line(canvas, sx, sy, ex, ey, color, thickness=thickness, dash_len=dash_len)


def _draw_dashed_line(
    canvas: np.ndarray,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    color: tuple[int, int, int],
    *,
    thickness: int = 2,
    dash_len: int = 8,
) -> None:
    length = int(np.hypot(x2 - x1, y2 - y1))
    if length == 0:
        return
    for i in range(0, length, dash_len * 2):
        t0 = i / length
        t1 = min((i + dash_len) / length, 1.0)
        sx = int(round(x1 + (x2 - x1) * t0))
        sy = int(round(y1 + (y2 - y1) * t0))
        ex = int(round(x1 + (x2 - x1) * t1))
        ey = int(round(y1 + (y2 - y1) * t1))
        cv2.line(canvas, (sx, sy), (ex, ey), color, thickness)
