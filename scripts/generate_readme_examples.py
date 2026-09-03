"""Build compact README example images from remaining test_images (2008–2012)."""

from __future__ import annotations

import csv
import math
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TEST = ROOT / "test_images"
REPORTS = ROOT / "reports" / "test_images"
OUT = ROOT / "docs" / "readme_examples"
MAX_W = 800
SOLO_FILE = "20120608_114901.600.jpg"

_PARSE_COLORS = np.array(
    [
        [0, 0, 0],
        [255, 170, 150],
        [80, 50, 50],
        [255, 220, 80],
        [50, 50, 50],
        [180, 120, 80],
        [220, 160, 120],
        [40, 40, 200],
        [40, 40, 200],
        [200, 40, 40],
        [200, 40, 40],
        [40, 200, 40],
        [40, 200, 40],
        [200, 80, 200],
        [80, 200, 200],
        [255, 255, 255],
        [200, 200, 200],
        [120, 80, 200],
        [80, 160, 255],
    ],
    dtype=np.uint8,
)


def _is_kept_photo(name: str) -> bool:
    return name.endswith(".jpg") and not name.startswith("2026")


def _load_bgr(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    return image


def _resize_max_w(image: np.ndarray, max_w: int = MAX_W) -> np.ndarray:
    height, width = image.shape[:2]
    if width <= max_w:
        return image
    scale = max_w / width
    return cv2.resize(image, (max_w, int(round(height * scale))), interpolation=cv2.INTER_AREA)


def _save_jpg(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(path), image, [cv2.IMWRITE_JPEG_QUALITY, 82])
    if not ok:
        raise RuntimeError(f"failed to write {path}")


def _sidecar_face(file_name: str, tool: str) -> dict:
    from meta_face.sidecar import get_face_section, load_or_create

    doc, _ = load_or_create(TEST / file_name)
    faces = get_face_section(doc, tool).get("faces") or []
    if not faces:
        raise RuntimeError(f"no {tool} faces in sidecar for {file_name}")
    return faces[0]


def _pixel_bbox(file_name: str) -> tuple[int, int, int, int]:
    from meta_face.coordinates import record_to_pixels

    image = _load_bgr(TEST / file_name)
    height, width = image.shape[:2]
    rec = record_to_pixels(_sidecar_face(file_name, "scrfd"), (width, height))
    x1, y1, x2, y2 = (int(round(float(v))) for v in rec["bbox"][:4])
    return x1, y1, x2, y2


def _faces_for(file_name: str) -> list[dict[str, str]]:
    with (REPORTS / "faces.csv").open(newline="") as handle:
        rows = [
            row
            for row in csv.DictReader(handle)
            if row["file"] == file_name and _is_kept_photo(row["file"])
        ]
    rows.sort(key=lambda row: float(row["confidence"]), reverse=True)
    return rows


def _draw_boxes(image: np.ndarray, rows: list[dict[str, str]], thickness: int = 3) -> np.ndarray:
    out = image.copy()
    for index, row in enumerate(rows):
        x1 = int(round(float(row["x1"])))
        y1 = int(round(float(row["y1"])))
        x2 = int(round(float(row["x2"])))
        y2 = int(round(float(row["y2"])))
        color = (0, 220, 80) if row["small_face_under_40px"] != "True" else (0, 165, 255)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)
        label = str(index)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(out, (x1, max(0, y1 - th - 8)), (x1 + tw + 6, y1), color, -1)
        cv2.putText(
            out,
            label,
            (x1 + 3, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )
    return out


def _caption_bar(image: np.ndarray, text: str) -> np.ndarray:
    bar_h = 44
    bar = np.full((bar_h, image.shape[1], 3), 24, dtype=np.uint8)
    cv2.putText(bar, text, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (240, 240, 240), 2, cv2.LINE_AA)
    return np.vstack([bar, image])


def make_overlay(file_name: str, caption: str, out_name: str, thickness: int = 4) -> None:
    image = _load_bgr(TEST / file_name)
    boxed = _draw_boxes(image, _faces_for(file_name), thickness=thickness)
    framed = _caption_bar(_resize_max_w(boxed), caption)
    _save_jpg(OUT / out_name, framed)


def _zoom_around_face(image: np.ndarray, bbox: tuple[int, int, int, int], pad: float = 2.4) -> tuple[np.ndarray, int, int]:
    x1, y1, x2, y2 = bbox
    height, width = image.shape[:2]
    bw, bh = x2 - x1, y2 - y1
    gx1 = max(0, int(x1 - pad * bw))
    gy1 = max(0, int(y1 - pad * bh))
    gx2 = min(width, int(x2 + pad * bw))
    gy2 = min(height, int(y2 + pad * bh))
    return image[gy1:gy2, gx1:gx2].copy(), gx1, gy1


def make_solo() -> None:
    image = _load_bgr(TEST / SOLO_FILE)
    x1, y1, x2, y2 = _pixel_bbox(SOLO_FILE)
    zoom, ox, oy = _zoom_around_face(image, (x1, y1, x2, y2))
    lx1, ly1, lx2, ly2 = x1 - ox, y1 - oy, x2 - ox, y2 - oy
    cv2.rectangle(zoom, (lx1, ly1), (lx2, ly2), (0, 220, 80), 4)
    cx, cy = (lx1 + lx2) // 2, (ly1 + ly2) // 2
    gaze = _sidecar_face(SOLO_FILE, "yakhyo_gaze")["gaze"]
    length = 0.9 * (ly2 - ly1)
    yaw = math.radians(float(gaze["yaw"]))
    pitch = math.radians(float(gaze["pitch"]))
    end = (
        int(round(cx + length * math.sin(yaw))),
        int(round(cy - length * math.sin(pitch))),
    )
    cv2.arrowedLine(zoom, (cx, cy), end, (0, 200, 255), 3, tipLength=0.18)
    framed = _caption_bar(
        _resize_max_w(zoom),
        "Solo photo: one face. Green box = found face. Yellow arrow = gaze guess.",
    )
    _save_jpg(OUT / "find_faces_solo.jpg", framed)

    crop = image[max(0, y1 - 20) : min(image.shape[0], y2 + 20), max(0, x1 - 20) : min(image.shape[1], x2 + 20)]
    mask = np.asarray(_sidecar_face(SOLO_FILE, "bisenet")["parsing_mask"], dtype=np.int32)
    color = _PARSE_COLORS[np.clip(mask, 0, len(_PARSE_COLORS) - 1)]
    color_bgr = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)
    color_bgr = cv2.resize(color_bgr, (crop.shape[1], crop.shape[0]), interpolation=cv2.INTER_NEAREST)
    blend = cv2.addWeighted(crop, 0.45, color_bgr, 0.55, 0)
    size = 320
    left = cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)
    mid = cv2.resize(blend, (size, size), interpolation=cv2.INTER_AREA)
    expr = str(_sidecar_face(SOLO_FILE, "opencv_fer").get("emotion_label", "?"))
    race = str(_sidecar_face(SOLO_FILE, "fairface").get("race_label", "?"))
    panel = np.full((size, 340, 3), 24, dtype=np.uint8)
    lines = [
        "Extra guesses on this face",
        f"Expression: {expr}",
        f"Gaze yaw: {float(gaze['yaw']):.1f} deg",
        f"Gaze pitch: {float(gaze['pitch']):.1f} deg",
        f"FairFace: {race}",
        "(guesses, not facts)",
    ]
    y = 36
    for i, line in enumerate(lines):
        scale = 0.62 if i == 0 else 0.55
        cv2.putText(panel, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (240, 240, 240), 1, cv2.LINE_AA)
        y += 42 if i == 0 else 36
    gap = np.full((size, 10, 3), 24, dtype=np.uint8)
    row = np.hstack([left, gap, mid, gap, panel])
    extras = _caption_bar(row, "Same solo face: photo, skin/hair map, Yakhyo gaze + FairFace")
    _save_jpg(OUT / "face_extras_solo.jpg", extras)
    _save_jpg(OUT / "face_parsing.jpg", _caption_bar(np.hstack([left, gap, mid]), "Same face: photo (left) and skin/hair/eye map (right)"))


def make_crop_row() -> None:
    picks = [
        (SOLO_FILE, None),
        ("20100904_163717.960-3.jpg", 4),
        ("20100904_163717.960-3.jpg", 10),
        ("20100911_164552.260.jpg", 4),
        ("20110903_172733.840.jpg", 0),
        ("20100918_120908.480-2.jpg", 0),
    ]
    tiles: list[np.ndarray] = []
    tile = 220
    pad = 8
    for file_name, face_index in picks:
        image = _load_bgr(TEST / file_name)
        if face_index is None:
            x1, y1, x2, y2 = _pixel_bbox(file_name)
            expr = str(_sidecar_face(file_name, "opencv_fer").get("emotion_label", "?"))
        else:
            row = next(r for r in _faces_for(file_name) if int(r["face_index"]) == face_index)
            x1, y1, x2, y2 = (int(round(float(row[k]))) for k in ("x1", "y1", "x2", "y2"))
            expr = row["opencv_expression"]
        bw = max(8, int(0.25 * (x2 - x1)))
        bh = max(8, int(0.25 * (y2 - y1)))
        h, w = image.shape[:2]
        crop = image[max(0, y1 - bh) : min(h, y2 + bh), max(0, x1 - bw) : min(w, x2 + bw)]
        crop = cv2.resize(crop, (tile, tile), interpolation=cv2.INTER_AREA)
        cv2.putText(crop, expr, (8, tile - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(crop, expr, (8, tile - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(crop)
    gap = np.full((tile, pad, 3), 24, dtype=np.uint8)
    row_img = tiles[0]
    for tile_img in tiles[1:]:
        row_img = np.hstack([row_img, gap, tile_img])
    framed = _caption_bar(row_img, "Cut-out faces from test_images (label = expression guess)")
    _save_jpg(OUT / "face_crops.jpg", framed)


def make_detector_chart() -> None:
    with (REPORTS / "photos.csv").open(newline="") as handle:
        photos = [row for row in csv.DictReader(handle) if _is_kept_photo(row["file"])]
    fig, ax = plt.subplots(figsize=(12, 8))
    y = np.arange(len(photos))
    series = [
        (-0.24, "scrfd_640", "SCRFD 640", "#4d7fa6"),
        (0.0, "scrfd_1280", "SCRFD 1280", "#0b956c"),
        (0.24, "dlib_faces", "dlib HOG", "#d89531"),
    ]
    for offset, key, label, color in series:
        ax.barh(y + offset, [int(p[key]) for p in photos], height=0.23, label=label, color=color)
    ax.set_yticks(y, [p["file"] for p in photos])
    ax.invert_yaxis()
    ax.set_xlabel("Face detections (not unique people)")
    ax.set_title("Face finder counts on remaining test photos", loc="left")
    ax.legend(loc="lower right")
    ax.grid(axis="x", alpha=0.2)
    ax.set_axisbelow(True)
    fig.tight_layout()
    tmp = OUT / "_chart.png"
    fig.savefig(tmp, dpi=140)
    plt.close(fig)
    chart = _load_bgr(tmp)
    tmp.unlink()
    _save_jpg(OUT / "detector_comparison.jpg", _resize_max_w(chart, 800))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    make_solo()
    make_overlay(
        "20110903_172733.840.jpg",
        "Find every face in a team photo (green box = easier; orange = very small)",
        "find_faces_team.jpg",
        thickness=3,
    )
    make_overlay(
        "20100904_163717.960-3.jpg",
        "Small group: faces are larger and easier to find",
        "find_faces_group.jpg",
        thickness=4,
    )
    make_overlay(
        "20100918_120908.480-2.jpg",
        "Sports action: some faces are small, turned, or partly hidden",
        "find_faces_action.jpg",
        thickness=3,
    )
    make_overlay(
        "20090912_165430.000.jpg",
        "Crowded photo: many tiny faces. The tool still tries to mark each one.",
        "find_faces_crowd.jpg",
        thickness=2,
    )
    make_crop_row()
    make_detector_chart()
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
