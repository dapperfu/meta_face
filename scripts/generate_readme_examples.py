"""Build compact README example images from test_images and reports."""

from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TEST = ROOT / "test_images"
REPORTS = ROOT / "reports" / "test_images"
OUT = ROOT / "docs" / "readme_examples"
MAX_W = 800

# CelebAMask-HQ / BiSeNet 19 classes (skin through cloth).
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


def _faces_for(file_name: str) -> list[dict[str, str]]:
    with (REPORTS / "faces.csv").open(newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row["file"] == file_name]
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


def make_crop_row() -> None:
    picks = [
        ("20260509_102946.570.jpg", 0),
        ("20260509_104151.150.jpg", 0),
        ("20260425_083120.350.jpg", 0),
        ("20100904_163717.960-3.jpg", 0),
        ("20110903_172733.840.jpg", 0),
        ("20100918_120908.480-2.jpg", 0),
    ]
    tiles: list[np.ndarray] = []
    tile = 220
    pad = 8
    for file_name, face_index in picks:
        row = next(r for r in _faces_for(file_name) if int(r["face_index"]) == face_index)
        image = _load_bgr(TEST / file_name)
        x1, y1, x2, y2 = (int(round(float(row[k]))) for k in ("x1", "y1", "x2", "y2"))
        bw = max(8, int(0.25 * (x2 - x1)))
        bh = max(8, int(0.25 * (y2 - y1)))
        h, w = image.shape[:2]
        crop = image[max(0, y1 - bh) : min(h, y2 + bh), max(0, x1 - bw) : min(w, x2 + bw)]
        crop = cv2.resize(crop, (tile, tile), interpolation=cv2.INTER_AREA)
        expr = row["opencv_expression"]
        cv2.putText(crop, expr, (8, tile - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(crop, expr, (8, tile - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(crop)
    gap = np.full((tile, pad, 3), 24, dtype=np.uint8)
    row_img = tiles[0]
    for tile_img in tiles[1:]:
        row_img = np.hstack([row_img, gap, tile_img])
    framed = _caption_bar(row_img, "Cut-out faces from test_images (label = expression guess)")
    _save_jpg(OUT / "face_crops.jpg", framed)


def make_parsing() -> None:
    file_name = "20260509_102946.570.jpg"
    row = _faces_for(file_name)[0]
    image = _load_bgr(TEST / file_name)
    x1, y1, x2, y2 = (int(round(float(row[k]))) for k in ("x1", "y1", "x2", "y2"))
    bw = int(0.2 * (x2 - x1))
    bh = int(0.2 * (y2 - y1))
    h, w = image.shape[:2]
    crop = image[max(0, y1 - bh) : min(h, y2 + bh), max(0, x1 - bw) : min(w, x2 + bw)]
    mask_path = REPORTS / "masks" / "20260509_102946.570_face_000.png"
    mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise FileNotFoundError(mask_path)
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    color = _PARSE_COLORS[np.clip(mask, 0, len(_PARSE_COLORS) - 1)]
    color_bgr = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)
    color_bgr = cv2.resize(color_bgr, (crop.shape[1], crop.shape[0]), interpolation=cv2.INTER_NEAREST)
    blend = cv2.addWeighted(crop, 0.45, color_bgr, 0.55, 0)
    size = 360
    left = cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)
    right = cv2.resize(blend, (size, size), interpolation=cv2.INTER_AREA)
    gap = np.full((size, 12, 3), 24, dtype=np.uint8)
    pair = np.hstack([left, gap, right])
    framed = _caption_bar(pair, "Same face: photo (left) and skin/hair/eye map (right)")
    _save_jpg(OUT / "face_parsing.jpg", framed)


def copy_detector_chart() -> None:
    src = _load_bgr(REPORTS / "detector_comparison.png")
    _save_jpg(OUT / "detector_comparison.jpg", _resize_max_w(src, 800))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    make_overlay(
        "20110903_172733.840.jpg",
        "Find every face in a team photo (green box = easier; orange = very small)",
        "find_faces_team.jpg",
        thickness=3,
    )
    make_overlay(
        "20260509_102946.570.jpg",
        "Close photo: one clear face is easy to find",
        "find_faces_portrait.jpg",
        thickness=8,
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
    make_parsing()
    copy_detector_chart()
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
