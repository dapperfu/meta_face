"""Pretty-print all face attributes extracted by the annotation pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from meta_face.sidecar import get_face_section, load_or_create, sidecar_path_for_media

_KPS_NAMES = ("left_eye", "right_eye", "nose", "left_mouth", "right_mouth")


def cluster_labels_for_media(media_path: Path) -> list[int] | None:
    """Load HDBSCAN cluster labels from the sibling .scar, if present."""
    scar_path = sidecar_path_for_media(media_path)
    if not scar_path.exists():
        return None
    doc, _ = load_or_create(media_path)
    section = get_face_section(doc, "cluster")
    labels = section.get("labels")
    if not isinstance(labels, list):
        return None
    return [int(x) for x in labels]


def _gender_label(gender: Any) -> str | None:
    if gender is None:
        return None
    return "M" if int(gender) == 1 else "F"


def _sex_label(record: dict[str, Any]) -> str | None:
    sex = record.get("sex")
    if sex:
        return str(sex)
    return _gender_label(record.get("gender"))


def _bbox_size(bbox: list[float]) -> tuple[float, float]:
    x1, y1, x2, y2 = (float(v) for v in bbox[:4])
    return x2 - x1, y2 - y1


def _format_pose(pose: Any) -> str | None:
    if not isinstance(pose, list) or len(pose) < 3:
        return None
    pitch, yaw, roll = float(pose[0]), float(pose[1]), float(pose[2])
    return f"pitch {pitch:+.1f}°, yaw {yaw:+.1f}°, roll {roll:+.1f}°"


def _format_points(name: str, points: Any, *, max_show: int = 5) -> list[str]:
    if not isinstance(points, list) or not points:
        return []
    lines = [f"  {name}: {len(points)} point(s)"]
    for idx, pt in enumerate(points[:max_show]):
        if isinstance(pt, list) and len(pt) >= 2:
            lines.append(f"    [{idx}] ({float(pt[0]):.1f}, {float(pt[1]):.1f})")
        elif isinstance(pt, list) and len(pt) >= 3:
            lines.append(
                f"    [{idx}] ({float(pt[0]):.1f}, {float(pt[1]):.1f}, {float(pt[2]):.1f})"
            )
    if len(points) > max_show:
        lines.append(f"    ... {len(points) - max_show} more")
    return lines


def _format_kps(kps: Any) -> list[str]:
    if not isinstance(kps, list) or not kps:
        return []
    lines = [f"  kps (5-point): {len(kps)} point(s)"]
    for idx, pt in enumerate(kps):
        label = _KPS_NAMES[idx] if idx < len(_KPS_NAMES) else f"pt{idx}"
        if isinstance(pt, list) and len(pt) >= 2:
            lines.append(f"    {label}: ({float(pt[0]):.1f}, {float(pt[1]):.1f})")
    return lines


def compact_face_label(
    record: dict[str, Any],
    *,
    face_index: int,
    cluster_label: int | None = None,
) -> str:
    """One-line summary for crop titles (age, gender, det, pose, cluster)."""
    parts = [f"Face {face_index + 1}"]

    age = record.get("age")
    if age is not None:
        parts.append(f"age {int(age)}")

    sex = _sex_label(record)
    if sex:
        parts.append(sex)

    det = record.get("det_score")
    if det is not None:
        parts.append(f"det {float(det):.2f}")

    pose = record.get("pose")
    if isinstance(pose, list) and len(pose) >= 3:
        pitch, yaw, roll = float(pose[0]), float(pose[1]), float(pose[2])
        parts.append(f"pose {pitch:+.0f}/{yaw:+.0f}/{roll:+.0f}")

    if cluster_label is not None:
        if cluster_label < 0:
            parts.append("cluster noise")
        else:
            parts.append(f"cluster {cluster_label}")

    return " · ".join(parts)


def format_face_attributes(
    record: dict[str, Any],
    *,
    face_index: int,
    cluster_label: int | None = None,
) -> str:
    """Return a multi-line summary of every field in a face annotation record."""
    lines: list[str] = [f"=== Face {face_index + 1} ==="]

    bbox = record.get("bbox")
    if isinstance(bbox, list) and len(bbox) >= 4:
        w, h = _bbox_size(bbox)
        lines.append("Detection")
        lines.append(f"  bbox: {[round(float(v), 1) for v in bbox[:4]]}")
        lines.append(f"  size: {w:.0f} x {h:.0f} px")
    det = record.get("det_score")
    if det is not None:
        lines.append(f"  det_score: {float(det):.4f}")

    lines.append("")
    lines.append("Demographics (InsightFace genderage)")
    age = record.get("age")
    if age is not None:
        lines.append(f"  age: {int(age)}")
    else:
        lines.append("  age: (not in sidecar — re-scan with `mf scan --tools scrfd --force`)")

    gender = record.get("gender")
    if gender is not None:
        lines.append(f"  gender: {int(gender)} ({_gender_label(gender)})")
    else:
        lines.append("  gender: (not available)")

    sex = _sex_label(record)
    if sex:
        lines.append(f"  sex: {sex}")

    lines.append("")
    lines.append("Emotion / mood")
    lines.append("  (not available — buffalo_l has no emotion model)")

    pose_text = _format_pose(record.get("pose"))
    lines.append("")
    lines.append("Head pose")
    lines.append(f"  {pose_text}" if pose_text else "  (not available)")

    lines.append("")
    lines.append("Landmarks")
    kps_lines = _format_kps(record.get("kps"))
    if kps_lines:
        lines.extend(kps_lines)
    else:
        lines.append("  kps: (not available)")

    landmark_keys = sorted(k for k in record if isinstance(k, str) and k.startswith("landmark_"))
    if landmark_keys:
        for key in landmark_keys:
            lines.extend(_format_points(key, record[key], max_show=3))
    else:
        lines.append("  dense landmarks: (not available)")

    lines.append("")
    lines.append("Clustering")
    if cluster_label is not None:
        if cluster_label < 0:
            lines.append("  cluster: noise (HDBSCAN outlier)")
        else:
            lines.append(f"  cluster: {cluster_label}")
    else:
        lines.append("  cluster: (no cluster labels in sidecar)")

    if "embedding" in record:
        emb = record["embedding"]
        if hasattr(emb, "shape"):
            lines.append("")
            lines.append(f"Embedding: shape {tuple(emb.shape)} (omitted)")

    return "\n".join(lines)


def print_all_face_attributes(
    records: list[dict[str, Any]],
    *,
    cluster_labels: list[int] | None = None,
    source: str = "detect",
) -> None:
    """Print every extracted attribute for each face."""
    print(f"Source: {source}")
    print(f"Faces: {len(records)}")
    if source == "sidecar":
        print(
            "Note: re-scan with `mf scan IMAGE --tools scrfd,arcface,dlib_detect,dlib_embed --force` "
            "if fields are missing (pre-1.1.0 sidecars stored partial data)."
        )
    print()

    for idx, record in enumerate(records):
        cluster: int | None = None
        if cluster_labels is not None and idx < len(cluster_labels):
            cluster = cluster_labels[idx]
        print(format_face_attributes(record, face_index=idx, cluster_label=cluster))
        print()
