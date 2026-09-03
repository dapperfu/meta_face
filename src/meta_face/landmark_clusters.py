"""Tool-scoped control-point clustering and a local face review gallery.

Groups describe landmark configurations, not identity. No recognition model is
loaded. Analysis tools can reuse explicitly linked detector controls, but each
tool and each control-point layout is clustered independently.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps
from sidecar_rs import SidecarDocument

from meta_face.coordinates import section_records_in_pixels
from meta_face.imaging import is_image_path
from meta_face.sidecar import get_face_section, list_face_tools, sidecar_path_for_media
from meta_face.tools.cluster import run_hdbscan

ALIASES = {"dlib": "dlib_detect", "mediapipe": "mediapipe_blendshapes"}
CONTROL_FIELDS = ("landmark_2d_106", "landmark_3d_478", "landmark_3d_68", "landmarks", "kps")


def _box(value: Any) -> list[float] | None:
    try:
        a = np.asarray(value, dtype=float)
        if a.shape == (4,) and np.isfinite(a).all() and a[2] > a[0] and a[3] > a[1]:
            return a.tolist()
    except (TypeError, ValueError):
        pass
    return None


def control_sets(record: dict[str, Any]) -> dict[str, list[list[float]]]:
    """Preserve every well-formed named control array, including projected depth."""
    names = list(CONTROL_FIELDS) + sorted(
        k for k in record if k.startswith("landmark_") and k not in CONTROL_FIELDS)
    result = {}
    for key in names:
        try:
            points = np.asarray(record.get(key), dtype=float)
        except (TypeError, ValueError):
            continue
        if (points.ndim == 2 and points.shape[0] >= 3 and points.shape[1] in (2, 3)
                and np.isfinite(points).all()):
            result[key] = points.tolist()
    return result


def geometry_features(points: list[list[float]]) -> np.ndarray:
    """Remove translation and isotropic scale; retain orientation and configuration."""
    xy = np.asarray(points, dtype=np.float64)[:, :2]
    centered = xy - xy.mean(axis=0)
    scale = float(np.sqrt(np.mean(np.sum(centered ** 2, axis=1))))
    if not np.isfinite(scale) or scale <= 1e-8:
        raise ValueError("Control points have zero or invalid spatial extent")
    # RMS distance per point, so denser layouts do not inflate distances.
    return (centered / (scale * np.sqrt(len(xy)))).reshape(-1)


def _face_index(record: dict[str, Any], position: int) -> int:
    index = record.get("face_index", position)
    if isinstance(index, bool) or not isinstance(index, int) or index < 0:
        raise ValueError("Invalid face_index")
    return index


def records_from_sections(
    sections: dict[str, dict[str, Any]], image_size: tuple[int, int],
) -> list[dict[str, Any]]:
    """Resolve explicit parent indices by value, never by array position or overlap."""
    records_by_tool: dict[str, list[dict[str, Any]]] = {}
    indexes: dict[str, dict[int, dict[str, Any]]] = {}
    for tool, section in sections.items():
        if not isinstance(section.get("faces"), list):
            continue
        raw_faces = section["faces"]
        if any(not isinstance(f, dict) for f in raw_faces):
            raise ValueError(f"{tool}: faces must contain objects")
        faces = section_records_in_pixels(section, image_size)
        records_by_tool[tool] = faces
        index: dict[int, dict[str, Any]] = {}
        for pos, face in enumerate(faces):
            idx = _face_index(face, pos)
            if idx in index:
                raise ValueError(f"{tool}: duplicate face_index {idx}")
            index[idx] = face
        indexes[tool] = index

    def resolve(tool, face, idx, visited):
        if tool in visited:
            return None, {}, None, "Control-point source cycle"
        controls = control_sets(face)
        box = _box(face.get("bbox"))
        if controls and box:
            return box, controls, tool, None
        parent = sections[tool].get("face_index_source")
        if isinstance(parent, str) and parent != tool and parent in indexes:
            source = indexes[parent].get(idx)
            if source is None:
                return box, controls, tool, f"Missing {parent} face_index {idx}"
            pbox, pcontrols, owner, issue = resolve(parent, source, idx, visited | {tool})
            if box and pbox:
                center = ((pbox[0] + pbox[2]) / 2, (pbox[1] + pbox[3]) / 2)
                if not (box[0] <= center[0] <= box[2] and box[1] <= center[1] <= box[3]):
                    return box, controls, tool, "Linked control points belong to a different image region"
            return box or pbox, controls or pcontrols, tool if controls else owner, issue
        return box, controls, tool, "No usable control points" if not controls else "No valid face box"

    result = []
    for tool, faces in records_by_tool.items():
        for pos, face in enumerate(faces):
            idx = _face_index(face, pos)
            box, controls, owner, error = resolve(tool, face, idx, set())
            if sections[tool].get("entity_type") == "person":
                error = "Person detection; no face control points"
            field = next(iter(controls), None)
            record = dict(tool=tool, face_index=idx, record_index=pos, bbox=box,
                          control_source=owner, inherited_controls=owner != tool,
                          controls=controls, feature_field=field,
                          layout=f"{field}:{len(controls[field])}:xy" if field else None,
                          control_count=len(controls[field]) if field else 0,
                          status="excluded" if error else "pending", reason=error,
                          detection_score=face.get("det_score"))
            if not error:
                try:
                    geometry_features(controls[field])
                except ValueError as exc:
                    record.update(status="excluded", reason=str(exc))
            result.append(record)
    return result


def load_faces(root: Path, tools: list[str] | None = None, recursive: bool = True):
    root = root.expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(root)
    paths = ([root] if root.is_file() else
             sorted(p for p in (root.rglob("*") if recursive else root.iterdir())
                    if p.is_file() and is_image_path(p)))
    selected = {ALIASES.get(t, t) for t in tools} if tools else None
    faces, issues, inventory = [], [], []
    for path in paths:
        if not is_image_path(path):
            continue
        scar = sidecar_path_for_media(path)
        if not scar.exists():
            issues.append(dict(image=str(path), reason="Missing sidecar"))
            continue
        try:
            with Image.open(path) as image:
                size = ImageOps.exif_transpose(image).size
            doc = SidecarDocument.from_path(str(scar))
            sections = {t: get_face_section(doc, t) for t in list_face_tools(doc)}
            # Parent detector records remain available even when only a child tool is selected.
            records = records_from_sections(sections, size)
            inventory.append(dict(image=str(path), image_size=list(size),
                                  image_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                                  sidecar_sha256=hashlib.sha256(scar.read_bytes()).hexdigest()))
            for record in records:
                if selected is not None and record["tool"] not in selected:
                    continue
                uid = hashlib.sha256(f"{path}\0{record['tool']}\0{record['face_index']}".encode()).hexdigest()[:20]
                faces.append(dict(record, id=uid, image=str(path), filename=path.name,
                                  image_size=list(size)))
        except Exception as exc:
            issues.append(dict(image=str(path), reason=f"{type(exc).__name__}: {exc}"))
    if selected:
        absent = selected - {f["tool"] for f in faces}
        issues.extend(dict(tool=t, reason="No face records for requested tool") for t in sorted(absent))
    return faces, issues, inventory


def cluster_faces(faces, *, min_cluster_size=5, min_samples=2, selection_method="leaf"):
    batches = defaultdict(list)
    for face in faces:
        if face["status"] != "excluded":
            batches[(face["tool"], face["layout"])].append(face)
    for (tool, layout), batch in sorted(batches.items()):
        features = np.stack([geometry_features(f["controls"][f["feature_field"]]) for f in batch])
        result = run_hdbscan(features, min_cluster_size=min_cluster_size,
                             min_samples=min_samples, selection_method=selection_method)
        # Stable local numbering by first member, independent of HDBSCAN label numbering.
        labels = sorted(set(int(v) for v in result.labels) - {-1},
                        key=lambda label: next(i for i, v in enumerate(result.labels) if v == label))
        numbering = {v: i for i, v in enumerate(labels)}
        for idx, (face, label) in enumerate(zip(batch, result.labels)):
            number = numbering.get(int(label), -1)
            face.update(status="clustered" if number >= 0 else "unassigned", cluster=number,
                        cluster_id=f"{tool}/{layout}/{number}" if number >= 0 else None,
                        membership=float(result.probabilities[idx]) if result.probabilities is not None else None,
                        outlier_score=(float(result.outlier_scores[idx])
                                       if result.outlier_scores is not None and np.isfinite(result.outlier_scores[idx]) else None))
    return faces


def export_review(root: Path, output: Path, *, tools=None, min_cluster_size=5,
                  min_samples=2, selection_method="leaf", recursive=True):
    root, output = root.expanduser().resolve(), output.expanduser().resolve()
    if root.is_dir() and output.is_relative_to(root):
        raise ValueError("Choose an output directory outside the input photo directory")
    faces, issues, inventory = load_faces(root, tools, recursive)
    cluster_faces(faces, min_cluster_size=min_cluster_size, min_samples=min_samples,
                  selection_method=selection_method)
    output.mkdir(parents=True, exist_ok=True)
    (output / "crops").mkdir(exist_ok=True)
    by_image = defaultdict(list)
    for face in faces:
        by_image[face["image"]].append(face)
    for path, records in by_image.items():
        with Image.open(path) as original:
            image = ImageOps.exif_transpose(original).convert("RGB")
            cached = set()
            for face in records:
                box = face["bbox"]
                if box is None or face["status"] == "excluded":
                    continue
                x1, y1, x2, y2 = box
                pad = .15 * max(x2-x1, y2-y1)
                bounds = [max(0, int(np.floor(x1-pad))), max(0, int(np.floor(y1-pad))),
                          min(image.width, int(np.ceil(x2+pad))), min(image.height, int(np.ceil(y2+pad)))]
                if bounds[2] <= bounds[0] or bounds[3] <= bounds[1]:
                    raise ValueError(f"Empty face crop: {face['id']}")
                crop_id = hashlib.sha256(f"{path}:{bounds}".encode()).hexdigest()[:20]
                name = f"crops/{crop_id}.jpg"
                if name not in cached:
                    crop = ImageOps.contain(image.crop(bounds), (240, 240))
                    crop.save(output / name, quality=88)
                    cached.add(name)
                face.update(crop=name, crop_bbox_pixels=bounds)
    tools_summary = []
    for tool in sorted({f["tool"] for f in faces}):
        records = [f for f in faces if f["tool"] == tool]
        counts = Counter(f["status"] for f in records)
        tools_summary.append(dict(tool=tool, faces=len(records), eligible=len(records)-counts["excluded"],
                                  clustered=counts["clustered"], unassigned=counts["unassigned"],
                                  excluded=counts["excluded"],
                                  clusters=len({f["cluster_id"] for f in records if f.get("cluster_id")}),
                                  layouts=sorted({f["layout"] for f in records if f["layout"]}),
                                  control_sources=sorted({f["control_source"] for f in records if f["controls"]})))
    payload = dict(schema=1, purpose="control-point geometry clustering; not identity matching",
                   created_at=datetime.now(timezone.utc).isoformat(), root=str(root),
                   parameters=dict(algorithm="HDBSCAN", min_cluster_size=min_cluster_size,
                                   min_samples=min_samples, selection_method=selection_method,
                                   normalization="center and isotropic RMS scale; preserve orientation",
                                   metric="euclidean", features="2D control-point configuration"),
                   tools=tools_summary, faces=faces, issues=issues, inventory=inventory)
    (output / "clusters.json").write_text(json.dumps(payload, indent=2, allow_nan=False)+"\n")
    # Gallery needs only the selected point set; complete arrays remain in clusters.json.
    gallery = dict(payload, inventory=[], faces=[])
    for face in faces:
        compact = {k: v for k, v in face.items() if k not in {"controls", "image"}}
        compact["points"] = face["controls"].get(face["feature_field"], [])
        gallery["faces"].append(compact)
    template = (Path(__file__).parent / "templates/landmark_clusters.html").read_text()
    data = json.dumps(gallery, separators=(",", ":"), allow_nan=False).replace("<", "\\u003c")
    (output / "index.html").write_text(template.replace("__CLUSTER_DATA__", data))
    lines = ["# Control-point clusters by tool", "", payload["purpose"], "",
             "Open `index.html` to switch tools and groups, inspect face crops, and toggle control points.", "",
             "| Tool | Faces with controls | Clusters | Unassigned | Excluded | Control source |",
             "|---|---:|---:|---:|---:|---|"]
    lines += [f"| {t['tool']} | {t['eligible']} | {t['clusters']} | {t['unassigned']} | {t['excluded']} | "
              f"{', '.join(t['control_sources'])} |" for t in tools_summary]
    lines += ["", "Each tool/layout has its own HDBSCAN run and cluster namespace. Analysis outputs with an explicit "
              "face_index_source retain that link and reuse the source controls; they are still clustered separately. "
              "Unassigned faces are visible and are not treated as one cluster. Tools without face controls are listed as excluded.",
              "", "All source control arrays and pixel coordinates are in `clusters.json`. Input image and sidecar hashes "
              "are recorded there; this command does not modify source images or sidecars.", "",
              "Parameters: " + json.dumps(payload["parameters"]), ""]
    (output / "README.md").write_text("\n".join(lines))
    return dict(report=str(output / "index.html"), manifest=str(output / "clusters.json"),
                tools=tools_summary, issues=issues)
