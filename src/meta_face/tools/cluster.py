"""FAISS index + HDBSCAN clustering across a photo collection."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import faiss
import hdbscan
import numpy as np

from meta_face.config import (
    FAISS_INDEX_PATH,
    FAISS_META_PATH,
    ensure_data_dir,
    tool_data_key,
)
from meta_face.imaging import is_image_path
from meta_face.sidecar import has_tool, load_or_create, save, write_tool_result
from meta_face.tools.registry import normalize_tool_name


@dataclass
class FaceRef:
    media_path: Path
    face_index: int


def _iter_sidecar_images(root: Path, recursive: bool = True) -> list[Path]:
    if root.is_file() and is_image_path(root):
        return [root]
    if not root.is_dir():
        return []
    if recursive:
        return sorted(p for p in root.rglob("*") if p.is_file() and is_image_path(p))
    return sorted(p for p in root.iterdir() if p.is_file() and is_image_path(p))


def collect_embeddings(root: Path, *, recursive: bool = True) -> tuple[np.ndarray, list[FaceRef]]:
    refs: list[FaceRef] = []
    vectors: list[list[float]] = []

    for media_path in _iter_sidecar_images(root, recursive=recursive):
        doc, _ = load_or_create(media_path)
        if not has_tool(doc, "arcface"):
            continue
        emb_key = tool_data_key("arcface", "embeddings")
        if emb_key not in doc:
            continue
        embeddings = doc[emb_key]
        if not isinstance(embeddings, list):
            continue
        for idx, emb in enumerate(embeddings):
            if not isinstance(emb, list):
                continue
            vectors.append([float(x) for x in emb])
            refs.append(FaceRef(media_path=media_path, face_index=idx))

    if not vectors:
        return np.array([], dtype=np.float32).reshape(0, 0), refs

    matrix = np.array(vectors, dtype=np.float32)
    faiss.normalize_L2(matrix)
    return matrix, refs


def build_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index


def run_hdbscan(embeddings: np.ndarray) -> np.ndarray:
    if embeddings.shape[0] == 0:
        return np.array([], dtype=np.int64)
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=2,
        metric="euclidean",
        cluster_selection_method="eom",
    )
    return clusterer.fit_predict(embeddings)


def save_faiss_artifacts(index: faiss.Index, refs: list[FaceRef]) -> None:
    ensure_data_dir()
    faiss.write_index(index, str(FAISS_INDEX_PATH))
    meta = [{"path": str(r.media_path), "face_index": r.face_index} for r in refs]
    FAISS_META_PATH.write_text(json.dumps(meta, indent=2))


def write_cluster_labels(refs: list[FaceRef], labels: np.ndarray) -> int:
    """Write face.cluster.labels back into each image sidecar."""
    by_image: dict[Path, dict[int, int]] = {}
    for ref, label in zip(refs, labels, strict=True):
        by_image.setdefault(ref.media_path, {})[ref.face_index] = int(label)

    updated = 0
    for media_path, index_labels in by_image.items():
        doc, scar_path = load_or_create(media_path)
        emb_key = tool_data_key("arcface", "embeddings")
        if emb_key not in doc:
            continue
        embeddings = doc[emb_key]
        if not isinstance(embeddings, list):
            continue
        cluster_labels = [-1] * len(embeddings)
        for idx, label in index_labels.items():
            if 0 <= idx < len(cluster_labels):
                cluster_labels[idx] = label
        write_tool_result(
            doc,
            "cluster",
            {"labels": cluster_labels, "num_clusters": len(set(cluster_labels) - {-1})},
        )
        save(doc, scar_path)
        updated += 1
    return updated


def run_cluster_pipeline(root: Path, *, force: bool = False, recursive: bool = True) -> dict[str, Any]:
    root = root.resolve()
    if not force:
        sample = _iter_sidecar_images(root, recursive=recursive)
        if sample:
            doc, _ = load_or_create(sample[0])
            if has_tool(doc, "cluster") and normalize_tool_name("cluster") == "cluster":
                pass  # still re-run collection level; force controls skip at scan time only

    embeddings, refs = collect_embeddings(root, recursive=recursive)
    if embeddings.shape[0] == 0:
        return {"status": "no_embeddings", "faces": 0, "updated_sidecars": 0}

    index = build_faiss_index(embeddings.copy())
    save_faiss_artifacts(index, refs)

    labels = run_hdbscan(embeddings)
    updated = write_cluster_labels(refs, labels)

    unique_clusters = len(set(int(x) for x in labels) - {-1})
    return {
        "status": "ok",
        "faces": int(embeddings.shape[0]),
        "embedding_dim": int(embeddings.shape[1]),
        "clusters": unique_clusters,
        "updated_sidecars": updated,
        "faiss_index": str(FAISS_INDEX_PATH),
    }
