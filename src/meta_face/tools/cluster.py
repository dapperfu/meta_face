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
    cluster_tool_for_embedding,
    ensure_data_dir,
    faiss_index_path,
    faiss_meta_path,
    tool_data_key,
)
from meta_face.imaging import is_image_path
from meta_face.sidecar import load_or_create, update_sidecar, write_tool_result


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


def collect_embeddings(
    root: Path,
    *,
    embedding_tool: str = "arcface",
    recursive: bool = True,
) -> tuple[np.ndarray, list[FaceRef]]:
    refs: list[FaceRef] = []
    vectors: list[list[float]] = []

    for media_path in _iter_sidecar_images(root, recursive=recursive):
        doc, _ = load_or_create(media_path)
        emb_key = tool_data_key(embedding_tool, "embeddings")
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


def save_faiss_artifacts(
    index: faiss.Index,
    refs: list[FaceRef],
    *,
    embedding_tool: str = "arcface",
) -> None:
    ensure_data_dir()
    index_path = faiss_index_path(embedding_tool)
    meta_path = faiss_meta_path(embedding_tool)
    faiss.write_index(index, str(index_path))
    meta = [{"path": str(r.media_path), "face_index": r.face_index} for r in refs]
    meta_path.write_text(json.dumps(meta, indent=2))


def write_cluster_labels(
    refs: list[FaceRef],
    labels: np.ndarray,
    *,
    embedding_tool: str = "arcface",
) -> int:
    """Write cluster labels back into each image sidecar."""
    cluster_tool = cluster_tool_for_embedding(embedding_tool)
    by_image: dict[Path, dict[int, int]] = {}
    for ref, label in zip(refs, labels, strict=True):
        by_image.setdefault(ref.media_path, {})[ref.face_index] = int(label)

    updated = 0
    for media_path, index_labels in by_image.items():
        emb_key = tool_data_key(embedding_tool, "embeddings")
        wrote = False

        def _patch(doc: object, *, labels: dict[int, int] = index_labels) -> None:
            nonlocal wrote
            if emb_key not in doc:  # type: ignore[operator]
                return
            embeddings = doc[emb_key]  # type: ignore[index]
            if not isinstance(embeddings, list):
                return
            cluster_labels = [-1] * len(embeddings)
            for idx, label in labels.items():
                if 0 <= idx < len(cluster_labels):
                    cluster_labels[idx] = label
            write_tool_result(
                doc,  # type: ignore[arg-type]
                cluster_tool,
                {
                    "labels": cluster_labels,
                    "num_clusters": len(set(cluster_labels) - {-1}),
                },
            )
            wrote = True

        update_sidecar(media_path, _patch)
        if wrote:
            updated += 1
    return updated


def run_cluster_pipeline(
    root: Path,
    *,
    force: bool = False,
    embedding_tool: str = "arcface",
    recursive: bool = True,
) -> dict[str, Any]:
    root = root.resolve()
    _ = force  # reserved for future skip logic at pipeline level

    embeddings, refs = collect_embeddings(root, embedding_tool=embedding_tool, recursive=recursive)
    if embeddings.shape[0] == 0:
        return {
            "status": "no_embeddings",
            "faces": 0,
            "updated_sidecars": 0,
            "embedding_tool": embedding_tool,
        }

    index = build_faiss_index(embeddings.copy())
    save_faiss_artifacts(index, refs, embedding_tool=embedding_tool)

    labels = run_hdbscan(embeddings)
    updated = write_cluster_labels(refs, labels, embedding_tool=embedding_tool)

    unique_clusters = len(set(int(x) for x in labels) - {-1})
    return {
        "status": "ok",
        "faces": int(embeddings.shape[0]),
        "embedding_dim": int(embeddings.shape[1]),
        "clusters": unique_clusters,
        "updated_sidecars": updated,
        "faiss_index": str(faiss_index_path(embedding_tool)),
        "embedding_tool": embedding_tool,
    }
