"""FAISS index + HDBSCAN clustering across a photo collection."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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


HDBSCAN_MIN_CLUSTER_SIZE = 2


@dataclass
class FaceRef:
    media_path: Path
    face_index: int


@dataclass
class ClusterResult:
    labels: np.ndarray
    probabilities: np.ndarray | None
    outlier_scores: np.ndarray | None


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
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    np.divide(matrix, norms, out=matrix, where=norms > 0)
    return matrix, refs


def build_faiss_index(embeddings: np.ndarray) -> Any:
    import faiss

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index


def run_hdbscan(
    embeddings: np.ndarray,
    *,
    min_cluster_size: int = HDBSCAN_MIN_CLUSTER_SIZE,
    min_samples: int | None = None,
    selection_method: str = "eom",
) -> ClusterResult:
    """Cluster one homogeneous feature matrix without requiring FAISS or a GPU."""
    if min_cluster_size < 2:
        raise ValueError("min_cluster_size must be at least 2")
    if min_samples is not None and min_samples < 1:
        raise ValueError("min_samples must be positive")
    if selection_method not in {"eom", "leaf"}:
        raise ValueError("selection_method must be eom or leaf")
    if embeddings.ndim != 2 or not np.isfinite(embeddings).all():
        raise ValueError("Expected a finite, two-dimensional feature matrix")
    if len(embeddings) < min_cluster_size or len(embeddings) <= (min_samples or 0):
        return ClusterResult(
            labels=np.full(len(embeddings), -1, dtype=np.int64),
            probabilities=np.zeros(len(embeddings)),
            outlier_scores=np.zeros(len(embeddings)),
        )
    import hdbscan

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
        cluster_selection_method=selection_method,
    )
    labels = clusterer.fit_predict(embeddings)
    probabilities = getattr(clusterer, "probabilities_", None)
    outlier_scores = getattr(clusterer, "outlier_scores_", None)
    return ClusterResult(
        labels=labels,
        probabilities=probabilities,
        outlier_scores=outlier_scores,
    )


def save_faiss_artifacts(
    index: Any,
    refs: list[FaceRef],
    *,
    embedding_tool: str = "arcface",
) -> None:
    import faiss

    ensure_data_dir()
    index_path = faiss_index_path(embedding_tool)
    meta_path = faiss_meta_path(embedding_tool)
    faiss.write_index(index, str(index_path))
    meta = [{"path": str(r.media_path), "face_index": r.face_index} for r in refs]
    meta_path.write_text(json.dumps(meta, indent=2))


def write_cluster_labels(
    refs: list[FaceRef],
    cluster: ClusterResult,
    *,
    embedding_tool: str = "arcface",
) -> int:
    """Write all cluster tool outputs back into each image sidecar."""
    cluster_tool = cluster_tool_for_embedding(embedding_tool)
    by_image_labels: dict[Path, dict[int, int]] = {}
    by_image_prob: dict[Path, dict[int, float]] = {}
    by_image_outlier: dict[Path, dict[int, float]] = {}

    for ref_idx, ref in enumerate(refs):
        label = int(cluster.labels[ref_idx])
        by_image_labels.setdefault(ref.media_path, {})[ref.face_index] = label
        if cluster.probabilities is not None:
            by_image_prob.setdefault(ref.media_path, {})[ref.face_index] = float(
                cluster.probabilities[ref_idx]
            )
        if cluster.outlier_scores is not None:
            by_image_outlier.setdefault(ref.media_path, {})[ref.face_index] = float(
                cluster.outlier_scores[ref_idx]
            )

    global_labels = [int(x) for x in cluster.labels]
    num_clusters = len(set(global_labels) - {-1})

    updated = 0
    for media_path, index_labels in by_image_labels.items():
        emb_key = tool_data_key(embedding_tool, "embeddings")
        wrote = False

        def _patch(
            doc: object,
            *,
            labels_map: dict[int, int] = index_labels,
            prob_map: dict[int, float] = by_image_prob.get(media_path, {}),
            outlier_map: dict[int, float] = by_image_outlier.get(media_path, {}),
        ) -> None:
            nonlocal wrote
            if emb_key not in doc:  # type: ignore[operator]
                return
            embeddings = doc[emb_key]  # type: ignore[index]
            if not isinstance(embeddings, list):
                return
            n = len(embeddings)
            cluster_labels = [-1] * n
            probabilities = [0.0] * n
            outlier_scores = [0.0] * n
            for idx, label in labels_map.items():
                if 0 <= idx < n:
                    cluster_labels[idx] = label
            for idx, prob in prob_map.items():
                if 0 <= idx < n:
                    probabilities[idx] = prob
            for idx, score in outlier_map.items():
                if 0 <= idx < n:
                    outlier_scores[idx] = score
            payload: dict[str, Any] = {
                "labels": cluster_labels,
                "num_clusters": len(set(cluster_labels) - {-1}),
                "embedding_tool": embedding_tool,
                "hdbscan_min_cluster_size": HDBSCAN_MIN_CLUSTER_SIZE,
                "collection_num_clusters": num_clusters,
                "collection_face_count": len(global_labels),
            }
            if cluster.probabilities is not None:
                payload["probabilities"] = probabilities
            if cluster.outlier_scores is not None:
                payload["outlier_scores"] = outlier_scores
            write_tool_result(doc, cluster_tool, payload)  # type: ignore[arg-type]
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

    cluster_result = run_hdbscan(embeddings)
    updated = write_cluster_labels(refs, cluster_result, embedding_tool=embedding_tool)

    unique_clusters = len(set(int(x) for x in cluster_result.labels) - {-1})
    return {
        "status": "ok",
        "faces": int(embeddings.shape[0]),
        "embedding_dim": int(embeddings.shape[1]),
        "clusters": unique_clusters,
        "updated_sidecars": updated,
        "faiss_index": str(faiss_index_path(embedding_tool)),
        "embedding_tool": embedding_tool,
    }
