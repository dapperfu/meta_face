"""Tests for per-embedding FAISS index paths."""

from __future__ import annotations

from meta_face.config import cluster_tool_for_embedding, faiss_index_path, faiss_meta_path


def test_faiss_paths_differ_by_embedding_tool() -> None:
    arcface_index = faiss_index_path("arcface")
    dlib_index = faiss_index_path("dlib_embed")
    assert arcface_index != dlib_index
    assert "arcface" in arcface_index.name
    assert "dlib_embed" in dlib_index.name


def test_faiss_meta_paths_differ_by_embedding_tool() -> None:
    arcface_meta = faiss_meta_path("arcface")
    dlib_meta = faiss_meta_path("dlib_embed")
    assert arcface_meta != dlib_meta


def test_cluster_tool_for_embedding() -> None:
    assert cluster_tool_for_embedding("arcface") == "cluster"
    assert cluster_tool_for_embedding("dlib_embed") == "cluster_dlib"
