"""Tests for RQ worker startup."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from meta_face.worker import (
    DEFAULT_WORKER_STAGES,
    _worker_main,
    resolve_worker_queues,
)


def test_worker_uses_simple_worker_for_cuda_jobs() -> None:
    mock_conn = MagicMock()
    with (
        patch("meta_face.worker._validate_worker_deps"),
        patch("meta_face.worker.get_redis", return_value=mock_conn),
        patch("meta_face.worker.SimpleWorker") as mock_simple_worker,
    ):
        mock_simple_worker.return_value = MagicMock()
        _worker_main(["meta-face"])

    mock_simple_worker.assert_called_once_with(["meta-face"], connection=mock_conn)
    mock_simple_worker.return_value.work.assert_called_once_with(with_scheduler=False)


def test_resolve_worker_queues_selects_one_stage() -> None:
    assert resolve_worker_queues(["scan"]) == ["meta-face-scan"]
    assert resolve_worker_queues(["face"]) == ["meta-face"]
    assert resolve_worker_queues(["cluster"]) == ["meta-face-cluster"]


def test_default_worker_queues_cover_pipeline() -> None:
    assert resolve_worker_queues(list(DEFAULT_WORKER_STAGES)) == [
        "meta-face-scan",
        "meta-face-images",
        "meta-face-image-add",
        "meta-face",
        "meta-face-cluster",
    ]


def test_resolve_worker_queues_rejects_unknown_stage() -> None:
    with pytest.raises(ValueError, match="Unknown worker queues"):
        resolve_worker_queues(["scan", "embeddings"])
