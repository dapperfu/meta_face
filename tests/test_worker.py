"""Tests for RQ worker startup."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from meta_face.worker import _worker_main


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
