"""Tests for RQ worker startup."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from rq import Worker

from meta_face.worker import IsolatedWorker, _worker_main


def test_isolated_worker_is_fork_worker() -> None:
    assert issubclass(IsolatedWorker, Worker)


def test_worker_uses_isolated_fork_worker() -> None:
    mock_conn = MagicMock()
    with (
        patch("meta_face.worker.prepare_isolated_process"),
        patch("meta_face.worker._validate_worker_deps_isolated"),
        patch("meta_face.worker.get_redis", return_value=mock_conn),
        patch("meta_face.worker.IsolatedWorker") as mock_worker,
    ):
        mock_worker.return_value = MagicMock()
        _worker_main(["meta-face"])

    mock_worker.assert_called_once_with(["meta-face"], connection=mock_conn)
    mock_worker.return_value.work.assert_called_once_with(with_scheduler=False)


def test_isolated_worker_records_killed_horse_status() -> None:
    worker = IsolatedWorker.__new__(IsolatedWorker)
    worker._horse_crash_status = None
    with patch.object(Worker, "handle_work_horse_killed"):
        IsolatedWorker.handle_work_horse_killed(worker, MagicMock(), 1, 11, None)
    assert worker._horse_crash_status == 11


def test_isolated_worker_rewrites_signal_death_as_job_failure() -> None:
    worker = IsolatedWorker.__new__(IsolatedWorker)
    worker._horse_crash_status = 11  # SIGSEGV wait status
    recorded: dict[str, str] = {}

    def fake_handle(self, job, queue, started_job_registry=None, exc_string=""):
        recorded["exc_string"] = exc_string

    with patch.object(Worker, "handle_job_failure", fake_handle):
        IsolatedWorker.handle_job_failure(worker, job=MagicMock(), queue=MagicMock())

    assert "SIGSEGV" in recorded["exc_string"]
    assert worker._horse_crash_status is None


def test_isolated_worker_keeps_python_failure_message() -> None:
    worker = IsolatedWorker.__new__(IsolatedWorker)
    worker._horse_crash_status = None
    recorded: dict[str, str] = {}

    def fake_handle(self, job, queue, started_job_registry=None, exc_string=""):
        recorded["exc_string"] = exc_string

    with patch.object(Worker, "handle_job_failure", fake_handle):
        IsolatedWorker.handle_job_failure(
            worker,
            job=MagicMock(),
            queue=MagicMock(),
            exc_string="RuntimeError: uniface failed",
        )

    assert recorded["exc_string"] == "RuntimeError: uniface failed"
