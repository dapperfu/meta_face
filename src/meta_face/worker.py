"""RQ worker startup.

Jobs run in a forked child so a native SIGSEGV/SIGABRT fails that job instead of
dumping core and killing the worker. The parent must not import CUDA/dlib.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import signal
import sys
import tempfile
from pathlib import Path

from rq import Worker
from rq.job import Job
from rq.queue import Queue

from meta_face.config import RQ_CLUSTER_QUEUE_NAME, RQ_QUEUE_NAME, RQ_SCAN_QUEUE_NAME
from meta_face.deps import (
    PipelineDependencyError,
    require_cluster_runtime,
    require_inference_runtime,
)
from meta_face.isolation import crash_message, prepare_isolated_process
from meta_face.queue import get_redis


class IsolatedWorker(Worker):
    """Fork-per-job RQ worker that records native crashes as job failures."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._horse_crash_status: int | None = None

    def main_work_horse(self, job: Job, queue: Queue) -> None:
        prepare_isolated_process()
        super().main_work_horse(job, queue)

    def handle_work_horse_killed(self, job, retpid, ret_val, rusage) -> None:
        self._horse_crash_status = ret_val
        super().handle_work_horse_killed(job, retpid, ret_val, rusage)

    def handle_job_failure(self, job: Job, queue: Queue, started_job_registry=None, exc_string="") -> None:
        if self._horse_crash_status is not None:
            exc_string = crash_message(self._horse_crash_status)
            self._horse_crash_status = None
        super().handle_job_failure(
            job,
            queue,
            started_job_registry=started_job_registry,
            exc_string=exc_string,
        )


def _validate_worker_deps(queue_names: list[str]) -> None:
    if RQ_QUEUE_NAME in queue_names:
        require_inference_runtime()
    if RQ_CLUSTER_QUEUE_NAME in queue_names:
        require_cluster_runtime()


def _dep_check_entry(queue_names: list[str], error_file: str) -> None:
    try:
        _validate_worker_deps(queue_names)
    except PipelineDependencyError as exc:
        Path(error_file).write_text(str(exc), encoding="utf-8")
        raise SystemExit(1) from exc


def _validate_worker_deps_isolated(queue_names: list[str]) -> None:
    """Import CUDA/dlib in a spawn child so the worker parent stays fork-safe."""
    fd, error_file = tempfile.mkstemp(prefix="meta-face-deps-")
    os.close(fd)
    try:
        ctx = mp.get_context("spawn")
        proc = ctx.Process(target=_dep_check_entry, args=(queue_names, error_file))
        proc.start()
        proc.join()
        if proc.exitcode == 0:
            return
        message = Path(error_file).read_text(encoding="utf-8").strip()
        if not message:
            message = f"dependency check exited {proc.exitcode}"
        print(f"meta-face worker: {message}", file=sys.stderr)
        raise SystemExit(1)
    finally:
        Path(error_file).unlink(missing_ok=True)


def _worker_main(queue_names: list[str]) -> None:
    prepare_isolated_process()
    try:
        _validate_worker_deps_isolated(queue_names)
    except PipelineDependencyError as exc:
        print(f"meta-face worker: {exc}", file=sys.stderr)
        sys.exit(1)

    redis_conn = get_redis()
    worker = IsolatedWorker(queue_names, connection=redis_conn)
    worker.work(with_scheduler=False)


def start_workers(workers: int = 1, *, cluster: bool = False) -> None:
    """Start one or more RQ workers (multiprocessing when workers > 1)."""
    queue_names = [RQ_SCAN_QUEUE_NAME, RQ_QUEUE_NAME]
    if cluster:
        queue_names.append(RQ_CLUSTER_QUEUE_NAME)

    if workers <= 1:
        _worker_main(queue_names)
        return

    processes: list[mp.Process] = []

    def _shutdown(signum: int, frame: object) -> None:
        for proc in processes:
            if proc.is_alive():
                proc.terminate()
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    for _ in range(workers):
        proc = mp.Process(target=_worker_main, args=(queue_names,))
        proc.start()
        processes.append(proc)

    for proc in processes:
        proc.join()
