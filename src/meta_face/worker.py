"""RQ worker startup."""

from __future__ import annotations

import multiprocessing as mp
import signal
import sys

from rq import SimpleWorker

from meta_face.config import (
    RQ_CLUSTER_QUEUE_NAME,
    RQ_IMAGE_ADD_QUEUE_NAME,
    RQ_IMAGE_QUEUE_NAME,
    RQ_QUEUE_NAME,
    RQ_SCAN_QUEUE_NAME,
)
from meta_face.deps import (
    PipelineDependencyError,
    require_cluster_runtime,
    require_inference_runtime,
)
from meta_face.queue import get_redis

# Stage name -> Redis queue. Face work shares one queue; each tool is still its own job.
WORKER_STAGES: dict[str, str] = {
    "scan": RQ_SCAN_QUEUE_NAME,
    "images": RQ_IMAGE_QUEUE_NAME,
    "image-add": RQ_IMAGE_ADD_QUEUE_NAME,
    "face": RQ_QUEUE_NAME,
    "cluster": RQ_CLUSTER_QUEUE_NAME,
}
DEFAULT_WORKER_STAGES: tuple[str, ...] = ("scan", "images", "image-add", "face", "cluster")


def resolve_worker_queues(stages: list[str]) -> list[str]:
    """Map stage names to Redis queue names, preserving order."""
    names: list[str] = []
    unknown: list[str] = []
    for stage in stages:
        key = stage.strip().lower()
        if not key:
            continue
        queue_name = WORKER_STAGES.get(key)
        if queue_name is None:
            unknown.append(stage.strip())
            continue
        if queue_name not in names:
            names.append(queue_name)
    if unknown:
        valid = ", ".join(WORKER_STAGES)
        raise ValueError(f"Unknown worker queues: {', '.join(unknown)}. Valid: {valid}")
    if not names:
        raise ValueError("Select at least one worker queue.")
    return names


def _validate_worker_deps(queue_names: list[str]) -> None:
    if RQ_QUEUE_NAME in queue_names:
        require_inference_runtime()
    if RQ_CLUSTER_QUEUE_NAME in queue_names:
        require_cluster_runtime()


def _worker_main(queue_names: list[str]) -> None:
    try:
        _validate_worker_deps(queue_names)
    except PipelineDependencyError as exc:
        print(f"meta-face worker: {exc}", file=sys.stderr)
        sys.exit(1)

    redis_conn = get_redis()
    # SimpleWorker runs jobs in-process. RQ's default fork Worker can break CUDA
    # after native dlib/face_recognition imports in the parent process.
    worker = SimpleWorker(queue_names, connection=redis_conn)
    worker.work(with_scheduler=False)


def start_workers(workers: int = 1, *, queues: list[str] | None = None) -> None:
    """Start one or more RQ workers.

    `queues` is a list of stage names (scan, images, image-add, face, cluster).
    The default listens on every stage.
    """
    selected = list(queues) if queues is not None else list(DEFAULT_WORKER_STAGES)
    queue_names = resolve_worker_queues(selected)

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
