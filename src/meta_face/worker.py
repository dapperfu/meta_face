"""RQ worker startup."""

from __future__ import annotations

import multiprocessing as mp
import signal
import sys

from rq import Worker

from meta_face.config import RQ_CLUSTER_QUEUE_NAME, RQ_QUEUE_NAME, REDIS_URL
from meta_face.queue import get_redis


def _worker_main(queue_names: list[str]) -> None:
    redis_conn = get_redis()
    worker = Worker(queue_names, connection=redis_conn)
    worker.work(with_scheduler=False)


def start_workers(workers: int = 1, *, cluster: bool = False) -> None:
    """Start one or more RQ workers (multiprocessing when workers > 1)."""
    queue_names = [RQ_QUEUE_NAME]
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
