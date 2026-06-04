"""RQ queue connection and enqueue helpers."""

from __future__ import annotations

from pathlib import Path

from redis import Redis
from rq import Queue

from meta_face.config import (
    RQ_CLUSTER_QUEUE_NAME,
    RQ_JOB_TIMEOUT,
    RQ_QUEUE_NAME,
    REDIS_URL,
)


def get_redis() -> Redis:
    return Redis.from_url(REDIS_URL)


def get_queue(name: str | None = None) -> Queue:
    return Queue(name or RQ_QUEUE_NAME, connection=get_redis(), default_timeout=RQ_JOB_TIMEOUT)


def get_cluster_queue() -> Queue:
    return get_queue(RQ_CLUSTER_QUEUE_NAME)


def enqueue_process_image(
    image_path: Path,
    tools: list[str],
    force: bool = False,
) -> str | None:
    """Enqueue a per-image processing job. Returns job id."""
    from meta_face.jobs import job_id_for_path, process_image

    queue = get_queue()
    job = queue.enqueue(
        process_image,
        str(image_path),
        tools,
        force,
        job_id=job_id_for_path("image", image_path),
        failure_ttl=86400,
    )
    return job.id


def enqueue_cluster(root: Path, force: bool = False) -> str:
    from meta_face.jobs import job_id_for_path, run_cluster

    queue = get_cluster_queue()
    job = queue.enqueue(
        run_cluster,
        str(root),
        force,
        job_id=job_id_for_path("cluster", root),
        failure_ttl=86400,
    )
    return job.id
