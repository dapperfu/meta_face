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


def failed_job_traceback(job_id: str, *, queue_name: str | None = None) -> str | None:
    """Return the traceback string for a failed job, if available."""
    from rq.job import Job

    queue = get_queue(queue_name)
    job = Job.fetch(job_id, connection=queue.connection)
    latest = job.latest_result()
    if latest and latest.exc_string:
        return latest.exc_string
    return job.exc_info


def iter_failed_jobs(queue_name: str | None = None, limit: int = 10) -> list[tuple[str, str | None]]:
    """Return (job_id, traceback) pairs for failed jobs in a queue."""
    queue = get_queue(queue_name)
    job_ids = queue.failed_job_registry.get_job_ids(0, limit)
    return [(job_id, failed_job_traceback(job_id, queue_name=queue_name)) for job_id in job_ids]


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
