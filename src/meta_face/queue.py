"""RQ queue connection and enqueue helpers."""

from __future__ import annotations

from pathlib import Path

from redis import Redis
from rq import Queue

from meta_face.config import (
    RQ_CLUSTER_QUEUE_NAME,
    RQ_IMAGE_ADD_QUEUE_NAME,
    RQ_IMAGE_QUEUE_NAME,
    RQ_JOB_TIMEOUT,
    RQ_QUEUE_NAME,
    RQ_SCAN_QUEUE_NAME,
    REDIS_URL,
    rq_job_timeout,
)
from meta_face.job_ids import job_id_for_path


def get_redis() -> Redis:
    return Redis.from_url(REDIS_URL)


def get_queue(name: str | None = None) -> Queue:
    return Queue(name or RQ_QUEUE_NAME, connection=get_redis(), default_timeout=RQ_JOB_TIMEOUT)


def get_face_queue() -> Queue:
    return get_queue(RQ_QUEUE_NAME)


def get_cluster_queue() -> Queue:
    return get_queue(RQ_CLUSTER_QUEUE_NAME)


def get_scan_queue() -> Queue:
    return get_queue(RQ_SCAN_QUEUE_NAME)


def get_image_queue() -> Queue:
    return get_queue(RQ_IMAGE_QUEUE_NAME)


def get_image_add_queue() -> Queue:
    return get_queue(RQ_IMAGE_ADD_QUEUE_NAME)


def failed_job_traceback(job_id: str, *, queue_name: str | None = None) -> str | None:
    """Return the traceback string for a failed job, if available."""
    from rq.job import Job

    queue = get_queue(queue_name)
    job = Job.fetch(job_id, connection=queue.connection)
    latest = job.latest_result()
    if latest and latest.exc_string:
        return latest.exc_string
    return job.exc_info


def iter_failed_jobs(
    queue_name: str | None = None,
    limit: int = 10,
) -> list[tuple[str, str | None]]:
    """Return (job_id, traceback) pairs for failed jobs in a queue."""
    queue = get_queue(queue_name)
    job_ids = queue.failed_job_registry.get_job_ids(0, limit)
    return [(job_id, failed_job_traceback(job_id, queue_name=queue_name)) for job_id in job_ids]


def enqueue_face_tool(
    image_path: Path,
    tool: str,
    force: bool = False,
    depends_on: str | None = None,
) -> str:
    """Enqueue one face job for a single tool. Returns the job id."""
    enqueue_kwargs: dict[str, object] = {
        "job_id": job_id_for_path(f"image-{tool}", image_path),
        "failure_ttl": 86400,
        "job_timeout": rq_job_timeout(tool),
    }
    if depends_on is not None:
        enqueue_kwargs["depends_on"] = depends_on
    job = get_face_queue().enqueue(
        "meta_face.jobs.process_image",
        str(image_path),
        [tool],
        force,
        **enqueue_kwargs,
    )
    return job.id


def enqueue_cluster(
    root: Path,
    force: bool = False,
    embedding_tool: str = "arcface",
) -> str:
    job = get_cluster_queue().enqueue(
        "meta_face.jobs.run_cluster",
        str(root),
        force,
        embedding_tool,
        job_id=job_id_for_path(f"cluster-{embedding_tool}", root),
        failure_ttl=86400,
    )
    return job.id


def enqueue_scan_path(
    directory: Path,
    tools: list[str],
    force: bool = False,
    recursive: bool = True,
) -> str:
    """Enqueue a per-directory scan job on the scan queue."""
    job = get_scan_queue().enqueue(
        "meta_face.scan_jobs.scan_path",
        str(directory),
        tools,
        force,
        recursive,
        job_id=job_id_for_path("scan", directory),
        failure_ttl=86400,
    )
    return job.id


def enqueue_query_images(
    directory: Path,
    tools: list[str],
    force: bool = False,
) -> str:
    """Enqueue a job that lists images in one directory and queues add-image jobs."""
    job = get_image_queue().enqueue(
        "meta_face.scan_jobs.query_images",
        str(directory),
        tools,
        force,
        job_id=job_id_for_path("images", directory),
        failure_ttl=86400,
    )
    return job.id


def enqueue_add_image(
    image_path: Path,
    tools: list[str],
    force: bool = False,
) -> str:
    """Enqueue a job that fans one image out into individual face-tool jobs."""
    job = get_image_add_queue().enqueue(
        "meta_face.scan_jobs.add_image",
        str(image_path),
        tools,
        force,
        job_id=job_id_for_path("image-add", image_path),
        failure_ttl=86400,
    )
    return job.id
