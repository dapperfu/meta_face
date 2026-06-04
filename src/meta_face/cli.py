"""Click CLI for meta-face."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click

from meta_face import __version__
from meta_face.config import DEFAULT_TOOLS, REDIS_HOST, REDIS_PORT
from meta_face.queue import enqueue_cluster, enqueue_process_image
from meta_face.scanner import scan_directory
from meta_face.sidecar import get_face_section, list_face_tools, sidecar_path_for_media
from meta_face.tools.registry import validate_tools
from meta_face.worker import start_workers


@click.group()
@click.version_option(__version__, prog_name="meta-face")
def main() -> None:
    """Face detection pipeline writing results to sidecar-rs .scar files."""


@main.command()
@click.argument("path", type=click.Path(exists=True, path_type=Path))
@click.option("--force", is_flag=True, help="Re-queue even when face.<tool> data exists.")
@click.option(
    "--tools",
    default=",".join(DEFAULT_TOOLS),
    show_default=True,
    help="Comma-separated tools: scrfd, arcface, hdbscan (alias for cluster).",
)
@click.option("--recursive/--no-recursive", default=True, show_default=True)
def scan(path: Path, force: bool, tools: str, recursive: bool) -> None:
    """Discover images under PATH and enqueue processing jobs."""
    tool_list = validate_tools([t.strip() for t in tools.split(",") if t.strip()])
    to_enqueue, stats, run_cluster = scan_directory(
        path,
        tool_list,
        force=force,
        recursive=recursive,
    )

    from meta_face.scanner import resolve_per_image_tools

    per_image_tools = resolve_per_image_tools(tool_list)

    job_ids: list[str] = []
    for image_path in to_enqueue:
        job_id = enqueue_process_image(image_path, per_image_tools, force=force)
        if job_id:
            job_ids.append(job_id)

    cluster_job_id = None
    if run_cluster:
        cluster_job_id = enqueue_cluster(path.resolve(), force=force)

    click.echo(
        f"Discovered {stats.discovered} images; "
        f"enqueued {stats.enqueued}; skipped {stats.skipped}."
    )
    if cluster_job_id:
        click.echo(f"Cluster job enqueued: {cluster_job_id}")
    if not job_ids and not cluster_job_id:
        click.echo("Nothing to enqueue.")
        sys.exit(0 if stats.discovered else 1)


@main.command()
@click.option("--workers", default=1, show_default=True, help="Number of RQ worker processes.")
@click.option("--cluster/--no-cluster", default=True, show_default=True, help="Also listen on cluster queue.")
def worker(workers: int, cluster: bool) -> None:
    """Start RQ worker(s) connected to Redis."""
    click.echo(f"Starting {workers} worker(s) on redis://{REDIS_HOST}:{REDIS_PORT}/")
    start_workers(workers, cluster=cluster)


@main.command()
@click.argument("path", type=click.Path(exists=True, path_type=Path))
@click.option("--force", is_flag=True, help="Re-run clustering even if face.cluster exists.")
@click.option("--enqueue/--run-now", default=True, show_default=True, help="Enqueue vs run inline.")
def cluster(path: Path, force: bool, enqueue: bool) -> None:
    """Run or enqueue HDBSCAN clustering over ArcFace embeddings."""
    if enqueue:
        job_id = enqueue_cluster(path.resolve(), force=force)
        click.echo(f"Cluster job enqueued: {job_id}")
        return

    from meta_face.jobs import run_cluster

    result = run_cluster(str(path.resolve()), force=force)
    click.echo(json.dumps(result, indent=2))


@main.command("info")
@click.argument("path", type=click.Path(exists=True, path_type=Path))
@click.option("--json", "as_json", is_flag=True, help="Emit JSON instead of human-readable output.")
def info(path: Path, as_json: bool) -> None:
    """Show face.* sidecar data for an image or .scar file."""
    scar_path = sidecar_path_for_media(path)
    if not scar_path.exists():
        click.echo(f"No sidecar found at {scar_path}", err=True)
        sys.exit(1)

    from sidecar_rs import SidecarDocument

    doc = SidecarDocument.from_path(str(scar_path))
    tools = list_face_tools(doc)
    payload = {tool: get_face_section(doc, tool) for tool in tools}

    if as_json:
        click.echo(json.dumps({"sidecar": str(scar_path), "tools": payload}, indent=2))
        return

    click.echo(f"Sidecar: {scar_path}")
    for tool in tools:
        click.echo(f"\n[{tool}]")
        for key, value in get_face_section(doc, tool).items():
            if key in {"embeddings", "faces"} and isinstance(value, list):
                click.echo(f"  {key}: {len(value)} item(s)")
            else:
                click.echo(f"  {key}: {value}")
