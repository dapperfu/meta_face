"""Click CLI for meta-face."""

from __future__ import annotations

import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Lock

import click
from tqdm import tqdm

from meta_face import __version__
from meta_face.config import (
    DEFAULT_SCAN_META_TOOLS,
    INSIGHTFACE_MODEL,
    REDIS_HOST,
    REDIS_PORT,
    RQ_QUEUE_NAME,
)
from meta_face.config import normalize_embedding_tool
from meta_face.deps import (
    PipelineDependencyError,
    adjust_per_image_tools_for_runtime,
    require_cluster_runtime,
    require_inference_runtime,
    require_insightface_runtime,
)
from meta_face.queue import (
    enqueue_cluster,
    enqueue_process_image,
    enqueue_scan_path,
    failed_job_traceback,
    iter_failed_jobs,
)
from meta_face.scanner import (
    ScanStats,
    resolve_cluster_embedding_tool,
    resolve_per_image_tools,
    run_cluster_requested,
    scan_directory_level,
)
from meta_face.sidecar import get_face_section, list_face_tools, sidecar_path_for_media
from meta_face.tools.registry import validate_tools
from meta_face.worker import start_workers


@click.group()
@click.version_option(__version__, prog_name="meta-face")
def main() -> None:
    """Face detection pipeline writing results to sidecar-rs .scar files."""


def _exit_on_dependency_error(exc: PipelineDependencyError) -> None:
    click.echo(str(exc), err=True)
    sys.exit(1)


@main.command()
@click.argument("path", type=click.Path(exists=True, path_type=Path))
@click.option("--force", is_flag=True, help="Re-queue even when face.<tool> data exists.")
@click.option(
    "--tools",
    default=",".join(DEFAULT_SCAN_META_TOOLS),
    show_default=True,
    help=(
        "Comma-separated tools: insightface (scrfd + arcface), "
        "face_recognition (dlib_detect + dlib_embed), detectron2, "
        "expression/emotion/gaze/au/blendshapes/attributes/parsing/liveness meta-tools, "
        "or individual tools (emotiefflib, opencv_fer, mediapipe_blendshapes, libreface, "
        "openface3, yakhyo_gaze, fairface, bisenet, uniface, deepface, ...). "
        "Default runs insightface, face_recognition, and detectron2 (no clustering)."
    ),
)
@click.option(
    "--embeddings",
    default="arcface",
    show_default=True,
    help="Embedding source for clustering: arcface or dlib_embed.",
)
@click.option("--recursive/--no-recursive", default=True, show_default=True)
@click.option(
    "--enqueue/--run-now",
    default=True,
    show_default=True,
    help="Enqueue jobs for a worker, or run face scanning inline now.",
)
@click.pass_context
def scan(
    ctx: click.Context,
    path: Path,
    force: bool,
    tools: str,
    embeddings: str,
    recursive: bool,
    enqueue: bool,
) -> None:
    """Discover images under PATH and run/enqueue face processing jobs."""
    tool_list = validate_tools([t.strip() for t in tools.split(",") if t.strip()])
    per_image_tools = resolve_per_image_tools(tool_list)
    requested_raw = {t.strip().lower() for t in tools.split(",") if t.strip()}
    analysis_explicit = requested_raw & {
        t for t in tool_list if t not in {"scrfd", "arcface", "dlib_detect", "dlib_embed", "detectron2"}
    }
    detectron2_explicit = (
        ctx.get_parameter_source("tools") == click.core.ParameterSource.COMMANDLINE
        and "detectron2" in requested_raw
    )
    try:
        per_image_tools, runtime_warnings = adjust_per_image_tools_for_runtime(
            per_image_tools,
            detectron2_explicit=detectron2_explicit,
            analysis_explicit=analysis_explicit,
        )
    except PipelineDependencyError as exc:
        _exit_on_dependency_error(exc)
    if "detectron2" not in per_image_tools:
        tool_list = [tool for tool in tool_list if tool != "detectron2"]
    for warning in runtime_warnings:
        click.echo(click.style(warning, fg="yellow"), err=True)
    run_cluster = run_cluster_requested(tool_list)
    embedding_tool = resolve_cluster_embedding_tool(tool_list, embeddings)

    if not enqueue:
        _scan_inline(
            path,
            tool_list,
            per_image_tools,
            run_cluster,
            embedding_tool,
            force,
            recursive,
        )
        return

    if per_image_tools:
        try:
            require_inference_runtime(per_image_tools)
        except PipelineDependencyError as exc:
            _exit_on_dependency_error(exc)
    if run_cluster:
        try:
            require_cluster_runtime()
        except PipelineDependencyError as exc:
            _exit_on_dependency_error(exc)

    stats, to_enqueue, subdirs = scan_directory_level(path, tool_list, force=force)

    scan_job_ids: list[str] = []
    if recursive and subdirs:
        for subdir in subdirs:
            scan_job_ids.append(
                enqueue_scan_path(subdir, tool_list, force=force, recursive=recursive)
            )

    backend_job_ids: list[str] = []
    if per_image_tools:
        for image_path in to_enqueue:
            backend_job_ids.extend(
                enqueue_process_image(image_path, per_image_tools, force=force)
            )

    cluster_job_id = None
    if run_cluster:
        cluster_job_id = enqueue_cluster(
            path.resolve(),
            force=force,
            embedding_tool=embedding_tool,
        )

    click.echo(
        f"Queued {len(scan_job_ids)} directory scan job(s), "
        f"{len(backend_job_ids)} backend job(s)."
    )
    if stats.discovered:
        click.echo(
            f"At scan root: discovered {stats.discovered} image(s); "
            f"enqueued {stats.enqueued}; skipped {stats.skipped}."
        )
    if cluster_job_id:
        click.echo(f"Cluster job enqueued: {cluster_job_id}")
    if not scan_job_ids and not backend_job_ids and not cluster_job_id:
        click.echo("Nothing to enqueue.")
        sys.exit(0 if stats.discovered else 1)


def _scan_inline(
    path: Path,
    tool_list: list[str],
    per_image_tools: list[str],
    run_cluster: bool,
    embedding_tool: str,
    force: bool,
    recursive: bool,
) -> None:
    """Run face scanning inline with parallel directory walks."""
    from meta_face.jobs import process_image, run_cluster as run_cluster_job
    from meta_face.scanner import scan_directory_level

    stats = ScanStats()
    stats_lock = Lock()
    processed = 0
    faces_total = 0
    result_lock = Lock()
    pending_dirs = 1
    pending_lock = Lock()
    executor = ThreadPoolExecutor()

    def scan_dir(dir_path: Path) -> None:
        nonlocal processed, faces_total, pending_dirs
        try:
            level_stats, to_enqueue, subdirs = scan_directory_level(
                dir_path,
                tool_list,
                force=force,
            )
            with stats_lock:
                stats.discovered += level_stats.discovered
                stats.enqueued += level_stats.enqueued
                stats.skipped += level_stats.skipped

            child_dirs = subdirs if recursive else []
            if child_dirs:
                with pending_lock:
                    pending_dirs += len(child_dirs)
                for subdir in child_dirs:
                    executor.submit(scan_dir, subdir)

            for image_path in to_enqueue:
                result = process_image(str(image_path), per_image_tools, force=force)
                if result.get("status") == "ok":
                    with result_lock:
                        processed += 1
                        faces_total += int(result.get("face_count", 0))
        except PipelineDependencyError:
            raise
        finally:
            with pending_lock:
                pending_dirs -= 1

    executor.submit(scan_dir, path.resolve())

    cluster_result = None
    try:
        with tqdm(desc="Scanning", unit=" images", file=sys.stderr, dynamic_ncols=True) as bar:
            last_discovered = 0
            while True:
                with pending_lock:
                    done = pending_dirs <= 0
                with stats_lock:
                    current_discovered = stats.discovered
                    current_enqueued = stats.enqueued
                    current_skipped = stats.skipped
                    current_pending = pending_dirs

                if current_discovered > last_discovered:
                    bar.update(current_discovered - last_discovered)
                    last_discovered = current_discovered
                bar.set_postfix(
                    processed=processed,
                    enqueued=current_enqueued,
                    skipped=current_skipped,
                    pending=current_pending,
                )
                bar.refresh()
                if done:
                    break
                time.sleep(0.1)

        if run_cluster:
            cluster_result = run_cluster_job(
                str(path.resolve()),
                force=force,
                embedding_tool=embedding_tool,
            )
    except PipelineDependencyError as exc:
        _exit_on_dependency_error(exc)
    finally:
        executor.shutdown(wait=True)

    click.echo(
        f"Discovered {stats.discovered} images; "
        f"processed {processed}; skipped {stats.skipped}; "
        f"detected {faces_total} face(s)."
    )
    if cluster_result is not None:
        click.echo(f"Cluster result: {json.dumps(cluster_result, indent=2)}")
    if processed == 0 and cluster_result is None:
        click.echo("Nothing to process.")
        sys.exit(0 if stats.discovered else 1)


@main.command("backends")
def backends_cmd() -> None:
    """List face detection backends and availability."""
    from meta_face.backends.registry import list_detection_backends

    click.echo("Detection backends:")
    for backend in list_detection_backends():
        status = "available" if backend.available() else "unavailable"
        click.echo(f"  {backend.name}: {status}")


@main.command("tools")
def tools_cmd() -> None:
    """List all registered face tools and runtime availability."""
    from meta_face.config import AGGREGATE_TOOLS, ANALYSIS_TOOLS, DETECTION_TOOLS, TOOL_GROUPS
    from meta_face.tools.analysis.registry import list_analysis_tools, tool_availability

    click.echo("Detection tools:")
    for name in sorted(DETECTION_TOOLS):
        click.echo(f"  {name}")
    click.echo("\nAnalysis tools (require scrfd crops):")
    for name in list_analysis_tools():
        issue = tool_availability(name)
        status = "available" if issue is None else "unavailable"
        click.echo(f"  {name}: {status}")
        if issue:
            click.echo(f"    {issue}")
    click.echo("\nAggregate tools:")
    for name in sorted(AGGREGATE_TOOLS):
        click.echo(f"  {name}")
    click.echo("\nMeta-tool groups:")
    for group, members in sorted(TOOL_GROUPS.items()):
        click.echo(f"  {group}: {', '.join(members)}")


@main.command("download")
@click.option(
    "--backend",
    type=click.Choice(
        [
            "insightface",
            "dlib",
            "detectron2",
            "opencv_fer",
            "fer_plus",
            "mediapipe",
            "fairface",
            "bisenet",
            "yakhyo_gaze",
            "face_antispoof_onnx",
            "analysis",
            "all",
        ],
        case_sensitive=False,
    ),
    default="all",
    show_default=True,
    help="Which backend model weights to download or verify.",
)
@click.option(
    "--model",
    default=INSIGHTFACE_MODEL,
    show_default=True,
    help="insightface model pack to download (SCRFD + ArcFace).",
)
@click.option("--force", is_flag=True, help="Re-download even if the model pack is present.")
def download(backend: str, model: str, force: bool) -> None:
    """Download face model weights ahead of running detection/embedding."""
    from meta_face.models import download as download_insightface
    from meta_face.models import (
        download_all,
        download_detectron2_weights,
        download_dlib_models,
        is_available,
        is_detectron2_available,
        is_dlib_available,
    )
    from meta_face.models import model_dir

    key = backend.lower()
    if key in {"analysis", "opencv_fer", "fer_plus", "mediapipe", "fairface", "bisenet", "yakhyo_gaze", "face_antispoof_onnx"}:
        from meta_face.analysis_models import download_all_analysis_models, download_analysis_model

        if key == "analysis":
            click.echo("Downloading/verifying analysis tool models...")
            paths = download_all_analysis_models(force=force)
            for name, path in paths.items():
                click.echo(f"  {name}: {path}")
            return
        click.echo(f"Downloading {key} model...")
        path = download_analysis_model(key, force=force)
        click.echo(f"{key} model ready at {path}")
        return

    if key == "all":
        from meta_face.analysis_models import download_all_analysis_models

        click.echo("Downloading/verifying all backend models...")
        paths = download_all(insightface_model=model, force=force)
        for name, path in paths.items():
            click.echo(f"  {name}: {path}")
        try:
            analysis_paths = download_all_analysis_models(force=force)
            for name, path in analysis_paths.items():
                click.echo(f"  analysis/{name}: {path}")
        except RuntimeError as exc:
            click.echo(click.style(f"Some analysis models failed: {exc}", fg="yellow"), err=True)
        return

    if key == "dlib":
        if is_dlib_available() and not force:
            from meta_face.models import dlib_model_dir

            click.echo(f"dlib models already available (see {dlib_model_dir()})")
            return
        click.echo("Verifying dlib / face_recognition models...")
        path = download_dlib_models(force=force)
        click.echo(f"dlib models ready at {path}")
        return

    if key == "detectron2":
        from meta_face.config import DETECTRON2_MODEL_ZOO
        from meta_face.detectron2_model import cached_model_zoo_weights_path, resolve_detectron2_model

        if is_detectron2_available() and not force:
            paths = resolve_detectron2_model()
            click.echo("Detectron2 models ready:")
            if paths.model_zoo:
                click.echo(f"  model_zoo: {paths.model_zoo}")
            click.echo(f"  config:  {paths.config}")
            click.echo(f"  weights: {paths.weights}")
            return
        click.echo(f"Downloading Detectron2 model zoo weights ({DETECTRON2_MODEL_ZOO})...")
        path = download_detectron2_weights(force=force)
        click.echo(f"Detectron2 weights cached at {path}")
        if path != cached_model_zoo_weights_path():
            click.echo(f"  (model zoo cache: {cached_model_zoo_weights_path()})")
        return

    if is_available(model) and not force:
        click.echo(f"Model '{model}' already present at {model_dir(model)}")
        return
    click.echo(f"Downloading model '{model}'...")
    path = download_insightface(model, force=force)
    click.echo(f"Model '{model}' ready at {path}")


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
@click.option(
    "--embeddings",
    default="arcface",
    show_default=True,
    help="Embedding source: arcface (512-d) or dlib_embed (128-d).",
)
@click.option("--enqueue/--run-now", default=True, show_default=True, help="Enqueue vs run inline.")
def cluster(path: Path, force: bool, embeddings: str, enqueue: bool) -> None:
    """Run or enqueue HDBSCAN clustering over face embeddings."""
    embedding_tool = normalize_embedding_tool(embeddings)
    if enqueue:
        try:
            require_cluster_runtime()
        except PipelineDependencyError as exc:
            _exit_on_dependency_error(exc)
        job_id = enqueue_cluster(
            path.resolve(),
            force=force,
            embedding_tool=embedding_tool,
        )
        click.echo(f"Cluster job enqueued: {job_id}")
        return

    from meta_face.jobs import run_cluster

    try:
        result = run_cluster(
            str(path.resolve()),
            force=force,
            embedding_tool=embedding_tool,
        )
    except PipelineDependencyError as exc:
        _exit_on_dependency_error(exc)
    click.echo(json.dumps(result, indent=2))


@main.command()
@click.option(
    "--queue",
    "queue_name",
    default=RQ_QUEUE_NAME,
    show_default=True,
    help="RQ queue to inspect.",
)
@click.option("--limit", default=10, show_default=True, help="Max failed jobs to list.")
@click.argument("job_id", required=False)
def failed(queue_name: str, limit: int, job_id: str | None) -> None:
    """Show tracebacks for failed RQ jobs."""
    if job_id:
        traceback = failed_job_traceback(job_id, queue_name=queue_name)
        if not traceback:
            click.echo(f"No traceback stored for job {job_id}", err=True)
            sys.exit(1)
        click.echo(traceback)
        return

    rows = iter_failed_jobs(queue_name, limit=limit)
    if not rows:
        click.echo(f"No failed jobs in queue '{queue_name}'.")
        return

    for idx, (failed_id, traceback) in enumerate(rows):
        if idx:
            click.echo("\n" + ("=" * 72) + "\n")
        click.echo(f"Job: {failed_id}")
        click.echo(traceback or "(no traceback stored)")


@main.command()
@click.argument("path", type=click.Path(exists=True, path_type=Path))
@click.option("--recursive/--no-recursive", default=True, show_default=True)
@click.option("--force", is_flag=True, help="Overwrite existing *_scrfd.* outputs.")
@click.option(
    "--no-dense-landmarks",
    is_flag=True,
    help="Skip 106-point landmark dots.",
)
def annotate(path: Path, recursive: bool, force: bool, no_dense_landmarks: bool) -> None:
    """Draw face overlays to sibling *_scrfd.* images (bbox, landmarks, pose, attributes)."""
    from meta_face.annotate import AnnotateStats, annotate_image, iter_annotate_paths

    try:
        require_insightface_runtime()
    except PipelineDependencyError as exc:
        _exit_on_dependency_error(exc)

    dense_landmarks = not no_dense_landmarks
    paths = iter_annotate_paths(path, recursive=recursive)
    if not paths:
        click.echo(f"No images found at {path}", err=True)
        sys.exit(1)

    if len(paths) == 1 and paths[0] == path.resolve():
        try:
            out = annotate_image(paths[0], force=force, dense_landmarks=dense_landmarks)
        except Exception as exc:
            click.echo(f"Annotate failed: {exc}", err=True)
            sys.exit(1)
        if out is None:
            click.echo(f"Skipped (exists): {paths[0]}")
            return
        click.echo(out)
        return

    from meta_face.annotate import annotated_output_exists

    stats = AnnotateStats(discovered=len(paths))
    with tqdm(paths, desc="Annotate", unit=" images", file=sys.stderr, dynamic_ncols=True) as bar:
        for media_path in bar:
            try:
                if annotated_output_exists(media_path) and not force:
                    stats.skipped += 1
                    continue
                result = annotate_image(
                    media_path,
                    force=force,
                    dense_landmarks=dense_landmarks,
                )
                if result is None:
                    stats.skipped += 1
                else:
                    stats.written += 1
            except Exception:
                stats.errors += 1

    click.echo(
        f"Discovered {stats.discovered} images; "
        f"wrote {stats.written}, skipped {stats.skipped}, errors {stats.errors}."
    )
    if stats.errors:
        sys.exit(1)


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
