"""`mf yolo ...` commands: image, camera, video, benchmark, export."""

from __future__ import annotations

import functools
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar

import click

from meta_face.yolo_face.config import AppConfig, load_config
from meta_face.yolo_face.errors import FaceDetectorError

F = TypeVar("F", bound=Callable[..., Any])


def model_options(func: F) -> F:
    """Options shared by every command that loads a model."""
    options = [
        click.option(
            "--config",
            "config_path",
            type=click.Path(path_type=Path, dir_okay=False),
            help="TOML config (default: config/yolo_face.toml if present).",
        ),
        click.option("--model", type=click.Path(path_type=Path), help="Weights (.pt or .onnx)."),
        click.option("--confidence", type=click.FloatRange(0, 1), help="Confidence threshold."),
        click.option("--iou", type=click.FloatRange(0, 1), help="NMS IoU threshold."),
        click.option("--size", "image_size", type=int, help="Inference size (multiple of 32)."),
        click.option("--device", help="auto, cpu, cuda, cuda:N, or mps."),
        click.option("--half/--no-half", default=None, help="FP16 inference (CUDA only)."),
        click.option("--min-width", type=int, help="Drop faces narrower than this (pixels)."),
        click.option("--min-height", type=int, help="Drop faces shorter than this (pixels)."),
        click.option("-v", "--verbose", is_flag=True, help="Per-frame debug logging."),
    ]
    for option in reversed(options):
        func = option(func)
    return func


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )


def _build_config(
    kwargs: dict[str, Any], extra: dict[str, dict[str, Any]] | None = None
) -> AppConfig:
    overrides: dict[str, dict[str, Any]] = {
        "model": {
            "path": kwargs.pop("model"),
            "confidence": kwargs.pop("confidence"),
            "iou": kwargs.pop("iou"),
            "image_size": kwargs.pop("image_size"),
            "device": kwargs.pop("device"),
            "half": kwargs.pop("half"),
        },
        "filters": {
            "min_face_width": kwargs.pop("min_width"),
            "min_face_height": kwargs.pop("min_height"),
        },
    }
    for section, values in (extra or {}).items():
        overrides.setdefault(section, {}).update(values)
    _setup_logging(kwargs.pop("verbose"))
    return load_config(kwargs.pop("config_path"), overrides)


def _make_detector(config: AppConfig) -> Any:
    from meta_face.yolo_face.detector import FaceDetector

    m = config.model
    return FaceDetector(
        model_path=m.path,
        confidence_threshold=m.confidence,
        iou_threshold=m.iou,
        device=m.device,
        image_size=m.image_size,
        min_face_width=config.filters.min_face_width,
        min_face_height=config.filters.min_face_height,
        half=m.half,
    )


def _handle_errors(func: F) -> F:
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return func(*args, **kwargs)
        except (FaceDetectorError, FileNotFoundError, ValueError) as exc:
            raise click.ClickException(str(exc)) from exc

    return wrapper  # type: ignore[return-value]


def _draw_options(config: AppConfig, show_fps: bool | None = None) -> Any:
    from meta_face.yolo_face.pipeline import DrawOptions

    d = config.display
    return DrawOptions(
        draw_boxes=d.draw_boxes,
        show_confidence=d.show_confidence,
        show_center=d.show_center,
        show_fps=d.show_fps if show_fps is None else show_fps,
    )


@click.group("yolo")
def yolo() -> None:
    """Local YOLOv8-Face detection (bounding boxes only, no recognition)."""


@yolo.command("image")
@click.argument("input_path", type=click.Path(path_type=Path, dir_okay=False))
@model_options
@click.option("--output", "-o", type=click.Path(path_type=Path), help="Save annotated image.")
@click.option("--json", "json_path", type=click.Path(path_type=Path), help="Save detections.")
@click.option("--normalized", is_flag=True, help="Include normalized coordinates in JSON.")
@click.option("--show", is_flag=True, help="Display the annotated image (needs GUI OpenCV).")
@_handle_errors
def image_cmd(
    input_path: Path,
    output: Path | None,
    json_path: Path | None,
    normalized: bool,
    show: bool,
    **kwargs: Any,
) -> None:
    """Detect faces in a single image."""
    import cv2

    from meta_face.yolo_face.drawing import draw_detections
    from meta_face.yolo_face.output import image_result, write_json
    from meta_face.yolo_face.sources import load_image

    config = _build_config(kwargs)
    frame = load_image(input_path)
    detector = _make_detector(config)
    detections = detector.detect(frame)
    height, width = frame.shape[:2]

    click.echo(f"{input_path}: {len(detections)} face(s) in {width}x{height}")
    for i, det in enumerate(detections):
        click.echo(
            f"  [{i}] x1={det.x1} y1={det.y1} x2={det.x2} y2={det.y2} "
            f"({det.width}x{det.height}) conf={det.confidence:.3f}"
        )

    if json_path is not None:
        write_json(json_path, image_result(detections, width, height, normalized=normalized))
        click.echo(f"Wrote {json_path}")
    if output is not None or show:
        d = config.display
        canvas = (
            draw_detections(frame, detections, d.show_confidence, d.show_center)
            if d.draw_boxes
            else frame
        )
        if output is not None:
            output.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(output), canvas):
                raise click.ClickException(f"Could not write image: {output}")
            click.echo(f"Wrote {output}")
        if show:
            window = "meta-face yolo"
            cv2.imshow(window, canvas)
            cv2.waitKey(0)
            cv2.destroyAllWindows()


@yolo.command("camera")
@model_options
@click.option("--camera", "camera_index", type=int, help="Camera index.")
@click.option(
    "--source",
    help="Stream URL instead of a local camera, e.g. rtsp://${CAM_USER}:${CAM_PASS}@host/stream.",
)
@click.option("--width", type=int, help="Requested capture width.")
@click.option("--height", type=int, help="Requested capture height.")
@click.option("--fps", type=float, help="Requested capture frame rate.")
@click.option("--frame-skip", type=click.IntRange(min=0), help="Skip inference on N frames.")
@click.option("--json", "json_path", type=click.Path(path_type=Path), help="JSON Lines output.")
@click.option("--headless", is_flag=True, help="Do not open a window (log/JSON only).")
@_handle_errors
def camera_cmd(
    camera_index: int | None,
    source: str | None,
    width: int | None,
    height: int | None,
    fps: float | None,
    frame_skip: int | None,
    json_path: Path | None,
    headless: bool,
    **kwargs: Any,
) -> None:
    """Live detection from a webcam or network stream. Press q to quit."""
    from meta_face.yolo_face.sources import FrameSource

    config = _build_config(
        kwargs,
        {
            "camera": {"index": camera_index, "width": width, "height": height, "fps": fps},
            "performance": {"frame_skip": frame_skip},
        },
    )
    detector = _make_detector(config)
    cam = config.camera
    spec: int | str = source if source else cam.index
    with FrameSource(spec, width=cam.width, height=cam.height, fps=cam.fps) as frames:
        stats = _run(
            frames, detector, config, json_path, window=None if headless else "meta-face yolo"
        )
    _report(stats)


@yolo.command("video")
@click.argument("input_path")
@model_options
@click.option("--output", "-o", type=click.Path(path_type=Path), help="Annotated output video.")
@click.option("--json", "json_path", type=click.Path(path_type=Path), help="JSON Lines output.")
@click.option("--frame-skip", type=click.IntRange(min=0), help="Skip inference on N frames.")
@click.option("--show", is_flag=True, help="Display frames while processing.")
@click.option("--fourcc", default="mp4v", show_default=True, help="Output codec.")
@_handle_errors
def video_cmd(
    input_path: str,
    output: Path | None,
    json_path: Path | None,
    frame_skip: int | None,
    show: bool,
    fourcc: str,
    **kwargs: Any,
) -> None:
    """Process a video file (or stream URL), keeping source resolution and frame rate."""
    from contextlib import ExitStack

    from meta_face.yolo_face.output import VideoSink
    from meta_face.yolo_face.sources import FrameSource

    config = _build_config(kwargs, {"performance": {"frame_skip": frame_skip}})
    detector = _make_detector(config)
    with ExitStack() as stack:
        frames = stack.enter_context(FrameSource(input_path))
        sink = None
        if output is not None:
            sink = stack.enter_context(
                VideoSink(output, frames.fps, (frames.width, frames.height), fourcc)
            )
        stats = _run(
            frames,
            detector,
            config,
            json_path,
            window="meta-face yolo" if show else None,
            sink=sink,
            show_fps=False,
        )
    _report(stats)
    if output is not None:
        click.echo(f"Wrote {output}")


def _run(
    frames: Any,
    detector: Any,
    config: AppConfig,
    json_path: Path | None,
    *,
    window: str | None,
    sink: Any = None,
    show_fps: bool | None = None,
) -> Any:
    from contextlib import ExitStack

    from meta_face.yolo_face.output import JsonLinesWriter
    from meta_face.yolo_face.pipeline import run_stream

    with ExitStack() as stack:
        writer = stack.enter_context(JsonLinesWriter(json_path)) if json_path else None
        try:
            return run_stream(
                frames,
                detector,
                frame_skip=config.performance.frame_skip,
                sink=sink,
                window=window,
                json_writer=writer,
                draw=_draw_options(config, show_fps),
            )
        except KeyboardInterrupt:
            click.echo("Interrupted")
            raise SystemExit(130) from None
        finally:
            if window is not None:
                import cv2

                cv2.destroyAllWindows()


def _report(stats: Any) -> None:
    click.echo(
        f"Processed {stats.frames} frame(s), inference on {stats.inferred_frames}, "
        f"{stats.mean_fps:.1f} FPS, mean inference {stats.mean_inference_ms:.1f} ms"
    )
