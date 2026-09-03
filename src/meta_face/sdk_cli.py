"""CLI access to optional SDK APIs and stateful recipes."""

from __future__ import annotations

import json
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any

import click

from meta_face.sdk import PROVIDERS, SDKSession, describe_sdk, write_recipe_output


@click.group("sdk")
def sdk_cmd() -> None:
    """Discover SDK APIs, or enqueue a JSON recipe for an RQ worker."""


@sdk_cmd.command("list")
@click.argument("provider", type=click.Choice(list(PROVIDERS)))
@click.option("--inspect", "target", default=None,
              help="Inspect a public class/module in the installed SDK; use '' for its root.")
def sdk_list(provider: str, target: str | None) -> None:
    """List capability families, or installed signatures and documentation."""
    try:
        with redirect_stdout(click.get_text_stream("stderr")):
            result = describe_sdk(provider, target)
        click.echo(json.dumps(result, indent=2))
    except (ImportError, AttributeError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc


@sdk_cmd.command("run")
@click.argument("provider", type=click.Choice(list(PROVIDERS)))
@click.argument("request", type=click.File("r"))
@click.option("--output", type=click.Path(path_type=Path), help="Write result instead of stdout.")
@click.option("--format", "output_format", type=click.Choice(["json", "npy", "image", "figure"]),
              default="json", show_default=True)
@click.option(
    "--enqueue/--run-now",
    default=True,
    show_default=True,
    help="Enqueue a worker job, or run the recipe in this process.",
)
def sdk_run(
    provider: str,
    request: Any,
    output: Path | None,
    output_format: str,
    enqueue: bool,
) -> None:
    """Execute a JSON recipe (REQUEST may be - for stdin).

    Photo sidecars use ``mf scan --tools sdk`` (deepface, uniface, py_feat).
    This command is for recipes that are not a per-image sidecar write
    (verify, search, tracking, matting). Default is to enqueue for ``mf worker``.
    """
    try:
        if output_format != "json" and output is None:
            raise ValueError("--output is required for non-JSON results")
        recipe = json.load(request)
        if enqueue:
            from meta_face.queue import enqueue_sdk_run

            job_id = enqueue_sdk_run(
                provider,
                recipe,
                output=output,
                output_format=output_format,
            )
            click.echo(f"SDK job enqueued: {job_id}")
            return
        with redirect_stdout(click.get_text_stream("stderr")):
            result = SDKSession(provider).run(recipe)
        encoded = write_recipe_output(
            result, output=output, output_format=output_format
        )
        if output_format == "json" and output is None:
            click.echo(json.dumps(encoded, indent=2, allow_nan=False))
    except Exception as exc:
        raise click.ClickException(f"{provider}: {exc}") from exc
