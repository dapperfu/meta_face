"""CLI access to optional SDK APIs and stateful recipes."""

from __future__ import annotations

import json
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any

import click

from meta_face.sdk import PROVIDERS, SDKSession, describe_sdk, encode_result


@click.group("sdk")
def sdk_cmd() -> None:
    """Discover and call DeepFace, UniFace and Py-Feat public APIs."""


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
def sdk_run(provider: str, request: Any, output: Path | None, output_format: str) -> None:
    """Execute a JSON recipe (REQUEST may be - for stdin).

    Original arguments and native objects remain available between steps. Use
    image/npy for masks, crops or anonymized frames; figure saves a plot.
    """
    try:
        if output_format != "json" and output is None:
            raise ValueError("--output is required for non-JSON results")
        recipe = json.load(request)
        with redirect_stdout(click.get_text_stream("stderr")):
            result = SDKSession(provider).run(recipe)
        if output_format == "json":
            encoded = json.dumps(encode_result(result), indent=2, allow_nan=False)
            if output is None:
                click.echo(encoded)
            else:
                output.write_text(encoded + "\n", encoding="utf-8")
        elif output_format == "npy":
            import numpy as np

            with output.open("wb") as stream:
                np.save(stream, result, allow_pickle=False)
        elif output_format == "image":
            import cv2

            if not cv2.imwrite(str(output), result):
                raise OSError(f"Could not write image: {output}")
        else:
            figure = result.figure if hasattr(result, "figure") else result
            figure.savefig(output)
    except Exception as exc:
        raise click.ClickException(f"{provider}: {exc}") from exc
