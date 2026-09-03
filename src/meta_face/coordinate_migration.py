"""Convert existing face geometry without inference or edits to pose.* metadata."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import click
from sidecar_rs import SidecarDocument

from meta_face.coordinates import RELATIVE_UNITS, to_normalized
from meta_face.sidecar import get_face_section, list_face_tools


def migrate_document(doc: SidecarDocument, *, write: bool = False,
                     source_size: tuple[int, int] | None = None) -> dict[str, list[str]]:
    report: dict[str, list[str]] = {"converted": [], "already_normalized": [], "missing_size": []}
    for tool in list_face_tools(doc):
        section = get_face_section(doc, tool)
        if not isinstance(section.get("faces"), list):
            continue
        size = section.get("image_size") or source_size
        if size is None and section.get("coordinates", {}).get("unit") not in RELATIVE_UNITS:
            report["missing_size"].append(tool)
            continue
        converted = to_normalized(section, size)
        if converted == section:
            report["already_normalized"].append(tool)
            continue
        report["converted"].append(tool)
        if write:
            for field, value in converted.items():
                doc.set(f"face.{tool}.{field}", value)
    return report


@click.command("normalize-coordinates")
@click.argument("path", type=click.Path(exists=True, path_type=Path))
@click.option("--write", is_flag=True, help="Apply the conversion under the sidecar lock.")
@click.option("--source-size", nargs=2, type=click.IntRange(min=1), default=None,
              help="Original WIDTH HEIGHT only for records without stored image dimensions.")
def normalize_coordinates(path: Path, write: bool, source_size: Any) -> None:
    """Preview or migrate existing face sidecars to clamped normalized image coordinates.

    Uses recorded original dimensions. Never guesses from a possibly resized
    image. Preserves processing timestamps, model versions, and pose.* keys.
    """
    if path.is_dir():
        paths = sorted(path.rglob("*.scar"))
    else:
        paths = [path if path.suffix.lower() == ".scar" else path.with_suffix(".scar")]
    totals = {"converted": 0, "already_normalized": 0, "missing_size": 0}
    try:
        for scar in paths:
            if not scar.is_file():
                raise FileNotFoundError(f"No existing sidecar: {scar}")
            if write:
                reports = []

                def patch(doc: SidecarDocument) -> None:
                    reports.append(migrate_document(doc, write=True, source_size=source_size))

                SidecarDocument.update_path(scar, patch)
                report = reports[0]
            else:
                report = migrate_document(SidecarDocument.from_path(str(scar)), source_size=source_size)
            for key in totals:
                totals[key] += len(report[key])
            if report["missing_size"]:
                click.echo(f"{scar}: missing original image size for {', '.join(report['missing_size'])}", err=True)
        verb = "Converted" if write else "Would convert"
        click.echo(f"{verb} {totals['converted']} tool section(s) in {len(paths)} sidecar(s); "
                   f"{totals['already_normalized']} already normalized; "
                   f"{totals['missing_size']} need original dimensions.")
    except Exception as exc:
        raise click.ClickException(str(exc)) from exc
