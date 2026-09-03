"""Tests for meta_face.analysis sidecar summarization and aggregates."""

from __future__ import annotations

from pathlib import Path

import pytest

from meta_face.analysis import (
    coverage_report,
    faces_per_photo_stats,
    stats_by_year,
    summarize_sidecar,
    to_dataframe,
)
from meta_face.sidecar import update_sidecar, write_tool_result


def _write_sidecar(image: Path) -> None:
    def apply(doc: object) -> None:
        write_tool_result(
            doc,  # type: ignore[arg-type]
            "scrfd",
            {
                "faces": [
                    {"bbox": [0, 0, 10, 10], "landmarks": [], "det_score": 0.9},
                    {"bbox": [20, 20, 30, 30], "landmarks": [], "det_score": 0.7},
                ]
            },
            image_size=(100, 100),
        )
        write_tool_result(
            doc,  # type: ignore[arg-type]
            "dlib_detect",
            {"faces": [{"bbox": [0, 0, 10, 10], "det_score": 1.0}]},
            image_size=(100, 100),
        )
        write_tool_result(
            doc,  # type: ignore[arg-type]
            "cluster",
            {"labels": [0, -1], "num_clusters": 1},
        )
        write_tool_result(
            doc,  # type: ignore[arg-type]
            "cluster_dlib",
            {"labels": [2], "num_clusters": 1},
        )

    update_sidecar(image, apply)


def test_summarize_sidecar_extracts_counts_and_clusters(tmp_path: Path) -> None:
    image = tmp_path / "photo.jpg"
    image.write_bytes(b"x")
    _write_sidecar(image)

    summary = summarize_sidecar(image)

    assert summary.has_sidecar is True
    assert summary.scrfd_face_count == 2
    assert summary.dlib_face_count == 1
    assert summary.scrfd_avg_det_score == pytest.approx(0.8)
    assert summary.cluster_labels == [0, -1]
    assert summary.cluster_unique == 1
    assert summary.cluster_noise == 1
    assert summary.cluster_dlib_labels == [2]
    assert summary.cluster_dlib_unique == 1
    assert summary.cluster_dlib_noise == 0
    assert "scrfd" in summary.tools_present


def test_summarize_sidecar_missing_sidecar(tmp_path: Path) -> None:
    image = tmp_path / "missing.jpg"
    image.write_bytes(b"x")

    summary = summarize_sidecar(image)

    assert summary.has_sidecar is False
    assert summary.scrfd_face_count is None
    assert summary.sidecar_path is None


def test_year_from_parent() -> None:
    from meta_face.analysis.records import ImageSummary

    assert ImageSummary.year_from_parent("2018") == 2018
    assert ImageSummary.year_from_parent("summer") is None


def test_aggregate_coverage_and_year(tmp_path: Path) -> None:
    pytest.importorskip("pandas")

    year_2018 = tmp_path / "2018"
    year_2019 = tmp_path / "2019"
    year_2018.mkdir()
    year_2019.mkdir()

    img_a = year_2018 / "a.jpg"
    img_b = year_2019 / "b.jpg"
    img_c = year_2019 / "c.jpg"
    for img in (img_a, img_b, img_c):
        img.write_bytes(b"x")
    _write_sidecar(img_a)
    _write_sidecar(img_b)

    summaries = [summarize_sidecar(p) for p in (img_a, img_b, img_c)]
    df = to_dataframe(summaries)

    report = coverage_report(df)
    assert report["total_images"] == 3
    assert report["with_sidecar"] == 2
    assert report["without_sidecar"] == 1
    assert report["with_scrfd"] == 2

    stats = faces_per_photo_stats(df, tool="scrfd")
    assert stats["scrfd"]["count"] == 2
    assert stats["scrfd"]["mean"] == 2.0

    yearly = stats_by_year(df)
    assert len(yearly) == 2
    assert int(yearly.loc[yearly["year"] == 2018, "photos"].iloc[0]) == 1
    assert int(yearly.loc[yearly["year"] == 2019, "photos"].iloc[0]) == 2
