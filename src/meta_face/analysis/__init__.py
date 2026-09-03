"""Bulk sidecar meta-analysis for photo collections."""

from meta_face.analysis.aggregate import (
    cluster_collection_stats,
    coverage_report,
    face_count_distribution,
    faces_per_photo_stats,
    gap_paths,
    stats_by_directory,
    stats_by_year,
    to_dataframe,
)
from meta_face.analysis.collect import collect_summaries, discover_images, discover_sidecars
from meta_face.analysis.load import summarize_sidecar
from meta_face.analysis.records import ImageSummary

__all__ = [
    "ImageSummary",
    "cluster_collection_stats",
    "collect_summaries",
    "coverage_report",
    "discover_images",
    "discover_sidecars",
    "face_count_distribution",
    "faces_per_photo_stats",
    "gap_paths",
    "stats_by_directory",
    "stats_by_year",
    "summarize_sidecar",
    "to_dataframe",
]
