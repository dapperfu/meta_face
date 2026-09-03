"""Aggregate statistics over ImageSummary collections."""

from __future__ import annotations

from typing import Any, Literal

from .records import ImageSummary

ToolChoice = Literal["scrfd", "dlib_detect", "both"]

_PANDAS_MSG = "pandas is required for analysis aggregates; install with: pip install -e ."


def _require_pandas() -> Any:
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError(_PANDAS_MSG) from exc
    return pd


def _count_column(tool: str) -> str:
    if tool == "scrfd":
        return "scrfd_face_count"
    if tool == "dlib_detect":
        return "dlib_face_count"
    raise ValueError(f"Unknown tool: {tool}")


def to_dataframe(summaries: list[ImageSummary]) -> Any:
    """Convert summaries to a pandas DataFrame."""
    pd = _require_pandas()
    rows = [
        {
            "media_path": str(s.media_path),
            "sidecar_path": str(s.sidecar_path) if s.sidecar_path else None,
            "parent_name": s.parent_name,
            "year": s.year,
            "has_sidecar": s.has_sidecar,
            "tools_present": ",".join(s.tools_present),
            "scrfd_face_count": s.scrfd_face_count,
            "dlib_face_count": s.dlib_face_count,
            "scrfd_avg_det_score": s.scrfd_avg_det_score,
            "cluster_unique": s.cluster_unique,
            "cluster_noise": s.cluster_noise,
            "cluster_dlib_unique": s.cluster_dlib_unique,
            "cluster_dlib_noise": s.cluster_dlib_noise,
        }
        for s in summaries
    ]
    return pd.DataFrame(rows)


def coverage_report(df: Any) -> dict[str, int | float]:
    """Summarize sidecar and tool coverage."""
    total = len(df)
    with_sidecar = int(df["has_sidecar"].sum())
    with_scrfd = int(df["scrfd_face_count"].notna().sum())
    with_dlib = int(df["dlib_face_count"].notna().sum())
    with_cluster = int(df["cluster_unique"].notna().sum())
    with_cluster_dlib = int(df["cluster_dlib_unique"].notna().sum())
    return {
        "total_images": total,
        "with_sidecar": with_sidecar,
        "without_sidecar": total - with_sidecar,
        "with_scrfd": with_scrfd,
        "with_dlib_detect": with_dlib,
        "with_cluster": with_cluster,
        "with_cluster_dlib": with_cluster_dlib,
        "sidecar_pct": round(100.0 * with_sidecar / total, 2) if total else 0.0,
        "scrfd_pct": round(100.0 * with_scrfd / total, 2) if total else 0.0,
        "dlib_pct": round(100.0 * with_dlib / total, 2) if total else 0.0,
    }


def faces_per_photo_stats(
    df: Any,
    *,
    tool: ToolChoice = "both",
) -> dict[str, dict[str, float | int | None]]:
    """Mean, median, std, min, max face counts per photo."""
    pd = _require_pandas()
    tools = ["scrfd", "dlib_detect"] if tool == "both" else [tool]
    out: dict[str, dict[str, float | int | None]] = {}
    for t in tools:
        col = _count_column(t)
        series = df[col].dropna()
        if series.empty:
            out[t] = {"count": 0, "mean": None, "median": None, "std": None, "min": None, "max": None}
            continue
        out[t] = {
            "count": int(series.count()),
            "mean": round(float(series.mean()), 3),
            "median": float(series.median()),
            "std": round(float(series.std()), 3) if len(series) > 1 else 0.0,
            "min": int(series.min()),
            "max": int(series.max()),
        }
    return out


def face_count_distribution(
    df: Any,
    *,
    tool: ToolChoice = "scrfd",
) -> Any:
    """Histogram of face counts (0, 1, 2, 3+)."""
    pd = _require_pandas()
    col = _count_column(tool if tool != "both" else "scrfd")

    def _bucket(count: float | int | None) -> str:
        if count is None or (isinstance(count, float) and pd.isna(count)):
            return "missing"
        c = int(count)
        if c >= 3:
            return "3+"
        return str(c)

    bucketed = df[col].map(_bucket)
    return bucketed.value_counts().sort_index()


def stats_by_year(df: Any) -> Any:
    """Per-year aggregates for 20XX parent folders."""
    pd = _require_pandas()
    yearly = df[df["year"].notna()].copy()
    if yearly.empty:
        return pd.DataFrame()
    return (
        yearly.groupby("year", as_index=False)
        .agg(
            photos=("media_path", "count"),
            scrfd_total=("scrfd_face_count", "sum"),
            scrfd_mean=("scrfd_face_count", "mean"),
            dlib_total=("dlib_face_count", "sum"),
            dlib_mean=("dlib_face_count", "mean"),
            with_sidecar=("has_sidecar", "sum"),
        )
        .sort_values("year")
    )


def stats_by_directory(df: Any, *, depth: int = 1) -> Any:
    """Group stats by immediate parent directory name."""
    pd = _require_pandas()
    _ = depth  # reserved for future path-depth grouping
    return (
        df.groupby("parent_name", as_index=False)
        .agg(
            photos=("media_path", "count"),
            scrfd_total=("scrfd_face_count", "sum"),
            scrfd_mean=("scrfd_face_count", "mean"),
            dlib_total=("dlib_face_count", "sum"),
            dlib_mean=("dlib_face_count", "mean"),
            with_sidecar=("has_sidecar", "sum"),
        )
        .sort_values("parent_name")
    )


def cluster_collection_stats(df: Any, *, top_n: int = 20) -> dict[str, Any]:
    """Collection-wide cluster statistics for arcface and dlib cluster keys."""
    pd = _require_pandas()

    def _stats_for(prefix: str) -> dict[str, Any]:
        unique_col = f"{prefix}_unique"
        noise_col = f"{prefix}_noise"
        present = df[unique_col].notna()
        photos_with = int(present.sum())
        total_identities = int(df.loc[present, unique_col].sum())
        total_noise = int(df.loc[present, noise_col].fillna(0).sum())
        return {
            "photos_with_clusters": photos_with,
            "total_identities": total_identities,
            "total_noise_faces": total_noise,
            "mean_identities_per_photo": round(float(df.loc[present, unique_col].mean()), 3)
            if photos_with
            else None,
        }

    return {
        "cluster": _stats_for("cluster"),
        "cluster_dlib": _stats_for("cluster_dlib"),
        "top_n": top_n,
    }


def gap_paths(df: Any, *, gap: str) -> list[str]:
    """Return media paths missing a given artifact."""
    if gap == "sidecar":
        mask = ~df["has_sidecar"]
    elif gap == "scrfd":
        mask = df["scrfd_face_count"].isna()
    elif gap == "dlib_detect":
        mask = df["dlib_face_count"].isna()
    elif gap == "cluster":
        mask = df["cluster_unique"].isna()
    elif gap == "cluster_dlib":
        mask = df["cluster_dlib_unique"].isna()
    else:
        raise ValueError(f"Unknown gap type: {gap}")
    return df.loc[mask, "media_path"].tolist()
