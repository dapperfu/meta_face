# 2x Meta Analysis Notebooks Plan

## Context

- **Sidecar format**: each image has a sibling `.scar` (CBOR) with keys like `face.scrfd.faces`, `face.dlib_detect.faces`, `face.cluster.labels` ([`src/meta_face/sidecar.py`](../src/meta_face/sidecar.py), [`meta-face-pipeline.md`](meta-face-pipeline.md)).
- **Current scan target**: `/tun/steph_pictures/20XX` year folders (2006–2025) via `mf scan` + workers.
- **Naming convention** (see `.cursor/rules/python/notebook-numbering.mdc`):
  - `01`–`09` — per-image annotation
  - `20`–`29` — collection-level meta analysis (this work)

```mermaid
flowchart LR
    root["Photo root e.g. /tun/steph_pictures"]
    discover["discover_images / discover_sidecars"]
    parallel["ProcessPoolExecutor load summaries"]
    df["pandas DataFrame"]
    stats["aggregate: coverage, dist, by_year, clusters"]
    nb["2x_*.ipynb charts and tables"]

    root --> discover --> parallel --> df --> stats --> nb
```

## 1. Analysis library (`src/meta_face/analysis/`)

Thin, importable layer — notebooks and future CLI can share it. **No GPU, no image loading**; only reads `.scar` files.

### `records.py` — `ImageSummary` dataclass

Per-image row extracted from a sidecar:

| Field | Source |
|-------|--------|
| `media_path`, `sidecar_path` | resolved paths |
| `parent_name` | immediate parent dir (e.g. `2018`) |
| `year` | `parent_name` when it matches `^20\d{2}$`, else `None` |
| `has_sidecar` | `.scar` exists |
| `tools_present` | from existing `list_face_tools()` |
| `scrfd_face_count`, `dlib_face_count` | `len(face.<tool>.faces)` |
| `scrfd_avg_det_score` | mean of `det_score` in scrfd faces |
| `cluster_labels`, `cluster_dlib_labels` | `face.cluster.labels` / `face.cluster_dlib.labels` |
| `cluster_unique`, `cluster_noise` | derived from labels (-1 = noise) |

### `load.py` — single-file extract

`summarize_sidecar(media_path)` uses `sidecar_path_for_media`, `SidecarDocument.from_path`, `get_face_section` / `list_face_tools`. Handles missing sidecar gracefully.

### `collect.py` — bulk + parallel

- `discover_images(root, *, recursive=True)`
- `discover_sidecars(root, *, recursive=True)`
- `collect_summaries(paths, *, workers=None)` with `ProcessPoolExecutor`

### `aggregate.py` — statistics helpers

- `to_dataframe`, `coverage_report`, `faces_per_photo_stats`, `face_count_distribution`
- `stats_by_year`, `stats_by_directory`, `cluster_collection_stats`

## 2. Notebooks (`notebooks/`)

| Notebook | Purpose |
|----------|---------|
| `20_collection_overview.ipynb` | High-level coverage and avg faces per photo |
| `21_faces_per_photo.ipynb` | Face count distributions; per-directory stats |
| `22_year_breakdown.ipynb` | Per-year table 2006–2025; bar charts |
| `23_coverage_gaps.ipynb` | Missing sidecar / scrfd / dlib; gap lists |
| `24_cluster_identity.ipynb` | Cluster stats; identities per year; noise rate |

## 3. Dependencies

Optional `[notebook]` extra: jupyter, pandas, matplotlib, seaborn.

## Design choices

- Both scrfd and dlib_detect as first-class columns.
- Parallel via processes; year from parent folder name (`2018` → year=2018).
- Cluster stats read existing sidecar keys; notebooks do not run HDBSCAN.

## Out of scope

- `1x_*` annotation notebooks
- `mf stats` CLI
- Re-clustering from notebooks
