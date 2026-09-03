# meta-face notebooks

Interactive notebooks for exploring face pipeline output. Each notebook does **one small analysis**.

## Setup

```bash
pip install -e .
jupyter notebook
```

Use the project virtualenv kernel (`venv_meta_face` or your local env).

## Numbering convention

Sequential two-digit prefixes group notebooks by topic:

| Range | Series | Purpose |
|-------|--------|---------|
| `01`–`09` | Annotation | Per-image visualization (GPU inference) |
| `20`–`29` | Meta analysis | Collection stats over `.scar` sidecars (no GPU) |

Within each series, increment the second digit for each new focused notebook.

### Annotation (`01`–`09`)

| Notebook | Description |
|----------|-------------|
| `01_annotate_overview.ipynb` | Original vs annotated side-by-side |
| `02_face_crops_buffered.ipynb` | Per-face crops with bbox buffer % |
| `03_face_attributes.ipynb` | Print all extracted face fields (age, gender, pose, landmarks) |
| `04_face_metadata_crops.ipynb` | Full photo with face attributes beside it; buffered crops as detail view |
| `05_directory_tool_review.ipynb` | Directory browser: all-tool overlays, complete per-face metadata, overlap and coverage analysis |

Annotation notebooks load `face.scrfd.faces` from the sibling `.scar` sidecar by default (no GPU inference). Sidecars written by `mf scan` include bbox, landmarks, det_score, age, gender, pose, and dense landmarks. Set `FORCE_DETECT = True` to re-run SCRFD. If no sidecar exists, run `mf scan IMAGE_PATH --tools scrfd` first or enable `FORCE_DETECT`.

Default test image (7 faces):

```python
IMAGE_PATH = Path("test_images/20110903_172733.840.jpg")
```

### Meta analysis (`20`–`29`)

| Notebook | Description |
|----------|-------------|
| `20_collection_overview.ipynb` | High-level: coverage, avg faces per photo |
| `21_faces_per_photo.ipynb` | Face count distributions and per-directory stats |
| `22_year_breakdown.ipynb` | Per-year aggregates for `20XX` folders |
| `23_coverage_gaps.ipynb` | Images missing sidecars or tool data |
| `24_cluster_identity.ipynb` | Cluster / identity statistics |

## Default paths

Set `ROOT` in each `20_*` notebook:

```python
ROOT = Path("/tun/steph_pictures/2018")   # single year
ROOT = Path("/tun/steph_pictures")        # full 2000s tree
```

Meta-analysis notebooks import `meta_face.analysis` and read `.scar` files in parallel.

## All-tool directory review

Open `05_directory_tool_review.ipynb` with the `venv_meta_face` kernel. Set `ROOT`
to a directory of images and sibling `.scar` files and run all cells. No GPU or
model inference is needed. Change `IMAGE_INDEX` and rerun sections 2–3 to browse;
change `FACE_INDEX` to inspect one face crop and spatially overlapping results.
`TOOLS` optionally filters the overlay. Every tool has a consistent color.

Complete saved metadata is expandable, including vectors and tool-level fields.
Missing/corrupt sidecars and orphan sidecars appear in the inventory. Analysis
boxes inherited from a detector are dashed and flagged in overlap tables; they
are not independent detections. Unknown geometry is retained as metadata.
Set `EXPORT_DIR` to save the selected overlay, full metadata, and CSV tables.
