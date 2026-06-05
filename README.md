# meta-face

**meta-face** walks a directory of photos, runs face detection and embedding models, and writes results into `.scar` sidecar files beside each image. Use Redis workers for large batches, or `--run-now` for a single-machine run.

## What it does

- Detects faces with InsightFace, dlib/`face_recognition`, and Detectron2 by default
- Computes face embeddings (512-d ArcFace or 128-d dlib)
- Clusters identities across a collection with HDBSCAN
- Writes all metadata to sibling `.scar` files via [sidecar-rs](https://github.com/dapperfu/sidecar-rs)
- Skips images already processed unless you pass `--force`

## Requirements

- Python 3.10+
- NVIDIA GPU with CUDA (`onnxruntime-gpu`, `faiss-gpu-cu12`)
- Rust toolchain (builds `sidecar-rs` from git)
- Docker (Redis and RQ dashboard only; workers run on the host)

## Quick start

```bash
docker compose up -d          # Redis on :26379, RQ dashboard on :29181

pip install -e ".[detectron2]"   # torch + torchvision only
# Build detectron2 against the same CUDA as PyTorch (CUDA_HOME must match):
CUDA_HOME=/usr/local/cuda-13.0 pip install --no-build-isolation \
  'git+https://github.com/facebookresearch/detectron2.git'

mf download                 # caches Detectron2 model-zoo weights (default: COCO RetinaNet R50)

mf worker                   # terminal 1
mf scan /path/to/photos     # terminal 2 (detect and embed by default)
```

Run without the queue:

```bash
mf scan /path/to/photos --run-now
```

## Commands

| Command | Purpose |
|---------|---------|
| `mf scan PATH` | Discover images and run/enqueue face processing |
| `mf cluster PATH` | Cluster embeddings for a directory |
| `mf annotate PATH` | Draw face overlays to sibling `*_scrfd.*` images |
| `mf info PATH` | Show face data from a sidecar (`--json` for raw output) |
| `mf backends` | List detection backends and availability |
| `mf tools` | List all face tools and optional-dep availability |
| `mf download` | Download model weights (`--backend dlib`, `detectron2`, `analysis`, or `all`) |
| `mf failed` | Show tracebacks for failed RQ jobs |

**Tool aliases:** `insightface` (scrfd + arcface), `face_recognition` (dlib_detect + dlib_embed), `hdbscan` (cluster), `hdbscan_dlib` (cluster_dlib).

**Analysis meta-tools** (crop-based; require scrfd): `expression`, `emotion`, `gaze`, `au`, `blendshapes`, `attributes`, `parsing`, `liveness`, `face_analysis`, `all_analysis`. Install optional deps first, e.g. `pip install -e ".[expression]"`. List availability with `mf tools`.

Default `mf scan` runs insightface, face_recognition, and detectron2 (no clustering). Cluster explicitly with `mf cluster PATH` or add `hdbscan` to `--tools`:

```bash
mf scan /photos --tools insightface,face_recognition,detectron2,hdbscan
mf scan /photos --tools face_recognition,hdbscan_dlib   # dlib embeddings
mf scan /photos --tools scrfd,expression --run-now      # emotion + blendshapes
mf download --backend analysis                          # ONNX/MediaPipe weights
```

## Output

For `photo.jpg`, results land in `photo.scar` in the same directory. Keys are prefixed `face.<tool>.` (detections, embeddings, cluster labels). Inspect with `mf info photo.jpg`.

The same `.scar` can also hold `pose.*` keys from [meta_pose](../meta_pose). Writes use `SidecarDocument.update_path` (sidecar-rs ≥ 0.2.1) with a per-file lock so `mf` and `mp` workers can run concurrently without dropping each other's data.

Supported images: JPEG, PNG, WebP, BMP, TIFF, HEIC/HEIF.

## Notebooks

Interactive examples live in [`notebooks/`](notebooks/). Each notebook covers one focused task; numbering uses sequential prefixes (`01`–`09` annotation, `20`–`29` meta analysis).

**Annotation** (`01`–`09`) — GPU inference:

| Notebook | Purpose |
|----------|---------|
| `01_annotate_overview.ipynb` | Original vs annotated side-by-side |
| `02_face_crops_buffered.ipynb` | Per-face crops with configurable bbox buffer % |
| `03_face_attributes.ipynb` | Print all extracted face fields (age, gender, pose, landmarks) |
| `04_face_metadata_crops.ipynb` | Buffered face crops with full metadata panel per face |

```bash
pip install -e .
mf download
make notebook
```

**Meta analysis** (`20`–`29`) — parallel `.scar` reads, no GPU:

| Notebook | Purpose |
|----------|---------|
| `20_collection_overview.ipynb` | High-level coverage and avg faces per photo |
| `21_faces_per_photo.ipynb` | Face count distributions |
| `22_year_breakdown.ipynb` | Per-year stats for `20XX` folders |
| `23_coverage_gaps.ipynb` | Missing sidecars / tool data |
| `24_cluster_identity.ipynb` | Cluster identity statistics |

```bash
pip install -e .
make notebook-analysis
```

For contributors, add test/lint tools with `pip install -e ".[dev]"`. Optional analysis tool deps: `[emotion]`, `[expression]`, `[gaze]`, `[attributes]`, `[liveness]`, or `[all-tools]`.

Set `ROOT` in meta-analysis notebooks (default `/tun/steph_pictures`). See [`notebooks/README.md`](notebooks/README.md).

## Configuration

Environment variables (`META_FACE_REDIS_HOST`, `META_FACE_MODEL`, `META_FACE_DATA`, etc.) are defined in [`src/meta_face/config.py`](src/meta_face/config.py).

## Limitations

- Requires GPU packages at install time (`onnxruntime-gpu`, `faiss-gpu-cu12`)
- Requires sidecar-rs ≥ 0.2.1 (`SidecarDocument.update_path`) for safe concurrent `.scar` updates
- File locking is fully supported on Unix; non-Unix platforms use best-effort writes
- Clustering requires embeddings for the chosen `--embeddings` source in sidecars
