# meta-face

**meta-face** walks a directory of photos, runs face detection and embedding models, and writes results into `.scar` sidecar files beside each image. Use Redis workers for large batches, or `--run-now` for a single-machine run.

## What it does

- Detects faces with InsightFace (default), dlib/`face_recognition`, or optional Detectron2
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

pip install .
mf download                 # fetch InsightFace weights (buffalo_l)

mf worker                   # terminal 1
mf scan /path/to/photos     # terminal 2
```

Detection and clustering in one pass:

```bash
mf scan /path/to/photos --tools insightface,hdbscan
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
| `mf download` | Download model weights (`--backend dlib`, `detectron2`, or `all`) |
| `mf failed` | Show tracebacks for failed RQ jobs |

**Tool aliases:** `insightface` (scrfd + arcface), `face_recognition` (dlib_detect + dlib_embed), `hdbscan` (cluster), `hdbscan_dlib` (cluster_dlib).

```bash
mf scan /photos --tools face_recognition,hdbscan_dlib
pip install -e ".[detectron2]"   # optional Detectron2 backend
```

## Output

For `photo.jpg`, results land in `photo.scar` in the same directory. Keys are prefixed `face.<tool>.` (detections, embeddings, cluster labels). Inspect with `mf info photo.jpg`.

The same `.scar` can also hold `pose.*` keys from [meta_pose](../meta_pose). Writes use `SidecarDocument.update_path` (sidecar-rs ≥ 0.2.1) with a per-file lock so `mf` and `mp` workers can run concurrently without dropping each other's data.

Supported images: JPEG, PNG, WebP, BMP, TIFF, HEIC/HEIF.

## Configuration

Environment variables (`META_FACE_REDIS_HOST`, `META_FACE_MODEL`, `META_FACE_DATA`, etc.) are defined in [`src/meta_face/config.py`](src/meta_face/config.py).

## Limitations

- Requires GPU packages at install time (`onnxruntime-gpu`, `faiss-gpu-cu12`)
- Requires sidecar-rs ≥ 0.2.1 (`SidecarDocument.update_path`) for safe concurrent `.scar` updates
- File locking is fully supported on Unix; non-Unix platforms use best-effort writes
- Clustering requires embeddings for the chosen `--embeddings` source in sidecars
