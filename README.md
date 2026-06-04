# meta-face

Multi-tool face detection pipeline that writes results into [sidecar-rs](https://github.com/dapperfu/sidecar-rs) `.scar` files. Uses SCRFD detection and ArcFace embeddings via [insightface](https://github.com/deepinsight/insightface), with optional collection-level HDBSCAN clustering and a FAISS nearest-neighbor index.

## Requirements

- Python 3.10+
- NVIDIA GPU with CUDA (for `onnxruntime-gpu` and `faiss-gpu-cu12`)
- Rust toolchain (to build `sidecar_rs` from git)
- Docker (for Redis)

## Quick start

```bash
# Start Redis (host port 26379 = 6379 + 20000)
docker compose up -d

# Install (GPU host)
pip install .

# Terminal 1: worker
mf worker

# Terminal 2: queue a scan
mf scan /keg/photos/

# Force re-process everything
mf scan /keg/photos/ --force

# Only queue clustering
mf scan /keg/photos/ --tools hdbscan

# Inspect sidecar data
mf info photo.jpg
mf info photo.jpg --json
```

### Multiple workers

```bash
mf worker --workers 4
```

## Sidecar layout

Each image `photo.jpg` gets a sibling `photo.scar` CBOR sidecar:

| Key | Description |
|-----|-------------|
| `face.scrfd.version` | SCRFD tool version |
| `face.scrfd.processed_at` | ISO8601 UTC timestamp |
| `face.scrfd.faces` | List of `{bbox, landmarks, det_score}` |
| `face.arcface.version` | ArcFace tool version |
| `face.arcface.processed_at` | ISO8601 UTC timestamp |
| `face.arcface.embeddings` | List of normalized embedding vectors |
| `face.cluster.version` | Cluster tool version |
| `face.cluster.processed_at` | ISO8601 UTC timestamp |
| `face.cluster.labels` | Per-face HDBSCAN cluster ids (-1 = noise) |

Skip behavior: unless `--force`, per-image tools skip files that already have `face.<tool>.version`.

## Architecture

- **Per-image jobs** (RQ queue `meta-face`): SCRFD detection + ArcFace embeddings
- **Aggregate job** (RQ queue `meta-face-cluster`): collect embeddings, build FAISS index at `~/.meta_face/faces.faiss`, run HDBSCAN, write cluster labels back

## Configuration

Environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `META_FACE_REDIS_HOST` | `127.0.0.1` | Redis host |
| `META_FACE_REDIS_PORT` | `26379` | Redis port |
| `META_FACE_REDIS_URL` | derived | Full Redis URL |
| `META_FACE_DATA` | `~/.meta_face` | FAISS index directory |
| `META_FACE_MODEL` | `buffalo_l` | insightface model pack |
| `META_FACE_GPU_ID` | `0` | CUDA device id |

## Supported image formats

JPEG, PNG, WebP, BMP, TIFF, HEIC/HEIF

## Development

```bash
pip install -e ".[dev]"
ruff check src/
```

## Limitations

- Requires GPU packages at install time (`onnxruntime-gpu`, `faiss-gpu-cu12`)
- No concurrent sidecar write coordination (last writer wins)
- Clustering requires ArcFace embeddings present in sidecars
