# Add dlib / face_recognition Backend

## Overview

Add a second face pipeline backed by the face_recognition/dlib library alongside the existing InsightFace SCRFD+ArcFace tools, with separate sidecar keys, FAISS indexes, and cluster outputs per embedding backend.

## Target architecture

Both pipelines can run on the **same image**; each writes independent sidecar keys. Default scan behavior stays InsightFace-only.

| Meta-tool | Expands to | Sidecar data | Embedding dim |
|-----------|------------|--------------|---------------|
| `insightface` (existing) | `scrfd`, `arcface` | `face.scrfd.faces`, `face.arcface.embeddings` | 512 |
| `face_recognition` (new) | `dlib_detect`, `dlib_embed` | `face.dlib_detect.faces`, `face.dlib_embed.embeddings` | 128 |

## Implementation

See attached plan for full implementation details including tool modules, config/registry updates, job dispatch refactor, clustering with separate FAISS indexes, CLI changes, and tests.
