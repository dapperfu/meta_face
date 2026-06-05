# Add Detectron2 Backend Support

## Assumptions

- **Role:** Face detection only — parallel to `scrfd`, not a replacement for ArcFace embeddings or clustering.
- **Packaging:** Optional extra (`pip install -e ".[detectron2]"`); core install stays InsightFace + ONNX.
- **Model:** User-configurable Detectron2 config + weights via env vars, with `mf download --backend detectron2` to fetch a WIDER FACE RetinaNet checkpoint.
- **ArcFace:** Unchanged — still requires the InsightFace `insightface` meta-tool (`scrfd` + `arcface`).

## Sidecar layout

| Key | Description |
|-----|-------------|
| `face.detectron2.version` | Tool version |
| `face.detectron2.processed_at` | ISO8601 UTC |
| `face.detectron2.faces` | List of `{bbox, landmarks, det_score}` |

## Out of scope

- Annotate overlays from `face.detectron2.*` sidecar data
- ArcFace embeddings from Detectron2 bboxes
- Parametric clustering over non-ArcFace embedding sources
