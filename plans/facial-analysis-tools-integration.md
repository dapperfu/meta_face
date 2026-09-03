# Facial analysis tools integration

Update 2026-09-02: DeepFace, UniFace and Py-Feat now detect independently, preserve complete results, and expose public APIs through `mf sdk`. See `docs/SDK_TOOLS.md` and `FACE_TOOL_COVERAGE.md` for the verified API coverage and remaining adapter defects. The phase table below originated as an implementation plan, not an accuracy benchmark.

## Goal

Integrate ALL external facial analysis libraries into meta_face as registered, crop-based tools that reuse SCRFD detections and write namespaced results to `.scar` sidecars.

## Architecture

```mermaid
flowchart LR
    scan["mf scan --tools expression"] --> scrfd[scrfd detect]
    scrfd --> crops[extract face crops]
    crops --> analysis[analysis tools]
    analysis --> scar["face.<tool>.* in .scar"]
```

- **Detection layer** (unchanged): `scrfd`, `arcface`, `dlib_*`
- **Analysis layer** (new): crop-based tools under `src/meta_face/tools/analysis/`
- **Dependency rule**: any analysis tool auto-includes `scrfd` via `expand_dependencies`
- **Optional deps**: `pyproject.toml` extras `[emotion]`, `[expression]`, `[gaze]`, `[attributes]`, `[liveness]`, `[all-tools]`
- **Graceful skip**: unavailable analysis tools are skipped with warnings unless explicitly requested

## Tools integrated (17)

| Tool | Category | Phase | Status |
|------|----------|-------|--------|
| `emotiefflib` | emotion | 1 | Full (ONNX via EmotiEffLib) |
| `opencv_fer` | emotion | 1 | Full (OpenCV zoo ONNX) |
| `mediapipe_blendshapes` | blendshapes | 1 | Full (Face Landmarker) |
| `fer_plus` | emotion | 1 | Full (legacy FER+ ONNX) |
| `libreface` | AU/emotion/gaze | 2 | Stub (requires libreface package) |
| `openface3` | AU/emotion/gaze | 2 | Stub (requires OpenFace 3.0 bindings) |
| `yakhyo_gaze` | gaze | 2 | Full (ONNX download) |
| `l2cs_net` | gaze | 2 | Stub (requires l2cs package) |
| `fairface` | attributes | 3 | Full (ONNX download) |
| `bisenet` | parsing | 3 | Full (ONNX download) |
| `face_antispoof_onnx` | liveness | 3 | Full (ONNX download) |
| `face_anti_spoofing` | liveness | 3 | Stub (Silent-Face-Anti-Spoofing) |
| `uniface` | multi | 3 | Expanded 4.x photo adapter and complete public SDK access |
| `py_feat` | multi | 4 | Full Fex results; legacy/current detectors and public SDK access |
| `emonet` | valence/arousal | 4 | Stub (requires EmoNet) |
| `deepface` | multi | 4 | Detection, embeddings, attributes, liveness and public SDK access |
| `inspireface` | multi | 4 | Stub (requires InspireFace bindings) |

## Meta-tool groups

- `expression` — Phase 1 emotion + blendshapes
- `emotion` — emotiefflib, opencv_fer, fer_plus, deepface, emonet
- `au` — libreface, openface3, py_feat
- `gaze` — yakhyo_gaze, l2cs_net, libreface, openface3, uniface
- `blendshapes` — mediapipe_blendshapes
- `attributes` — fairface, deepface
- `parsing` — bisenet, uniface
- `liveness` — face_antispoof_onnx, face_anti_spoofing, uniface, inspireface
- `face_analysis` — Phase 1 subset
- `all_analysis` — all 17 tools

## Sidecar schema (per analysis tool)

```
face.<tool>.version
face.<tool>.processed_at
face.<tool>.faces[] — per-face results with face_index
face.<tool>.face_count
face.<tool>.model
```

Per-face fields are namespaced inside each record: `emotion_*`, `action_units`, `gaze`, `blendshape_coefficients`, `race_*`, `liveness_*`, etc.

## Usage

```bash
pip install -e ".[expression]"     # Phase 1 deps
mf download --backend analysis     # ONNX/MediaPipe weights
mf tools                           # list availability
mf scan /photos --tools scrfd,expression --run-now
mf scan /photos --tools insightface,face_analysis
mf scan /photos --tools scrfd,emotiefflib,yakhyo_gaze,fairface
```

## Blockers / notes

- **UniFace**: released 4.x package is included in `[sdk-tools]`; other SDK adapters still require the API/runtime repairs described in the coverage report
- **DeepFace / py-feat / EmoNet**: heavy transitive deps; optional extras only
- **face_anti_spoofing**: expects Silent-Face-Anti-Spoofing repo layout (`src.anti_spoof_predict`)
- **Model URLs**: best-effort in `analysis_models.py`; some upstream URLs may 404 — manual placement documented in error messages
- **L2CS-Net**: package name/API varies (`l2cs-nnet` on PyPI); stub wraps common import path

## TODO

- [x] Save plan to plans/
- [x] Create analysis tool infrastructure
- [x] Register all 17 tools
- [x] Wire into jobs.py pipeline
- [x] Add pyproject optional deps
- [x] Add mf tools / download support
- [x] Add registry smoke tests
- [ ] End-to-end GPU test with emotiefflib on sample image
- [ ] Notebook demonstrating expression meta-tool
