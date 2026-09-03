# Sports photo analysis

Completed analysis of all **24 JPEGs** in `test_images/`. Original file hashes were verified unchanged. Results include 24 sidecars, per-photo observations, detector comparisons, facial geometry, expression estimates, gaze estimates, segmentation masks, and image-quality measurements.

## Main findings

- **378 face instances** detected with SCRFD at 1280, versus **326** at 640: 52 additional detections (+16.0%). All 326 baseline detections have a one-to-one spatial match at intersection-over-union (IoU) ≥0.30. These are detections across photos, not unique people or audited ground truth.
- **255 dlib HOG detections**; 251 overlap SCRFD and four do not at IoU ≥0.30. Disagreement is a review signal, not a correctness score.
- **141/378 faces (37.3%)** are under 40 native pixels on their shorter side. Median face width is approximately 43 pixels. Resizing these crops cannot restore missing detail.
- **295 faces** yielded spatially checked MediaPipe meshes and 52 blendshapes; 83 attempts yielded no result or a mesh outside the expected face box.
- Detectron2's default COCO checkpoint produced **229 person detections**. Those boxes cover bodies and must not be interpreted as face counts; crowd occlusion also reduces person detection.
- The two expression models agree on **250/378 labels (66.1%)**. Treat these as uncertain visual-expression estimates; neither agreement nor a high model score establishes a person's feelings.

![Face detector comparison](detector_comparison.png)

## Collection and photographic review

The collection contains seven team/trophy frames, five rugby action frames, two rugby gatherings, one indoor candid, and nine photographs in a soccer setting. Files span 2008–2011 and 2026. Total source size is 85.38 MB, with image sizes from 1.71 to 20.67 megapixels.

EXIF camera models: Canon PowerShot A540 (3), Canon EOS DIGITAL REBEL XT (3), Canon EOS DIGITAL REBEL XSi (9), and Nikon Z50_2 (9). Capture metadata, dimensions, file sizes, and quality measurements are in [photos.csv](photos.csv).

Strong selection candidates from the visual review are the compact trophy portrait `20100904_163717.960-3.jpg`, the organized team portrait `20110903_172733.840.jpg`, the running rugby action in `20100918_120908.480-2.jpg`, the close portrait `20260509_102946.570.jpg`, and the goalkeeper action in `20260509_104151.150.jpg`. These are editorial suggestions, not automatic keep/reject decisions.

The indoor candid `20100911_164552.260.jpg` has visible motion blur and a recorded 1/15 s exposure. It is the clearest case where moving subjects limit facial detail. The wide soccer frames are useful for field context, while their small faces make fine analysis harder. The side-profile goalkeeper frame tests orientation; high overall resolution does not make a profile frontal.

The largest near-white pixel fractions are in `20090912_134812.000-2.jpg` (17.60%), `20110903_172929.140.jpg` (14.74%), and `20090912_123727.000-4.jpg` (10.12%). Sky and pale objects contribute to these values; this is not a measured percentage of overexposed faces.

Global sharpness is intentionally **not used to rank photographs**. The clear close portrait has a low whole-image Laplacian score because its background is soft, while textured grass can give wide action frames high scores. Compare native face crops and similar compositions instead.

## Per-photo results

Open **overlay** for SCRFD boxes and five landmarks; amber boxes flag faces under 40 pixels. Open **crops** for every detected face with its photo-local index, native size and detection score. Indices do not link identities across photos.

| # | Photo | SCRFD 640 | SCRFD 1280 | dlib | People | Small faces | Review |
|---:|---|---:|---:|---:|---:|---:|---|
| 01 | 20081115_145931.000.jpg | 44 | 46 | 43 | 10 | 14 | [overlay](overlays/20081115_145931.000.jpg) · [crops](face_sheets/20081115_145931.000.jpg) |
| 02 | 20081115_145939.000.jpg | 44 | 45 | 39 | 12 | 15 | [overlay](overlays/20081115_145939.000.jpg) · [crops](face_sheets/20081115_145939.000.jpg) |
| 03 | 20081115_145941.000.jpg | 35 | 41 | 26 | 10 | 16 | [overlay](overlays/20081115_145941.000.jpg) · [crops](face_sheets/20081115_145941.000.jpg) |
| 04 | 20090912_123727.000-4.jpg | 9 | 15 | 7 | 12 | 7 | [overlay](overlays/20090912_123727.000-4.jpg) · [crops](face_sheets/20090912_123727.000-4.jpg) |
| 05 | 20090912_134812.000-2.jpg | 10 | 26 | 5 | 15 | 0 | [overlay](overlays/20090912_134812.000-2.jpg) · [crops](face_sheets/20090912_134812.000-2.jpg) |
| 06 | 20090912_165430.000.jpg | 65 | 71 | 52 | 11 | 69 | [overlay](overlays/20090912_165430.000.jpg) · [crops](face_sheets/20090912_165430.000.jpg) |
| 07 | 20100904_090203.670-2.jpg | 6 | 9 | 0 | 11 | 4 | [overlay](overlays/20100904_090203.670-2.jpg) · [crops](face_sheets/20100904_090203.670-2.jpg) |
| 08 | 20100904_163717.960-3.jpg | 11 | 11 | 11 | 13 | 0 | [overlay](overlays/20100904_163717.960-3.jpg) · [crops](face_sheets/20100904_163717.960-3.jpg) |
| 09 | 20100907_154123.070-4.jpg | 10 | 10 | 9 | 13 | 0 | [overlay](overlays/20100907_154123.070-4.jpg) · [crops](face_sheets/20100907_154123.070-4.jpg) |
| 10 | 20100911_164552.260.jpg | 7 | 7 | 1 | 9 | 0 | [overlay](overlays/20100911_164552.260.jpg) · [crops](face_sheets/20100911_164552.260.jpg) |
| 11 | 20100918_120908.480-2.jpg | 12 | 12 | 7 | 13 | 9 | [overlay](overlays/20100918_120908.480-2.jpg) · [crops](face_sheets/20100918_120908.480-2.jpg) |
| 12 | 20100918_122112.850.jpg | 11 | 11 | 7 | 13 | 0 | [overlay](overlays/20100918_122112.850.jpg) · [crops](face_sheets/20100918_122112.850.jpg) |
| 13 | 20110903_102551.440.jpg | 13 | 14 | 8 | 16 | 6 | [overlay](overlays/20110903_102551.440.jpg) · [crops](face_sheets/20110903_102551.440.jpg) |
| 14 | 20110903_172733.840.jpg | 17 | 17 | 17 | 17 | 0 | [overlay](overlays/20110903_172733.840.jpg) · [crops](face_sheets/20110903_172733.840.jpg) |
| 15 | 20110903_172929.140.jpg | 14 | 15 | 12 | 14 | 0 | [overlay](overlays/20110903_172929.140.jpg) · [crops](face_sheets/20110903_172929.140.jpg) |
| 16 | 20260425_081137.190.jpg | 3 | 3 | 2 | 8 | 0 | [overlay](overlays/20260425_081137.190.jpg) · [crops](face_sheets/20260425_081137.190.jpg) |
| 17 | 20260425_083120.350.jpg | 1 | 1 | 0 | 1 | 0 | [overlay](overlays/20260425_083120.350.jpg) · [crops](face_sheets/20260425_083120.350.jpg) |
| 18 | 20260502_101610.330.jpg | 2 | 2 | 2 | 2 | 0 | [overlay](overlays/20260502_101610.330.jpg) · [crops](face_sheets/20260502_101610.330.jpg) |
| 19 | 20260502_164040.730.jpg | 6 | 6 | 0 | 6 | 0 | [overlay](overlays/20260502_164040.730.jpg) · [crops](face_sheets/20260502_164040.730.jpg) |
| 20 | 20260509_102946.570.jpg | 1 | 1 | 1 | 2 | 0 | [overlay](overlays/20260509_102946.570.jpg) · [crops](face_sheets/20260509_102946.570.jpg) |
| 21 | 20260509_104151.150.jpg | 1 | 1 | 1 | 2 | 0 | [overlay](overlays/20260509_104151.150.jpg) · [crops](face_sheets/20260509_104151.150.jpg) |
| 22 | 20260510_090743.860.jpg | 2 | 6 | 2 | 7 | 1 | [overlay](overlays/20260510_090743.860.jpg) · [crops](face_sheets/20260510_090743.860.jpg) |
| 23 | 20260510_090743.930.jpg | 1 | 7 | 3 | 7 | 0 | [overlay](overlays/20260510_090743.930.jpg) · [crops](face_sheets/20260510_090743.930.jpg) |
| 24 | 20260510_091002.470.jpg | 1 | 1 | 0 | 5 | 0 | [overlay](overlays/20260510_091002.470.jpg) · [crops](face_sheets/20260510_091002.470.jpg) |

### Visual notes

- **01 — 20081115_145931.000.jpg**: Large rugby group, several rows under the posts. Faces are small and partly overlapped; useful crowd-detection test. Close sequence with the next two frames.
- **02 — 20081115_145939.000.jpg**: Second rugby group frame. Cohesive pose with small faces throughout; compare individual expressions and occlusion against the preceding frame before selecting.
- **03 — 20081115_145941.000.jpg**: Group begins to break pose; more turned and occluded faces. Fewer detections here reflect a real change in face visibility as well as detector limits.
- **04 — 20090912_123727.000-4.jpg**: Tackle/ball-carrier action with overlapping bodies, a back-facing foreground player, dust and distant spectators. Useful profile and occlusion case.
- **05 — 20090912_134812.000-2.jpg**: Large gathering in matching shirts. Many people face away; small faces and backlighting challenge the default detector size. Faces and person counts measure different things.
- **06 — 20090912_165430.000.jpg**: Very large posed group. Strongest crowd stress case in this set; tiny faces dominate and upscaled crops reveal limited native detail.
- **07 — 20100904_090203.670-2.jpg**: Sideline players, tents and ropes. Side views and obscured faces make this a hard detector-comparison case.
- **08 — 20100904_163717.960-3.jpg**: Compact trophy portrait with larger faces. Good geometry test and a strong selection candidate; sunglasses and overlapping rows still need individual review.
- **09 — 20100907_154123.070-4.jpg**: Ball-carrier action with pursuers behind. Strong central action; background faces are smaller and attention is split across the group.
- **10 — 20100911_164552.260.jpg**: Indoor candid with strong warm light and visible motion blur. Slow exposure and moving faces limit fine landmarks and expression estimates.
- **11 — 20100918_120908.480-2.jpg**: Running ball carrier and approaching players. Useful action selection candidate with readable foreground faces and progressively smaller background faces.
- **12 — 20100918_122112.850.jpg**: Contact play with overlapping bodies, turned heads and dust. A face detector cannot count athletes whose faces are concealed.
- **13 — 20110903_102551.440.jpg**: Close rugby action with a back-facing foreground player. Strong action frame; partial faces, overlap and the background crowd need separate review.
- **14 — 20110903_172733.840.jpg**: Organized two-row team portrait. Clearer faces and an orderly arrangement make this a strong group selection candidate and an easier detector comparison.
- **15 — 20110903_172929.140.jpg**: Trophy celebration with raised arms and a foreground player. Good storytelling; several face directions and overlaps complicate geometry.
- **16 — 20260425_081137.190.jpg**: Soccer sideline/action frame with three foreground faces. One player is cut by the right edge; blurred distant players add context rather than usable face detail.
- **17 — 20260425_083120.350.jpg**: Side-profile goalkeeper portrait in bright light. Good profile test; face orientation matters more than the high image resolution.
- **18 — 20260502_101610.330.jpg**: Goalkeeper standing with a distant spectator behind. Main face has useful detail; any second face should be reviewed as background.
- **19 — 20260502_164040.730.jpg**: Players running toward the ball/camera. Main subject is clear, while faces near the edges and in the background are smaller.
- **20 — 20260509_102946.570.jpg**: Close smiling portrait in a soccer setting. Largest, clearest face study in the set and a strong portrait selection candidate.
- **21 — 20260509_104151.150.jpg**: Goalkeeper action with the main subject isolated against soft background. Strong expression/action selection candidate; foreground obstruction at the right edge.
- **22 — 20260510_090743.860.jpg**: Wide match frame with extensive pitch and small distant faces. Use for field context; fine facial measurements need caution. Near-sequence with the next frame.
- **23 — 20260510_090743.930.jpg**: Second wide match frame, 70 ms later according to the filenames. Similar framing with changed running poses and ball position; compare action timing before selecting.
- **24 — 20260510_091002.470.jpg**: Ball in the air with several players facing away. Useful spatial/action story; visible bodies greatly outnumber frontal faces.

## Similar frames

No byte-identical files were found. Candidate sequences are listed below. Time intervals use the filenames; image-hash distances describe overall appearance, not identity. These methods suggest frames to compare and do not establish duplicates.

| First frame | Second frame | Interval | dHash distance |
|---|---|---:|---:|
| 20081115_145931.000.jpg | 20081115_145939.000.jpg | 8.000 s | 10 |
| 20081115_145931.000.jpg | 20081115_145941.000.jpg | 10.000 s | 9 |
| 20081115_145939.000.jpg | 20081115_145941.000.jpg | 2.000 s | 5 |
| 20260510_090743.860.jpg | 20260510_090743.930.jpg | 0.070 s | 10 |

## Model coverage and interpretation

| Tool / pass | Result | Interpretation |
|---|---|---|
| SCRFD 640 | 326 records | Baseline face detector, confidence threshold 0.5. |
| SCRFD 1280 + geometry | 378 records | Primary face detector; 5 keypoints, 106 2D landmarks, 68 projected 3D landmarks and head pose. |
| dlib HOG | 255 records | Independent CPU face detector and 68 landmarks, native image with one upsample. |
| Detectron2 COCO RetinaNet | 229 records | Person boxes; not comparable to face boxes as the same entity. |
| OpenCV FER | 378 records | Seven expression classes, aligned crops, corrected class order; logits and softmax scores retained. |
| FER+ | 378 records | Eight expression classes on grayscale crops; logits and softmax scores retained. |
| Yakhyo gaze | 378 records | Separate 90-bin yaw/pitch outputs decoded to degrees; model estimates, not verified eye direction. |
| BiSeNet | 378 records | 19-class face parsing decoded by argmax; one indexed PNG mask per face. |
| MediaPipe | 295 records | 478-point mesh, 52 blendshape coefficients, transformation matrix; missing attempts are explicit. |
| MiniFASNetV2 diagnostic | 378 records | Three class probabilities only. A still-photo pass cannot establish liveness or image authenticity. |

DeepFace, UniFace, LibreFace, OpenFace 3, L2CS, EmoNet, InspireFace and the separate face-anti-spoofing package are not installed. EmotiEffLib 1.1.1 is installed but the repository adapter imports a module that this version does not expose. Py-Feat 0.6.1 fails to import (`scipy.stats.binom_test` is absent; NumPy ABI warnings also occur). These tools have no inferred results in this report. No packages were installed or downgraded.

The run covers photographic and visible facial analysis. Recognition embeddings, identity clustering and demographic heads were outside this review's scope.

## Adapter findings

The run used local adapters in `scripts/analyze_sports_photos.py`. Production wrappers in `src/meta_face/tools/analysis/` and default HOG dlib detection now follow the same contracts (OpenCV FER labels/alignment, gaze heads, BiSeNet argmax, MiniFASNet preprocess, per-crop MediaPipe, CPU HOG without a CUDA CNN import). Re-run `mf scan` after pulling; stale sidecar versions are processed again.

1. **OpenCV FER:** the current wrapper's class order is incorrect for the cached seven-output model. This run uses the upstream order and five-point face alignment. [OpenCV reference](https://github.com/opencv/opencv_zoo/blob/main/models/facial_expression_recognition/facial_fer_model.py).
2. **Gaze:** the model returns two 90-bin vectors. The wrapper reads two values from the first vector; this run uses a softmax-weighted expected angle for each output, in degrees. [Gaze reference](https://github.com/yakhyo/gaze-estimation/blob/main/onnx_inference.py).
3. **Parsing:** the model produces class logits. Casting unique logits to integers is not segmentation; this run takes argmax across the 19 classes and saves indexed masks. [Parsing reference](https://github.com/yakhyo/face-parsing/blob/main/onnx_inference.py).
4. **Anti-spoof diagnostic:** the current wrapper uses normalized RGB and a raw output as a score. This run uses BGR float pixels, approximately 2.7× crop context, and softmax probabilities, following the upstream contract. Results remain diagnostic only. [MiniFASNet reference](https://github.com/yakhyo/face-anti-spoofing/blob/main/onnx_inference.py).
5. **MediaPipe:** the current wrapper runs at most ten faces on the full image and assigns results by list order to SCRFD. This run evaluates each SCRFD crop separately and requires the mesh center to fall inside its expected face box. Optional audio I/O is disabled only in the photo process to avoid a PortAudio initialization hang.
6. **dlib:** importing face_recognition initializes a CUDA CNN even when HOG is requested. Direct calls to dlib's CPU HOG and shape predictor recovered this pass without a package change.

## Method, verification and files

Inference ran locally on CPU with cached weights. The primary SCRFD pass uses a 1280×1280 detector input and threshold 0.5; the baseline uses 640×640. Detector matches use one-to-one assignment and IoU ≥0.30. There are no manually labeled ground-truth boxes, so precision and recall were not computed. False detections and missed/occluded faces remain possible.

Full-image quality measures use grayscale luminance: near-black ≤5, near-white ≥250, and Laplacian variance after limiting the longest side to 1600 pixels. Face sharpness uses resized 128×128 crops. The under-40-pixel flag is a review heuristic, not a validated reliability threshold.

Sidecars were refreshed using the current coordinate writer. Image positions use normalized fractions with schema/unit tags; raw JSON and face CSV coordinates are source pixels. Native sizes and angles retain their units. Segmentation masks have a separate 512×512 crop frame; class values are 0–18. Sidecar geometry was checked by converting it back to clipped source-pixel coordinates.

All 24 source hashes, tool record counts, finite model outputs, and sidecar geometry were verified. All 378 source crops completed each of the five ONNX analysis passes; MediaPipe no-result cases are stored explicitly. Annotated previews and crop sheets were visually inspected for representative crowd, profile, indoor, portrait and wide-action cases.

- [Full structured results](results.json) and [summary](summary.json)
- [Per-photo metrics and observations](photos.csv)
- [Per-face geometry, quality, gaze and expression comparison](faces.csv)
- [Detector overlap details](detector_comparison.csv)
- [Similar-frame candidates](similar_frames.csv)
- [Contact sheet 1](contact_sheet_1.jpg) and [contact sheet 2](contact_sheet_2.jpg)

Reproduce from the repository root using the existing environment:

```bash
venv_meta_face/bin/python scripts/analyze_sports_photos.py --phase detect
venv_meta_face/bin/python scripts/analyze_sports_photos.py --phase analysis
venv_meta_face/bin/python scripts/analyze_sports_photos.py --phase mediapipe
venv_meta_face/bin/python scripts/report_sports_photos.py
```

Inference stages reuse completed JSON files by default. `--force` recomputes the requested stage; rerun downstream stages after changing detections. Do not substitute the ordinary `mf scan --tools all_analysis` command for this run, because its current wrappers differ as documented above.
