# Face detection and recognition coverage

Assessment date: 2026-09-02. This describes the inherited implementation, local stored results, upstream capabilities, and the changes made during this review. It complements [the project assessment](PROJECT_ASSESSMENT.md).

**Finding:** the inherited project had two face-detection paths, two embedding paths and a COCO person detector. The person detector has been removed from meta_face (body detection belongs in meta_pose). Seventeen analysis adapters did not constitute seventeen independent face recognizers. DeepFace, UniFace and Py-Feat exposed only small, sometimes incompatible portions of their SDKs. Their public APIs and photo adapters have now been expanded; detection accuracy and collection-level identity quality remain unmeasured.

## Core tools

| Tool | Actual implementation | Coverage and remaining gaps |
|---|---|---|
| SCRFD / InsightFace | `FaceAnalysis` using `buffalo_l`, SCRFD-10GF, hardcoded 640×640 detection preparation | General face detection with landmarks. Full-resolution sports photos are resized for detection; no tiling, resolution retry or rotation retry. Other pack heads also run because the app does not restrict allowed modules. |
| ArcFace / InsightFace | Normalized 512-dimensional embeddings from the same detected faces | Similarity representation, not a complete named-person identification system. Faces missed by SCRFD never receive this embedding. No quality gate, calibrated unknown-person threshold, reference gallery or review workflow. |
| dlib detection / face_recognition | Default frontal HOG detector; configurable CNN alternative; default upstream upsampling of one | Default HOG uses dlib's CPU detector and 68-point predictor without importing face_recognition's CUDA CNN. CNN remains opt-in via `META_FACE_DLIB_MODEL=cnn`. Stored score `1.0` is a placeholder, not calibrated confidence. |
| dlib embeddings | 128-dimensional descriptors, one jitter, subsequently L2-normalized | A second embedding space. Upstream raw-descriptor distance thresholds do not transfer unchanged after this normalization. The wrapper does not implement verification or database lookup. |
| FAISS / HDBSCAN | Separate ArcFace/dlib collections; normalized vectors; inner-product index; HDBSCAN with minimum cluster size 2 | Clustering is not stable named identity management. The saved FAISS index is not searched by the inherited clustering path. No persistent person IDs, incremental assignment, review/merge workflow or calibrated rejection rule. Cluster membership probabilities are not identity-correctness probabilities. |

Sources: [InsightFace package and model packs](https://github.com/deepinsight/insightface/tree/master/python-package), [SCRFD](https://github.com/deepinsight/insightface/tree/master/detection/scrfd), [ArcFace paper](https://arxiv.org/abs/1801.07698), [face_recognition API](https://face-recognition.readthedocs.io/en/latest/face_recognition.html), [dlib recognition example](https://dlib.net/dnn_face_recognition_ex.cpp.html), [FAISS metrics](https://github.com/facebookresearch/faiss/wiki/MetricType-and-distances), [HDBSCAN parameters](https://hdbscan.readthedocs.io/en/latest/parameter_selection.html).

The installed InsightFace 1.0.1 supports automatic multiscale detection upstream, but this project explicitly selects 640×640. Changing a default is not proof of improved sports-photo recall: larger input sizes and tiled inference should be evaluated against labeled images. Model-pack and embedding provenance must also be separated before mixing results from different recognition models in one collection.

## Evidence from existing metadata

I read 100 evenly spaced entries from the sorted 1,545 unique photo paths in the existing ArcFace index metadata. I inspected sidecars and image headers, not a newly inferred collection. The original path list was removed from the repository.

| Observation | Result |
|---|---:|
| Photos sampled | 100 |
| Stored SCRFD boxes | 328 |
| Stored dlib boxes | 176 |
| Stored Detectron2 boxes (legacy, now out of scope) | 811 person boxes |
| Photos with more SCRFD than dlib boxes | 70 |
| Photos with more dlib than SCRFD boxes | 16 |
| Photos with equal counts | 14 |
| Photos with any of the 17 optional analysis markers | 0 |

All sampled configurations record `buffalo_l`, 640×640 detection and dlib HOG. Historical sidecars may still contain COCO person boxes written before that detector was removed. The median detected SCRFD box width is about 88.9 original pixels, corresponding to about 10.2 pixels after scaling the image to a longest side of 640. Of the 328 stored detections, 270 project to widths below 16 pixels.

This is **not a recall/precision benchmark**: indexed photos are a biased sample, detected boxes are not ground truth, and more boxes may include false positives. The 16 photos where dlib has more detections suggest complementary cases worth reviewing; they do not prove 16 improved outcomes. The tiny projected boxes identify a concrete resolution question for evaluation.

## All optional adapters

The following separates upstream capability from the inherited adapter's implementation. The three requested SDKs were expanded during this review; remaining defects in other adapters were researched rather than silently treated as working coverage.

| Adapter | Upstream purpose | Inherited gap / current disposition |
|---|---|---|
| [DeepFace](https://github.com/serengil/deepface) | Multiple detector/recognition models, verification/search, attributes, streaming | Previously attributes only, wrong NumPy color convention and no identity outputs. Now independent detection, embeddings, full attribute results and optional anti-spoofing in scan; all public functions through SDK recipes. |
| [UniFace](https://github.com/yakhyo/uniface) | Detection/recognition plus landmarks, attributes, gaze, pose, quality, parsing, matting, tracking and search | Previously called obsolete `UniFace().analyze`. Now uses released 4.x component APIs, preserves all photo-head outputs, and exposes the complete public SDK for other operations. |
| [Py-Feat](https://py-feat.org/pages/models/) | Detection and facial behavior/identity analysis, Fex data processing | Previously passed an ndarray to a filename API, kept only a subset of columns and matched faces by list index. Now supports legacy/current detector interfaces, retains all Fex columns, and exposes detectors and Fex methods through SDK recipes. Newer capabilities require the corresponding installed version. |
| [EmotiEffLib](https://github.com/sb-ai-lab/EmotiEffLib) | Facial emotion analysis | The imported module path does not exist in installed 1.1.1. The declared model name is not passed to the factory. Requires adapter repair. |
| [OpenCV FER](https://github.com/opencv/opencv_zoo/tree/main/models/facial_expression_recognition) | Seven-class expression model | Adapter 1.1.0 uses the upstream seven-class order, 5-point alignment when landmarks exist, and softmax scores. |
| [MediaPipe Face Landmarker](https://developers.google.com/edge/mediapipe/solutions/vision/face_landmarker) | Landmarks, blendshapes, transformation matrices | Adapter 1.1.0 runs one SCRFD crop at a time, checks mesh center against the face box, keeps blendshapes and the transform matrix, and stubs optional audio I/O. Not an identity recognizer. |
| [FER+](https://github.com/onnx/models/tree/main/validated/vision/body_analysis/emotion_ferplus) | Eight-class emotion model | Adapter 1.1.0 stores logits and softmax probabilities. No identity coverage. |
| [LibreFace](https://github.com/ihp-lab/LibreFace) | Facial behavior analysis | Wrapper supplies an ndarray and unsupported `crop_face` keyword to the documented file-path API. Package absent locally. |
| [OpenFace 3.0](https://github.com/CMU-MultiComp-Lab/OpenFace-3.0) | Detection, landmarks and multitask behavior analysis | Wrapper's `OpenFace().predict` interface differs from the documented detector/predictor classes. This is not the older OpenFace identity-embedding package. |
| [Yakhyo gaze](https://github.com/yakhyo/gaze-estimation) | Gaze estimation | Adapter 1.1.0 softmax-decodes each 90-bin head to degrees. |
| [L2CS-Net](https://github.com/Ahmednull/L2CS-Net) | Gaze estimation | Wrapper supplies no weights and calls `predict`; documented pipeline uses `step`/`predict_gaze` with configured weights. Package absent locally. |
| [FairFace ONNX](https://github.com/yakhyo/fairface-onnx) | Demographic classification | Cached model has race, gender and age outputs; adapter reads only the first. Alignment also differs from upstream usage. |
| [BiSeNet face parsing](https://github.com/yakhyo/face-parsing) | Semantic segmentation | Adapter 1.1.0 takes class argmax and stores the spatial mask plus class pixel percents. |
| [ONNX anti-spoofing](https://github.com/yakhyo/face-anti-spoofing) | Presentation-attack/liveness classification | Adapter 1.1.0 uses BGR float pixels, 2.7× crop context, and softmax class probabilities. Still-photo output remains diagnostic only. |
| [Silent-Face-Anti-Spoofing](https://github.com/minivision-ai/Silent-Face-Anti-Spoofing) | Presentation-attack detection | The documented predictor takes an image and model path and returns probabilities. The wrapper omits the path and expects a label/score tuple. |
| [EmoNet](https://github.com/face-analysis/emonet) | Expression, valence, arousal | Network is constructed without loading pretrained weights. Installing the package alone would not establish a meaningful predictor. |
| [InspireFace](https://github.com/HyperInspire/InspireFace) | Detection, feature extraction/search and additional analysis | Wrapper only invokes detection and looks for optional attributes. Feature extraction, processing flags and FeatureHub are not integrated. Package absent locally. |

Static inspection of the cached ONNX graphs confirmed the relevant output shapes. Small synthetic-output checks reproduced the OpenCV class-order, gaze-head and BiSeNet decoding errors; those checks validate integration semantics, not model accuracy.

Implementation validation: 130 `meta_face` tests and 34 companion `meta_pose` tests passed, including normalized coordinate persistence, boundary clamping, anisotropic resizing, legacy migrations, all SDK photo-head contracts and namespace preservation. Two face tests are excluded from the passing run: the optional all-tool availability probe stalls during MediaPipe/PortAudio initialization, and the existing threaded sidecar-lock test intermittently stalls inside `SidecarDocument.update_path` (it passed in an earlier isolated run). The pose cross-process merge test passes. The pose backend imports MediaPipe only when needed, so metadata reading and migration work independently.

## Coverage gaps that remain after the requested changes

1. **Small, distant and difficult faces.** Evaluate resolution/tiling, profiles, occlusion, motion blur, backlighting, hats and partially visible faces. More SDK choices provide candidates; they do not establish recall.
2. **Children and change over time.** The upstream face_recognition project explicitly describes limitations for children. Evaluate the actual age ranges and across-session appearances in this collection rather than transferring headline benchmark accuracy. [Upstream limitations](https://github.com/ageitgey/face_recognition).
3. **Detection reconciliation.** Independent tool results remain separate. No spatial matching, deduplicated union, person-to-face association or selection of the best detection is implemented. A person with no visible face cannot be recovered by adding another identity embedding model.
4. **Identity management.** New embeddings are exposed, but the existing HDBSCAN command still accepts its original ArcFace/dlib sources. SDK-native verification/search is available separately. A shared identity registry needs model-separated storage, representative galleries, unknown-person rejection, stable IDs and human correction.
5. **Quality and calibration.** UniFace quality estimation is now exposed; the collection pipeline still needs an explicit policy for excluding or downweighting poor samples. Recognition thresholds and clustering parameters need labeled calibration. Detector confidence, similarity and clustering membership scores have different meanings.
6. **Geometry and provenance.** New face and companion pose writes now store image boxes/landmarks as normalized width/height fractions clamped to `[0, 1]`, with backward-compatible readers and migration commands. See [coordinate details](docs/COORDINATES.md). Scan skip now requires the current `TOOL_VERSIONS` value, so corrected adapters re-run without `--force`. Schema 1 percent and legacy pixel readers remain so existing collections still load.
7. **End-to-end runtime evidence.** Source compatibility, synthetic contracts and persistence/resize behavior have been tested. DeepFace/UniFace were not installed in the inherited main environment; Py-Feat was 0.6.1. Full inference for every model requires installing the new SDK extras and weights. None of the 100 sampled photos demonstrated prior analysis-adapter coverage.

## Suggested evaluation order

Start with a labeled, representative subset including photos with zero stored detections, disagreements between detectors, groups, close portraits and distant sports action. Include several sessions and separate calibration/test sessions to prevent near-identical burst photos appearing on both sides.

Measure detection precision/recall by original face size, pose and occlusion; compare SCRFD resolutions/tiles, dlib HOG/CNN and selected newly exposed SDK detectors on the same labels. Then evaluate same/different-person pairs and clustering purity, fragmentation, unknown rejection and singleton handling. Record runtime and memory alongside accuracy.

Fix confirmed remaining adapter defects before counting those tools as useful coverage. Improve detector recall before embedding comparison: recognition cannot recover a face for which no usable region was extracted. Keep the current project goal—reviewable, reliable metadata and identity grouping for a photo collection—separate from adding tool names to a registry.
