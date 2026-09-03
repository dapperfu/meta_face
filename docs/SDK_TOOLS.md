# DeepFace, UniFace and Py-Feat

The three SDKs run as independent image pipelines in `mf scan`. Their public Python APIs are also accessible through `mf sdk`, including operations requiring several images, a database, video or persistent state.

```bash
pip install -e '.[dev]'
mf sdk list deepface
mf sdk list uniface
mf sdk list py_feat
mf scan /photos --tools deepface,uniface,py_feat --run-now
```

`pip install -e '.[dev]'` installs DeepFace 0.0.100+, UniFace 4.x, Py-Feat 2.1.1+, and the rest of the analysis packages. The inherited Py-Feat 0.6 API remains supported, but lacks newer capabilities. Upstream SDKs manage their model downloads. Listing the catalog does not import or initialize models.

| SDK | Photo pipeline | Additional public API access |
|---|---|---|
| DeepFace | Extraction, embeddings, selected attributes, optional anti-spoofing | `verify`, `find`, `register`, `search`, `build_index`, `stream`, `build_model`, model/backend choices and compatibility functions |
| UniFace | Selectable detector/recognizer; AgeGender, FairFace, Emotion, FaceAttribNet, Landmark106, PIPNet, FaceMesh, MobileGaze, HeadPose, EDifFIQA, MiniFASNet, BiSeNet, XSeg | All detector/recognizer families, FaceAnalyzer, MODNet, BlurFace, BYTETracker, FAISS, alignment, similarity, cache/model utilities and public submodules |
| Py-Feat | Complete Fex rows and column metadata from Detector/Detectorv1/Detectorv2 | Batch/tensor/video detection, detector operations, Fex identity/statistical/temporal methods, plotting and transformations |

Arguments and native result objects are forwarded without restricting model choices. The catalog is a discovery aid, not a fixed execution allowlist. Inspect the installed version's methods and signatures without constructing a model:

```bash
mf sdk list deepface --inspect ''
mf sdk list uniface --inspect FaceMesh
mf sdk list uniface --inspect constants
mf sdk list py_feat --inspect Detectorv2
mf sdk list py_feat --inspect Fex
```

## Recipes

A JSON recipe contains ordered `steps`, each with a unique `id`, public `call`, optional positional `args`, and optional `kwargs`. Constructor results stay in memory; `$step.method` invokes a public method on a prior result. The last result is returned unless `output` selects other results. Trackers and stores preserve state across steps.

```json
{
  "steps": [{
    "id": "pair",
    "call": "verify",
    "kwargs": {
      "img1_path": "/photos/reference.jpg",
      "img2_path": "/photos/candidate.jpg",
      "model_name": "ArcFace",
      "detector_backend": "retinaface",
      "distance_metric": "cosine"
    }
  }]
}
```

Run `mf sdk run deepface request.json --output result.json`; use `-` for stdin. SDK recipes return native geometry in native units; `mf scan` stores image coordinates as normalized fractions clamped to `[0, 1]`. Logs go to stderr.

| JSON expression | Native value |
|---|---|
| `{"$ref":"faces","path":[0,"landmarks"]}` | A previous result, with optional list indices, mapping keys or public attributes |
| `{"$symbol":"constants.AdaFaceWeights.IR_101"}` | SDK constant or enum |
| `{"$image":"photo.jpg"}` | BGR image; optional `color: "rgb"`, `layout: "CHW"`, `batch: true` |
| `{"$array":[[1,2]],"dtype":"float32"}` | NumPy array; accepts nested references |
| `{"$tensor":{"$array":[1,2]},"dtype":"float32","device":"cpu"}` | Torch tensor |
| `{"$tuple":[640,640]}` | Tuple |

Use `--format npy --output mask.npy` for complete masks/arrays, `--format image --output blurred.png` for BGR images, or `--format figure --output plot.png` for a Figure/Axes. JSON retains dataclass fields, arrays, tensors and Fex columns; non-finite values become null. Model objects stay as intermediate results rather than being reduced to strings.

See `examples/sdk/` for verification, analysis, matting, anonymization, tracking, vector search and current Py-Feat recipes. Replace their example paths. Database writes, file exports and streaming happen only when requested by an executed recipe.

Python callers retain the complete native interface:

```python
from meta_face.sdk import SDKSession
sdk = SDKSession("py_feat")
detector = sdk.call("Detectorv2", device="cpu")
fex = detector.detect(["/photos/team.jpg"], data_type="image")
# Every public Fex method remains available on fex.
```

## Photo configuration

Set these JSON environment variables in the inline process or RQ worker. Options and SDK versions are saved with results. Use `--force` to replace results after changing models/options.

DeepFace defaults to `detect`, `represent`, `analyze`; `liveness` enables anti-spoofing during extraction. A single alignment pass supplies BGR crops to downstream heads using the `skip` detector. All returned dictionaries and embeddings are retained; cropped image pixels are available through SDK extraction rather than duplicated in sidecars.

```bash
export META_FACE_DEEPFACE_OPTIONS='{"operations":["detect","represent","analyze","liveness"],"extract_faces":{"detector_backend":"retinaface"},"represent":{"model_name":"Facenet512"},"analyze":{"actions":["emotion","age","gender","race"]}}'
```

Use recipes for arbitrary independent detection/alignment settings on each DeepFace operation. The scan adapter manages crop color, normalization and downstream skip/alignment settings to maintain face association.

UniFace defaults to SCRFD, ArcFace and all 13 photo analysis heads. An empty `analyses` list leaves detection/recognition; `recognizer: null` disables recognition. Constructor options go in detector/recognizer specs or `models`; inference options go in `calls`, keyed by `detect` or the analysis class name.

```bash
export META_FACE_UNIFACE_OPTIONS='{"detector":{"class":"SCRFD","kwargs":{"input_size":{"$tuple":[1280,1280]},"providers":["CUDAExecutionProvider"]}},"recognizer":{"class":"AdaFace","kwargs":{"model_name":{"$symbol":"constants.AdaFaceWeights.IR_101"}}},"analyses":["Landmark106","EDifFIQA","HeadPose"],"models":{"EDifFIQA":{"providers":["CUDAExecutionProvider"]}}}'
```

Each head receives its documented image/crop/box/landmark inputs. Failures propagate. Complete masks can make sidecars large, so select the analyses needed. Tracking/search and image-wide matting/anonymization are available through recipes.

Py-Feat defaults to Detectorv1 when available and legacy Detector otherwise. Select Detectorv2 explicitly for its multitask outputs. A lossless temporary image supports both filename APIs. Every Fex column is retained under each face's `raw`; the temporary input filename is removed. No-face placeholder rows are retained separately and excluded from face counts.

```bash
export META_FACE_PY_FEAT_OPTIONS='{"detector_class":"Detectorv2","detector":{"device":"cpu"},"detect":{"face_detection_threshold":0.6,"progress_bar":false}}'
```

Scan keeps original image dimensions; recipes expose arbitrary resize/batch/tensor/video options. Face indices belong to their own provider, recorded in `face_index_source`, and are never assigned to SCRFD by list order. Embeddings are exposed in outputs and sidecars. The existing `mf cluster --embeddings` command still supports its original ArcFace/dlib spaces; SDK verification/search is available now. Cross-provider reconciliation and a unified identity registry remain separate work.

## References and validation

Checked against released DeepFace 0.0.100, UniFace 4.0.0, Py-Feat 2.1.1 and installed legacy Py-Feat 0.6.1 source. Contract tests cover all UniFace photo heads, both Py-Feat interfaces, DeepFace heads, recipes and serialization. A native UniFace 4 import, similarity call, enum and Face dataclass were also checked. Model inference across every SDK/model still requires its runtime and weights; these checks do not measure collection accuracy.

- [DeepFace API](https://github.com/serengil/deepface/blob/master/deepface/DeepFace.py)
- [UniFace exports](https://github.com/yakhyo/uniface/blob/main/uniface/__init__.py) and [model documentation](https://yakhyo.github.io/uniface/models/)
- [Py-Feat models](https://py-feat.org/pages/models/) and [source](https://github.com/cosanlab/py-feat)
