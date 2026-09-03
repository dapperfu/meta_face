**meta_face — inherited project assessment**

Assessed September 2, 2026 against the current working tree and committed `main` at `ed1130f` (package version `0.1.0`).

**Assessment: a working experimental photo-processing pipeline with real historical output, incomplete reproducibility, and unresolved correctness gaps.** The original pipeline structure exists. The expanded feature set has not reached a dependable, validated goal state.

This assessment distinguishes documented intent, implementation, observed historical output, and checks performed in this session. The proposed completion criteria below are recommendations inferred from the repository; they are not previously agreed acceptance criteria.

**Intended goals**

The central goal is to turn a directory of photographs into reusable face metadata: locate faces, calculate embeddings, group similar faces across photographs, and preserve the results beside the originals. The `.scar` documents are the source of truth; Redis coordinates work, and FAISS indexes are derived artifacts. The same sidecars can contain metadata from `meta_pose` and other photography tools.

The original [pipeline plan](/projects/spring_photography/meta_face/plans/meta-face-pipeline.md:3) specified SCRFD detection, ArcFace embeddings, HDBSCAN clustering, GPU-host execution, and a runnable CLI/worker structure. Subsequent plans broadened this to dlib, annotation notebooks, collection statistics, and 17 optional facial-analysis adapters. A Detectron2 COCO person detector was later added and then removed; body detection belongs in meta_pose.

The business purpose is inferred as making personal and sports photo collections easier to inspect and group by people. The implemented identity mechanism produces anonymous cluster labels. No person-name catalogue, reviewed identity management, image-search interface, or end-user photo application was found; those should not automatically be treated as unfinished requirements.

| Goal | Inherited state | What would establish completion |
|---|---|---|
| Detect faces and compute embeddings | SCRFD/ArcFace and dlib pipelines are implemented. Existing sidecars demonstrate both ran. | A repeatable run on representative photos, verified face boxes, correct embedding dimensions, and consistent face-to-embedding alignment. |
| Preserve interoperable metadata | Namespaced records, tool versions/timestamps, and locked sidecar updates are implemented. Face/pose merge tests pass when the sibling package is available. | Documented schema and dependency versions; tests for concurrent writes, incomplete records, model changes, and invalidation of dependent outputs. |
| Process large collections and resume work | Recursive discovery, per-backend RQ jobs, multiple workers, inline execution, and skip markers exist. Error reporting and aggregate-job ordering have gaps. | Every discovered image is accounted for as successful, skipped, or failed; failures remain visible; aggregate work waits for all required image jobs. |
| Group identities across a collection | Both embedding sources are supported. A saved ArcFace run contains 5,059 face references across 1,545 photos. | Reviewed cluster quality, collection/run identifiers, correct distinct-identity statistics, and indexes isolated by collection. |
| Inspect results and coverage | Four annotation notebooks, five collection notebooks, crop helpers, and a statistics library exist. Most collection notebooks have no saved completed execution; statistics have correctness limitations. | Notebooks execute from a fresh kernel against a documented fixture and representative collection; totals and coverage reconcile with source records. |
| Add expression, gaze, attributes, parsing, and liveness | All 17 adapters are registered. The integration plan labels eight as full integrations and nine as stubs. Registration is tested; end-to-end model validation remains incomplete. | An explicit supported subset with tested installation, model loading, per-face output association, and reproducible sample results. |
| Make the project reproducible for its next owner | A local environment and substantial implementation exist, but required modules are untracked and dependency setup is incomplete. | A complete versioned checkout, reproducible environment, documented sibling dependencies, and automated checks that pass from a clean checkout. |

**Evidence of real use**

The existing ArcFace index metadata contains 5,059 face references for 1,545 distinct images. Its corresponding FAISS file exists.

A documented sample sidecar contained seven SCRFD detections with seven 512-dimensional ArcFace embeddings, and one dlib detection with one 128-dimensional embedding. A cluster record reported 373 clusters over 5,059 faces. These are algorithmic clusters, not a validated count of actual people.

Saved annotation notebook output also shows a historical inference run. These artifacts establish that parts of the pipeline were used successfully. They do not establish current GPU readiness, accuracy, or completion of `/tun/steph_pictures` or the full sports collection. No full collection census was performed.

**Issues that prevent a dependable goal state**

1. **The committed checkout is incomplete.** Before this assessment, there were 20 modified tracked files, 31 untracked files, and an additional dirty rules submodule. Untracked implementation includes `bbox.py`, `detectron2_model.py`, `tools/sidecar_encode.py`, the collection-analysis package, notebook helpers, and tests. Committed analysis code already imports some of these missing modules. Exporting `HEAD` to a temporary directory and importing `meta_face.tools.analysis.registry` fails with `ModuleNotFoundError: meta_face.tools.sidecar_encode`. A clean checkout therefore cannot reproduce the working tree. See [analysis base](/projects/spring_photography/meta_face/src/meta_face/tools/analysis/base.py:10) and [crop helpers](/projects/spring_photography/meta_face/src/meta_face/tools/analysis/crops.py:10).

2. **~~Default Detectron2 results have the wrong meaning.~~ Resolved.** COCO RetinaNet person detection was removed from meta_face. Body/person detection belongs in meta_pose.

3. **Inline scans can conceal failures.** `_scan_inline` submits directory tasks without retaining their futures or calling `result()`. An exception ends that directory task but is not propagated to the CLI. A temporary corrupt JPEG, with only dependency checks bypassed, produced exit status `0` and “Nothing to process.” Remaining images in that directory can also be abandoned after an error. See [inline scanning](/projects/spring_photography/meta_face/src/meta_face/cli.py:211).

4. **Queued clustering can start before its inputs are complete.** The CLI enqueues clustering immediately after directory/image jobs. The cluster enqueue call has no dependency or completion barrier. Queue priority does not ensure completion when other workers are still processing images. A mocked enqueue check confirmed no `depends_on` value. See [scan scheduling](/projects/spring_photography/meta_face/src/meta_face/cli.py:149) and [cluster queue helper](/projects/spring_photography/meta_face/src/meta_face/queue.py:80).

5. **Identity totals and index scope are unreliable across collections.** `cluster_collection_stats` sums each photo's distinct-label count. One identity appearing in three photographs therefore reports `total_identities = 3`; this was reproduced with synthetic summaries. Index filenames vary by embedding tool, but not by collection, so successive runs with the same data directory overwrite the previous index and reference map. Labels have no collection/run namespace for safely combining independent cluster runs. See [identity statistics](/projects/spring_photography/meta_face/src/meta_face/analysis/aggregate.py:161), [artifact paths](/projects/spring_photography/meta_face/src/meta_face/config.py:206), and [cluster payloads](/projects/spring_photography/meta_face/src/meta_face/tools/cluster.py:177).

6. **“Already processed” means only that a version key exists.** A sidecar containing an old version marker and no detection payload is treated as complete; an isolated check confirmed this. There is no comparison against the current model/tool version or image contents. Rerunning only a detector also leaves dependent embeddings, analysis, and clusters untouched, risking positional misalignment after detections change. This is consistent with the original simple skip rule, but insufficient for reliable ongoing maintenance. See [skip marker](/projects/spring_photography/meta_face/src/meta_face/sidecar.py:84) and [job selection](/projects/spring_photography/meta_face/src/meta_face/jobs.py:28).

7. **Optional analysis support is broader than its validation.** The [integration plan](/projects/spring_photography/meta_face/plans/facial-analysis-tools-integration.md:89) still leaves its end-to-end GPU test and demonstration notebook unchecked. Some adapters do not follow the stated crop-based contract: LibreFace repeatedly analyzes the full image for each supplied face index, without associating outputs to the individual crop. In addition, runtime filtering removes unavailable tools from the root image jobs but passes the original tool list to child-directory jobs. An isolated CLI check confirmed removed analysis tools reappear in child scan arguments. See [LibreFace adapter](/projects/spring_photography/meta_face/src/meta_face/tools/analysis/libreface.py:32) and [runtime filtering and recursion](/projects/spring_photography/meta_face/src/meta_face/cli.py:111).

**Documentation and operational gaps**

The original plans say ArcFace-only embeddings and an InsightFace-only default, while current defaults include dlib. Historical Detectron2 plans describe a face-trained WIDER FACE checkpoint that was never the shipped default. These plans are useful history but are not a consistent current specification.

Notebook `01` declares `FORCE_DETECT = False` but calls `resolve_face_records(..., force=True)`, overriding its documented sidecar-first behavior. Notebook `24` contains a saved `NameError`; its current source includes the named import, so the saved error alone does not prove the current notebook still fails from a fresh kernel. The other collection notebooks lack saved complete executions. Year aggregation recognizes only an immediate parent named `20XX`, which omits nested layouts such as `2026/05-May/photo.jpg`.

No dependency lockfile, CI workflow, representative accuracy benchmark, or agreed quality/throughput target was found in the project. Dependencies include GPU packages and notebook packages in the base install. `meta_pose` is needed by integration tests but is not declared as a test dependency. The source distribution configuration also excludes the notebooks and tests described in the README.

The clustering implementation creates a FAISS `IndexFlatIP` and separately feeds the full embedding matrix to HDBSCAN. It does not use FAISS to accelerate HDBSCAN or explicitly transfer the index to a GPU. Large-collection performance should therefore be measured rather than inferred from the GPU dependency list.

**Validation performed**

| Check | Result |
|---|---|
| Existing tests, excluding the stalled availability test | `87 passed, 3 failed, 1 deselected` in the inherited `venv_meta_face`; all three failures were missing `meta_pose`. |
| The three sidecar merge tests with sibling `meta_pose/src` on `PYTHONPATH` | `3 passed`, including concurrent face/pose namespace writes. No dependency installation was required. |
| Full-suite attempt | Interrupted after progress stalled in the analysis-tool availability check. That test initializes optional model runtimes; its success is unverified. |
| Ruff over `src`, `tests`, and `notebooks` | 21 findings in nine files: 14 import-placement findings, three unused locals, four unused imports. |
| Package metadata consistency | `pip check` reported no broken requirements. A separate pandas import emitted a NumPy/numexpr binary-compatibility diagnostic, despite aggregate checks completing. |
| GPU visibility | `nvidia-smi` could not communicate with the driver from this session. This does not establish the state of the intended external GPU runtime. |
| Committed checkout import | Analysis-registry import failed because a required module is untracked. |
| Isolated behavior checks | Confirmed hidden inline failure, duplicate identity counting, marker-only skipping, absent cluster dependency, and propagation of filtered tools into child jobs. |

The main test commands were:

```bash
venv_meta_face/bin/python -m pytest -q -p no:cacheprovider \
  -k 'not test_tool_availability_returns_message_or_none'

PYTHONPATH=/projects/spring_photography/meta_pose/src \
  venv_meta_face/bin/python -m pytest -q -p no:cacheprovider \
  tests/test_sidecar_merge.py

venv_meta_face/bin/ruff check src tests notebooks
```

No new photo inference or live queue execution was performed. Model URLs and live Git remotes were not checked. Application code and photo sidecars were not changed.

**Recommended goal state and next steps**

The next milestone should be a reproducible, observable core pipeline: from a fresh checkout, an owner can process a representative collection, see every failure, rerun safely, inspect correct face records, and obtain collection-scoped clusters and accurate coverage statistics. Treat optional analysis adapters as individually supported capabilities until their outputs are validated.

1. Preserve and review the inherited working tree, then version all required source, helpers, tests, and build instructions. Establish one documented environment and declare integration-test setup.
2. Correct the face/person semantics, propagate inline failures, enforce scan-to-cluster completion, and carry the filtered tool list through recursive scans.
3. Add version/content-aware reprocessing, dependent-output invalidation, collection/run identity, and correct distinct-cluster aggregation.
4. Validate the core on a fixed representative set containing no-face photos, group photos, small/profile faces, supported formats, and a corrupt file. Run both inline and queued modes; reconcile discovered/successful/skipped/failed totals and review detection/cluster quality.
5. Execute the notebooks from fresh kernels, reconcile their statistics, and define the supported optional analysis subset. Establish accuracy and performance thresholds with the owner before calling the expanded project complete.

The repository does not provide enough evidence to assign a meaningful completion percentage. Its architecture and historical core execution are established; reproducibility, reliable orchestration, data semantics, and expanded-model validation remain unfinished.
