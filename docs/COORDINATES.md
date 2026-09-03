# Normalized image geometry

New face and pose sidecars store image positions as **normalized fractions in `[0, 1]`**, with out-of-frame predictions clamped to the nearest boundary:

```python
x_normalized = max(0.0, min(1.0, x_pixel / image_width))
y_normalized = max(0.0, min(1.0, y_pixel / image_height))
x_pixel = x_normalized * current_image_width
y_pixel = y_normalized * current_image_height
```

A landmark at `(1000, 600)` in a `4000 × 2000` image is stored as `(0.25, 0.3)`. In a `1000 × 500` copy, it resolves to `(250, 150)`. Coordinates use image extents: `1` maps to the right or bottom boundary.

Sections and face records carry `coordinates` metadata:

```json
{
  "coordinates": {
    "schema": 2,
    "unit": "normalized",
    "space": "image",
    "x_reference": "width",
    "y_reference": "height",
    "projected_depth_reference": "width"
  },
  "image_size": [4000, 2000],
  "faces": [{"bbox": [0.2, 0.15, 0.4, 0.5], "landmarks": [[0.25, 0.3]]}]
}
```

`image_size` records the original inference dimensions; each face also carries `source_image_size` when known. Boxes remain `[left, top, right, bottom]`. Dlib's `location` keeps `[top, right, bottom, left]` with the appropriate axis for each value. Named dlib landmarks, InsightFace sparse/dense landmarks, UniFace landmarks/meshes and Py-Feat image-coordinate columns follow the same x/y rule. Box corners are clamped independently. Origin/extent rectangles and saved box widths/heights describe the clipped rectangle. Entirely out-of-frame boxes collapse to zero area and are skipped during face extraction. Non-finite image coordinates are rejected.

Third components are type-dependent: confidence/visibility remains unchanged; InsightFace projected landmark depth and UniFace FaceMesh pixel depth are divided by image width and remain signed. Depth is not an image x/y position and is not clamped. Py-Feat mesh depth is already crop-normalized and retains its native units. Pose angles, physical 3D translations, world landmarks, gaze angles/vectors, embeddings and scores retain their original units.

DeepFace aligned-crop subresults use `space: "aligned_crop"` and clamped normalized coordinates in their own crop frame. Their regions are not treated as full-image coordinates. The outer face box uses the full-image frame. Segmentation masks remain full arrays in their model output frame.

Inference uses pixels. `write_tool_result` converts geometry at persistence using job-supplied image dimensions; pixel geometry without dimensions is rejected before any write. The annotation reader, crop extractor and directory review notebook resolve positions against the current image size. Raw `get_face_section` returns stored values; geometry consumers can use `section_records_in_pixels` or `record_to_pixels` from `meta_face.coordinates`. `to_normalized` converts pixel or legacy relative payloads and clamps existing normalized payloads as well.

The companion `meta_pose` project applies the same convention to canonical keypoints, native image landmarks and boxes. `read_pose_result` returns pixels for the current image by default (`as_pixels=False` exposes stored fractions), and `draw_pose` handles either representation. Native depth retains its original units. Face writers preserve `pose.*`, and pose writers preserve `face.*`. A copy of the companion implementation is recorded in `research/meta_pose_coordinates.patch`.

## Existing sidecars

Readers support both legacy pixel records and schema 1 records marked `unit: "percent"` (0–100). Schema 1 values are divided by 100 and clamped before use. Pixel records with original `image_size` metadata can be rescaled for the current image. Without original dimensions, legacy pixel records retain their pixel interpretation because a reliable resize transform cannot be recovered.

```bash
mf normalize-coordinates /photos          # preview face migration
mf normalize-coordinates /photos --write  # apply without inference
mp normalize-coordinates /photos          # preview pose migration
mp normalize-coordinates /photos --write  # apply without inference
```

Migration converts pixels and schema 1 coordinates to schema 2 and repairs out-of-range normalized records. It uses the sidecar lock, preserves model versions and processing timestamps, and is idempotent. Each command leaves the other project's namespace intact. Schema 1 conversion does not require dimensions. Pixel records missing original dimensions are reported and skipped; supply `--source-size WIDTH HEIGHT` only when those dimensions are known. Migration never guesses from a possibly resized image. Existing collections are not rewritten automatically.

Normalized coordinates support resizing of the same content and orientation, including different horizontal/vertical scales. Cropping, rotation, orientation changes and perspective transforms require their own geometric transform.
