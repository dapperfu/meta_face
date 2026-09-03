"""Reproducible local sports-photo analysis; run with venv_meta_face/bin/python.

Detection and geometry only: no recognition or demographic heads are loaded.
Original JPEGs are read-only. Results use the project's normalized sidecars.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/meta-face-mpl")

import cv2
import numpy as np
import onnxruntime as ort
from PIL import ExifTags, Image, ImageOps

from meta_face.imaging import load_image
from meta_face.sidecar import update_sidecar, write_tool_result
from meta_face.tools.sidecar_encode import json_safe

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/test_images"
MODEL_ROOT = Path.home() / ".insightface/models/buffalo_l"
ANALYSIS_ROOT = Path.home() / ".meta_face/analysis_models"
cv2.setNumThreads(4)
ort.set_default_logger_severity(3)


def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(json_safe(data), indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def session_options():
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 4
    opts.inter_op_num_threads = 1
    opts.log_severity_level = 3
    return opts


def session(name):
    return ort.InferenceSession(str(ANALYSIS_ROOT / name), sess_options=session_options(),
                               providers=["CPUExecutionProvider"])


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def quality(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    scale = min(1.0, 1600 / max(gray.shape))
    preview = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    small = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    dhash = sum(int(v) << i for i, v in enumerate((small[:, 1:] > small[:, :-1]).flat))
    return dict(mean_luma=float(gray.mean()), luma_std=float(gray.std()),
                dark_pixel_pct=float(np.mean(gray <= 5) * 100),
                bright_pixel_pct=float(np.mean(gray >= 250) * 100),
                laplacian_variance_1600=float(cv2.Laplacian(preview, cv2.CV_64F).var()),
                dhash=f"{dhash:016x}")


def metadata(path):
    with Image.open(path) as im:
        exif = dict(im.getexif())
        try:
            exif.update(im.getexif().get_ifd(34665))
        except (KeyError, TypeError):
            pass
        wanted = {"Make", "Model", "DateTimeOriginal", "ExposureTime", "FNumber",
                  "ISOSpeedRatings", "PhotographicSensitivity", "FocalLength", "LensModel"}
        return {ExifTags.TAGS.get(k, str(k)): str(v) for k, v in exif.items()
                if ExifTags.TAGS.get(k) in wanted}


def persist(path, tool, payload, size):
    update_sidecar(path, lambda doc: write_tool_result(
        doc, tool, payload, version="sports-review-1.0", image_size=size))


def direct_dlib(image):
    """Same HOG/68-point models without importing face_recognition's CUDA CNN."""
    import dlib
    import face_recognition_models

    if not hasattr(direct_dlib, "models"):
        direct_dlib.models = (dlib.get_frontal_face_detector(), dlib.shape_predictor(
            face_recognition_models.pose_predictor_model_location()))
    detector, predictor = direct_dlib.models
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    locations = detector(rgb, 1)
    records = []
    for i, box in enumerate(sorted(locations, key=lambda b: (b.top(), b.left()))):
        shape = predictor(rgb, box)
        records.append(dict(face_index=i,
                            bbox=[box.left(), box.top(), box.right(), box.bottom()],
                            landmarks=[[p.x, p.y] for p in shape.parts()],
                            landmark_count=68, det_model="hog"))
    h, w = image.shape[:2]
    return dict(faces=records, face_count=len(records), image_size=[w, h],
                entity_type="face", det_model="hog", upsample=1, device="CPU",
                adapter="direct dlib HOG and shape_predictor; no CUDA CNN import",
                score_note="HOG detections have no probability score in this export")


def dlib_phase(paths, force):
    for n, path in enumerate(paths, 1):
        target = OUT / "detections" / f"{path.stem}.json"
        row = json.loads(target.read_text())
        if "dlib_detect" in row["tools"] and not force:
            continue
        start = time.perf_counter()
        payload = direct_dlib(load_image(path))
        persist(path, "dlib_detect", payload, (row["width"], row["height"]))
        row["tools"]["dlib_detect"] = payload
        if "dlib_detect" in row["errors"]:
            row.setdefault("recovered_errors", {})["dlib_detect"] = row["errors"].pop("dlib_detect")
        row["timings"]["dlib_detect"] = time.perf_counter() - start
        row["timings"]["total"] = sum(v for k, v in row["timings"].items() if k != "total")
        save_json(target, row)
        print(f"[{n}/{len(paths)}] dlib {path.name}: {payload['face_count']}; "
              f"{row['timings']['dlib_detect']:.1f}s", flush=True)
    target = OUT / "detection_runtime.json"
    status = json.loads(target.read_text())
    status["dlib_detect"] = "ready via direct CPU HOG adapter"
    save_json(target, status)


def softmax(values):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    weights = np.exp(values - values.max())
    return weights / weights.sum()


def crop_box(image, bbox, scale=1.0):
    x1, y1, x2, y2 = bbox
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    hw, hh = (x2 - x1) * scale / 2, (y2 - y1) * scale / 2
    h, w = image.shape[:2]
    a, b = max(0, int(cx - hw)), max(0, int(cy - hh))
    c, d = min(w, int(np.ceil(cx + hw))), min(h, int(np.ceil(cy + hh)))
    return image[b:d, a:c], [a, b, c, d]


def rgb_tensor(crop, size, imagenet=False):
    rgb = cv2.cvtColor(cv2.resize(crop, (size, size)), cv2.COLOR_BGR2RGB).astype(np.float32) / 255
    if imagenet:
        rgb = (rgb - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array(
            [0.229, 0.224, 0.225], dtype=np.float32)
    else:
        rgb = (rgb - 0.5) / 0.5
    return rgb.transpose(2, 0, 1)[None]


def analysis_phase(paths, force):
    from insightface.utils.face_align import norm_crop
    files = {"opencv_fer": "opencv_facial_expression_recognition.onnx",
             "fer_plus": "emotion-ferplus-8.onnx", "yakhyo_gaze": "yakhyo_gaze.onnx",
             "bisenet": "bisenet_face_parsing.onnx", "face_antispoof_onnx": "face_antispoof.onnx"}
    sessions = {k: session(v) for k, v in files.items()}
    save_json(OUT / "analysis_runtime.json", {
        k: dict(inputs=[dict(name=i.name, shape=i.shape) for i in s.get_inputs()],
                outputs=[dict(name=i.name, shape=i.shape) for i in s.get_outputs()],
                provider=s.get_providers(), weights_sha256=sha256(ANALYSIS_ROOT / files[k]))
        for k, s in sessions.items()})
    labels = {"opencv_fer": ["anger", "disgust", "fear", "happiness", "neutral", "sadness", "surprise"],
              "fer_plus": ["neutral", "happiness", "surprise", "sadness", "anger", "disgust", "fear", "contempt"]}
    for n, path in enumerate(paths, 1):
        target = OUT / "analysis" / f"{path.stem}.json"
        if target.exists() and not force:
            continue
        row = json.loads((OUT / "detections" / f"{path.stem}.json").read_text())
        image = load_image(path)
        size = (row["width"], row["height"])
        faces = row["tools"]["scrfd"]["faces"]
        data = {k: dict(faces=[], face_count=0, model=files[k], face_index_source="scrfd",
                       image_size=list(size), adapter="sports-review-1.0", errors=[]) for k in files}
        start = time.perf_counter()
        for face in faces:
            i, bbox = face["face_index"], face["bbox"]
            crop, crop_bounds = crop_box(image, bbox)
            aligned = norm_crop(image, np.asarray(face["kps"], dtype=np.float32), image_size=112)
            for name, s in sessions.items():
                record = dict(face_index=i, bbox=bbox, bbox_source="scrfd",
                              small_face_under_40px=face["small_face_under_40px"])
                try:
                    if name == "opencv_fer":
                        tensor = rgb_tensor(aligned, 112)
                    elif name == "fer_plus":
                        tensor = cv2.resize(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), (64, 64)).astype(
                            np.float32).reshape(1, 1, 64, 64)
                    elif name in ("yakhyo_gaze", "bisenet"):
                        tensor = rgb_tensor(crop, 448 if name == "yakhyo_gaze" else 512, imagenet=True)
                    else:
                        # MiniFASNetV2 upstream: 2.7x context, BGR, float [0,255].
                        context, _ = crop_box(image, bbox, scale=2.7)
                        tensor = cv2.resize(context, (80, 80)).astype(np.float32).transpose(2, 0, 1)[None]
                    outputs = s.run(None, {s.get_inputs()[0].name: tensor})
                    if name in labels:
                        logits = outputs[0].reshape(-1)
                        assert len(logits) == len(labels[name])
                        probs = softmax(logits)
                        record.update(expression_label=labels[name][int(probs.argmax())],
                                      expression_logits=dict(zip(labels[name], map(float, logits))),
                                      expression_scores=dict(zip(labels[name], map(float, probs))),
                                      interpretation="model expression estimate; not a person's internal emotion")
                    elif name == "yakhyo_gaze":
                        assert len(outputs) == 2 and all(np.asarray(o).size == 90 for o in outputs)
                        yaw, pitch = [float(softmax(o) @ np.arange(90) * 4 - 180) for o in outputs]
                        record.update(gaze=dict(yaw=yaw, pitch=pitch, units="degrees"),
                                      decoding="softmax expected bin index * 4 - 180, separately for yaw and pitch")
                    elif name == "bisenet":
                        logits = np.asarray(outputs[0])
                        assert logits.ndim == 4 and logits.shape[1] == 19
                        mask = logits[0].argmax(axis=0).astype(np.uint8)
                        values, counts = np.unique(mask, return_counts=True)
                        dest = OUT / "masks" / f"{path.stem}_face_{i:03d}.png"
                        dest.parent.mkdir(parents=True, exist_ok=True)
                        Image.fromarray(mask).save(dest)
                        record.update(parsing_labels_present=values.tolist(), parsing_shape=list(mask.shape),
                                      class_pixel_percent={str(v): float(c / mask.size * 100)
                                                           for v, c in zip(values, counts)},
                                      parsing_mask=str(dest.relative_to(ROOT)), crop_bbox_pixels=crop_bounds,
                                      mask_coordinate_space="512x512 resized face crop; pixel values are class IDs")
                    else:
                        probs = softmax(outputs[0])
                        assert len(probs) == 3
                        record.update(class_probabilities=probs.tolist(), live_class_probability=float(probs[1]),
                                      interpretation="diagnostic only; still photos cannot establish liveness or authenticity")
                    assert all(np.isfinite(np.asarray(o)).all() for o in outputs)
                    data[name]["faces"].append(record)
                except Exception as exc:
                    data[name]["errors"].append(dict(face_index=i, error=f"{type(exc).__name__}: {exc}"))
        for name, payload in data.items():
            payload["face_count"] = len(payload["faces"])
            persist(path, name, payload, size)
        save_json(target, dict(file=path.name, tools=data, seconds=time.perf_counter() - start))
        print(f"[{n}/{len(paths)}] analysis {path.name}: {len(faces)} faces, "
              f"{sum(len(d['errors']) for d in data.values())} errors; {time.perf_counter()-start:.1f}s", flush=True)


def mediapipe_phase(paths, force):
    import sys
    # MediaPipe eagerly imports optional audio I/O; PortAudio initialization
    # hangs on this host. The library explicitly supports absent sounddevice.
    sys.modules["sounddevice"] = None
    import mediapipe as mp
    from mediapipe.tasks.python.core.base_options import BaseOptions
    from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions
    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(ANALYSIS_ROOT / "face_landmarker.task"),
                                 delegate=BaseOptions.Delegate.CPU), num_faces=1,
        output_face_blendshapes=True, output_facial_transformation_matrixes=True)
    with FaceLandmarker.create_from_options(options) as landmarker:
        for n, path in enumerate(paths, 1):
            target = OUT / "analysis" / f"{path.stem}.json"
            analysis = json.loads(target.read_text())
            if "mediapipe_blendshapes" in analysis["tools"] and not force:
                continue
            row = json.loads((OUT / "detections" / f"{path.stem}.json").read_text())
            image = load_image(path)
            records, missing = [], []
            start = time.perf_counter()
            for face in row["tools"]["scrfd"]["faces"]:
                i, bbox = face["face_index"], face["bbox"]
                crop, bounds = crop_box(image, bbox, scale=1.8)
                # Upsampling supports the face detector; it adds no source detail.
                rgb = cv2.cvtColor(cv2.resize(crop, (256, 256)), cv2.COLOR_BGR2RGB)
                result = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
                if not result.face_landmarks:
                    missing.append(dict(face_index=i, reason="no landmarks detected in source crop"))
                    continue
                x, y, r, b = bounds
                landmarks = [[x+p.x*(r-x), y+p.y*(b-y), p.z*(r-x)] for p in result.face_landmarks[0]]
                points = np.asarray(landmarks)
                center = (points[:, :2].min(axis=0) + points[:, :2].max(axis=0)) / 2
                if not (bbox[0] <= center[0] <= bbox[2] and bbox[1] <= center[1] <= bbox[3]):
                    missing.append(dict(face_index=i, reason="mesh center outside expected SCRFD box"))
                    continue
                coefficients = {s.category_name: float(s.score) for s in result.face_blendshapes[0]}
                assert len(coefficients) == 52
                records.append(dict(face_index=i, bbox=bbox, bbox_source="scrfd",
                                    landmark_3d_478=landmarks, blendshape_coefficients=coefficients,
                                    facial_transformation_matrix=result.facial_transformation_matrixes[0].tolist(),
                                    small_face_under_40px=face["small_face_under_40px"]))
            payload = dict(faces=records, face_count=len(records), face_index_source="scrfd",
                           attempted_face_count=len(row["tools"]["scrfd"]["faces"]),
                           missing_faces=missing, model="face_landmarker.task", blendshape_count=52,
                           image_size=[row["width"], row["height"]], device="CPU",
                           adapter="1.8x crop per SCRFD face; mesh center checked against source bbox",
                           import_workaround="optional sounddevice unavailable for this photo-only process")
            persist(path, "mediapipe_blendshapes", payload, (row["width"], row["height"]))
            analysis["tools"]["mediapipe_blendshapes"] = payload
            analysis["mediapipe_seconds"] = time.perf_counter() - start
            save_json(target, analysis)
            print(f"[{n}/{len(paths)}] MediaPipe {path.name}: {len(records)}/{payload['attempted_face_count']}; "
                  f"{len(missing)} no-result; {time.perf_counter()-start:.1f}s", flush=True)


def detection_phase(paths, force):
    from insightface.model_zoo import get_model
    from insightface.app.common import Face
    from meta_face.tools.face_record import faces_to_sidecar_records

    models = {}
    for key, filename in [("det", "det_10g.onnx"), ("2d", "2d106det.onnx"),
                          ("3d", "1k3d68.onnx")]:
        models[key] = get_model(str(MODEL_ROOT / filename), sess_options=session_options(),
                                providers=["CPUExecutionProvider"])
        if key != "det":
            models[key].prepare(ctx_id=-1)
    status = {"scrfd": "ready", "dlib_detect": "pending",
              "device": "CPU", "onnx_providers": ort.get_available_providers()}
    save_json(OUT / "detection_runtime.json", status)
    for n, path in enumerate(paths, 1):
        target = OUT / "detections" / f"{path.stem}.json"
        if target.exists() and not force:
            print(f"[{n}/{len(paths)}] detection cached: {path.name}", flush=True)
            continue
        start = time.perf_counter()
        image = load_image(path)
        h, w = image.shape[:2]
        row = dict(file=path.name, sha256=sha256(path), width=w, height=h,
                   bytes=path.stat().st_size, exif=metadata(path), quality=quality(image),
                   tools={}, errors={}, timings={})
        for size in (640, 1280):
            t = time.perf_counter()
            models["det"].prepare(ctx_id=-1, input_size=(size, size), det_thresh=0.5)
            boxes, kps = models["det"].detect(image)
            faces = []
            # Stable image-local numbering; no cross-photo identity association.
            order = sorted(range(len(boxes)), key=lambda i: (boxes[i, 1], boxes[i, 0]))
            for i in order:
                face = Face(bbox=boxes[i, :4], det_score=boxes[i, 4], kps=kps[i])
                if size == 1280:
                    models["2d"].get(image, face)
                    models["3d"].get(image, face)
                faces.append(face)
            records = faces_to_sidecar_records(faces)
            for record in records:
                x1, y1, x2, y2 = record["bbox"]
                a, b = max(0, int(x1)), max(0, int(y1))
                c, d = min(w, int(np.ceil(x2))), min(h, int(np.ceil(y2)))
                crop = image[b:d, a:c]
                record["native_width_px"] = x2 - x1
                record["native_height_px"] = y2 - y1
                record["small_face_under_40px"] = min(x2 - x1, y2 - y1) < 40
                if crop.size:
                    g = cv2.cvtColor(cv2.resize(crop, (128, 128)), cv2.COLOR_BGR2GRAY)
                    record["laplacian_variance_128"] = float(cv2.Laplacian(g, cv2.CV_64F).var())
            tool = "scrfd" if size == 1280 else "scrfd_640"
            payload = dict(faces=records, face_count=len(records), model="buffalo_l",
                           det_size=[size, size], det_thresh=0.5, image_size=[w, h],
                           allowed_modules=["detection", "landmark_2d_106", "landmark_3d_68"]
                           if size == 1280 else ["detection"], entity_type="face",
                           device="CPU", run_profile="sports-review-1.0")
            row["tools"][tool] = payload
            row["timings"][tool] = time.perf_counter() - t
            persist(path, tool, payload, (w, h))
        t = time.perf_counter()
        try:
            payload = direct_dlib(image)
            row["tools"]["dlib_detect"] = payload
            persist(path, "dlib_detect", payload, (w, h))
            status["dlib_detect"] = "ready"
        except Exception as exc:
            row["errors"]["dlib_detect"] = f"{type(exc).__name__}: {exc}"
            status["dlib_detect"] = row["errors"]["dlib_detect"]
        row["timings"]["dlib_detect"] = time.perf_counter() - t
        row["timings"]["total"] = time.perf_counter() - start
        save_json(target, row)
        save_json(OUT / "detection_runtime.json", status)
        counts = {k: v["face_count"] for k, v in row["tools"].items()}
        print(f"[{n}/{len(paths)}] {path.name}: {counts}; {row['timings']['total']:.1f}s", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["detect", "dlib", "analysis", "mediapipe"], default="detect")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    paths = sorted((ROOT / "test_images").glob("*.jpg"))
    if args.limit:
        paths = paths[:args.limit]
    OUT.mkdir(parents=True, exist_ok=True)
    if args.phase == "detect":
        detection_phase(paths, args.force)
    elif args.phase == "dlib":
        dlib_phase(paths, args.force)
    elif args.phase == "mediapipe":
        mediapipe_phase(paths, args.force)
    else:
        analysis_phase(paths, args.force)


if __name__ == "__main__":
    main()
