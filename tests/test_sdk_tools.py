"""Public API contract tests without downloading weights or running ML models."""

import dataclasses
import json
from pathlib import Path
from types import ModuleType, SimpleNamespace

import cv2
import numpy as np
import pandas as pd
import pytest
from click.testing import CliRunner

from meta_face import sdk
from meta_face.cli import main
from meta_face.tools.analysis import deepface, py_feat, uniface


def fake_provider(monkeypatch, provider, module):
    original = sdk.importlib.import_module
    monkeypatch.setattr(sdk.importlib, "import_module", lambda name: (
        module if name == sdk.PROVIDERS[provider][0] else original(name)))


def test_native_stateful_recipe_and_unlisted_public_api(monkeypatch):
    module = ModuleType("uniface")

    class Tracker:
        def __init__(self, offset=0):
            self.count = offset

        def update(self, values):
            self.count += len(values)
            return {"count": self.count, "values": values}

    module.BYTETracker = Tracker
    module.future_operation = lambda value: value + 1
    fake_provider(monkeypatch, "uniface", module)
    recipe = {"steps": [
        {"id": "tracker", "call": "BYTETracker", "kwargs": {"offset": 10}},
        {"id": "first", "call": "$tracker.update", "args": [{"$array": [[1, 2]], "dtype": "float32"}]},
        {"id": "second", "call": "$tracker.update", "args": [{"$ref": "first", "path": ["values"]}]},
        {"id": "future", "call": "future_operation", "args": [{"$ref": "second", "path": ["count"]}]},
    ]}
    assert sdk.SDKSession("uniface").run(recipe) == 13
    with pytest.raises(ValueError, match="public"):
        sdk.SDKSession("uniface").get("__dict__")


def test_result_encoding_does_not_drop_arrays_dataclasses_or_fex():
    @dataclasses.dataclass
    class Face:
        embedding: np.ndarray
        landmarks: np.ndarray

    encoded = sdk.encode_result(Face(np.array([.5, .8]), np.array([[1, 2, 3]])))
    assert encoded["embedding"] == [.5, .8]
    assert encoded["landmarks"] == [[1, 2, 3]]
    assert sdk.encode_result(pd.DataFrame({"AU01": [np.nan], "mesh_x_0": [5]})) == [
        {"AU01": None, "mesh_x_0": 5}]


def test_sdk_cli_offline_discovery_and_pair_verification(monkeypatch):
    runner = CliRunner()
    result = runner.invoke(main, ["sdk", "list", "uniface"])
    assert result.exit_code == 0
    assert "MODNet" in result.output and "FAISS" in result.output
    module = ModuleType("deepface.DeepFace")

    def verify(img1_path, img2_path, model_name, threshold):
        assert (img1_path, img2_path, model_name, threshold) == ("a.jpg", "b.jpg", "SFace", .25)
        return {"verified": True, "distance": np.float32(.125)}

    module.verify = verify
    fake_provider(monkeypatch, "deepface", module)
    request = {"steps": [{"id": "pair", "call": "verify", "kwargs": {
        "img1_path": "a.jpg", "img2_path": "b.jpg", "model_name": "SFace", "threshold": .25}}]}
    result = runner.invoke(main, ["sdk", "run", "deepface", "-"], input=json.dumps(request))
    assert result.exit_code == 0, result.output
    assert json.loads(result.output) == {"verified": True, "distance": .125}


def test_deepface_all_heads_reuse_aligned_bgr_and_keep_every_result(monkeypatch):
    image = np.full((100, 200, 3), [1, 2, 3], dtype=np.uint8)
    crop = image[10:30, 20:60].copy()
    module = ModuleType("deepface.DeepFace")

    def extract_faces(img_path, color_face, normalize_face, enforce_detection, anti_spoofing):
        assert img_path is image
        assert (color_face, normalize_face, enforce_detection, anti_spoofing) == ("bgr", False, True, True)
        return [{"face": crop, "facial_area": {"x": 20, "y": 10, "w": 40, "h": 20},
                 "confidence": .95, "is_real": True, "antispoof_score": .8}]

    def represent(img_path, detector_backend, align, model_name):
        assert img_path is crop and detector_backend == "skip" and align is False
        assert model_name == "SFace"
        return [{"embedding": [.1, .2], "face_confidence": 0,
                 "facial_area": {"x": 0, "y": 0, "w": 40, "h": 20}}]

    def analyze(img_path, detector_backend, align, silent):
        assert img_path is crop and detector_backend == "skip" and align is False
        return [{"age": 25, "gender": {"Woman": 98}, "dominant_gender": "Woman",
                 "emotion": {"happy": 70}, "dominant_emotion": "happy", "future_head": .4,
                 "region": {"x": 0, "y": 0, "w": 40, "h": 20}}]

    module.extract_faces, module.represent, module.analyze = extract_faces, represent, analyze
    fake_provider(monkeypatch, "deepface", module)
    monkeypatch.setenv("META_FACE_DEEPFACE_OPTIONS", json.dumps({
        "operations": ["detect", "represent", "analyze", "liveness"],
        "represent": {"model_name": "SFace"}}))
    payload = deepface.analyze_faces(image, [])
    face = payload["faces"][0]
    assert face["bbox"] == [20, 10, 60, 30]
    assert face["embedding"] == [.1, .2]
    assert face["attributes"]["future_head"] == .4
    assert face["attributes"]["region"]["w"] == 1
    assert face["gender_scores"] == {"Woman": 98}
    assert face["is_real"] is True
    assert payload["face_index_source"] == "deepface"


class FexFixture(pd.DataFrame):
    _metadata = ["au_columns", "emotion_columns", "landmark_columns", "identity_columns"]


@pytest.mark.parametrize("modern", [False, True])
def test_pyfeat_filename_api_full_columns_and_independent_face_indices(monkeypatch, modern):
    image = np.full((30, 40, 3), [9, 8, 7], dtype=np.uint8)
    frame = FexFixture([{"FaceRectX": 5, "FaceRectY": 4, "FaceRectWidth": 10,
                         "FaceRectHeight": 12, "AU01": .8, "happiness": .9,
                         "x_0": 6, "y_0": 8, "Identity_0": .3, "mesh_x_0": 7,
                         "jawOpen": .5, "Pitch": 15}], index=[42])
    frame.au_columns = ["AU01"]
    frame.emotion_columns = ["happiness"]
    frame.landmark_columns = ["x_0", "y_0"]
    frame.identity_columns = ["Identity_0"]
    seen_paths = []

    def run(paths, **kwargs):
        assert isinstance(paths, list) and Path(paths[0]).is_file()
        assert np.array_equal(cv2.imread(paths[0]), image)
        assert kwargs == ({"data_type": "image"} if modern else {})
        seen_paths.extend(paths)
        return frame

    detector = SimpleNamespace(info={"identity_model": "arcface"})
    setattr(detector, "detect" if modern else "detect_image", run)
    monkeypatch.setattr(py_feat, "_get_detector", lambda _: detector)
    monkeypatch.delenv("META_FACE_PY_FEAT_OPTIONS", raising=False)
    output = py_feat.analyze_faces(image, [])
    face = output["faces"][0]
    assert face["face_index"] == 0  # DataFrame index 42 is not a SCRFD index.
    assert face["landmarks"] == {"x_0": 6, "y_0": 8}
    assert face["identity"] == {"Identity_0": .3}
    assert face["raw"]["mesh_x_0"] == 7 and face["raw"]["jawOpen"] == .5
    assert all(not Path(path).exists() for path in seen_paths)


@pytest.mark.parametrize("name", uniface.PHOTO_ANALYSES)
def test_uniface_each_analysis_uses_its_documented_input(name, monkeypatch):
    @dataclasses.dataclass
    class Face:
        bbox: np.ndarray
        landmarks: np.ndarray
        confidence: float = .99

    image = np.zeros((100, 200, 3), dtype=np.uint8)
    face = Face(np.array([20, 10, 60, 40]), np.array([[25, 15], [50, 15]]))
    result = {"retained": np.array([1, 2])}

    def call(*args, **kwargs):
        if name in {"MobileGaze", "HeadPose"}:
            assert args[0].shape == (30, 40, 3)
        else:
            assert args[0] is image
        if name in {"AgeGender", "FairFace", "Emotion", "FaceAttribNet"}:
            assert args[1] is face
        elif name in {"Landmark106", "PIPNet", "MiniFASNet"}:
            assert args[1] is face.bbox
        elif name == "EDifFIQA":
            assert args[1] is face.landmarks
        elif name == "FaceMesh":
            assert kwargs["faces"] == [face]
        elif name in {"BiSeNet", "XSeg"}:
            assert kwargs["landmarks"] is face.landmarks
        return result

    method = ("estimate" if name in {"MobileGaze", "HeadPose"} else
              "get_landmarks" if name in {"Landmark106", "PIPNet"} else
              "parse" if name in {"BiSeNet", "XSeg"} else "predict")
    model = SimpleNamespace(**{method: call})
    detector = SimpleNamespace(detect=lambda bgr: [face])
    recognizer = SimpleNamespace(get_normalized_embedding=lambda bgr, landmarks: np.array([.6, .8]))
    monkeypatch.setattr(uniface, "_get_models", lambda _: (detector, recognizer, {name: model}))
    monkeypatch.delenv("META_FACE_UNIFACE_OPTIONS", raising=False)
    output = uniface.analyze_faces(image, [])
    assert output["faces"][0]["analyses"][name] == {"retained": [1, 2]}
    assert output["faces"][0]["embedding"] == [.6, .8]


def test_independent_sdk_tools_do_not_require_scrfd(monkeypatch):
    from meta_face.deps import require_inference_runtime
    from meta_face.scanner import resolve_per_image_tools
    from meta_face.tools.registry import expand_dependencies

    def unexpected():
        pytest.fail("Independent SDK operation requested InsightFace")

    monkeypatch.setattr("meta_face.deps.require_insightface_runtime", unexpected)
    for provider in sdk.PROVIDERS:
        assert expand_dependencies([provider]) == [provider]
        assert resolve_per_image_tools([provider]) == [provider]
        require_inference_runtime([provider])


def test_sdk_photo_job_writes_normalized_geometry_without_scrfd(tmp_path, monkeypatch):
    from meta_face.jobs import process_image
    from meta_face.sidecar import get_face_section, load_or_create

    image = tmp_path / "sdk.png"
    cv2.imwrite(str(image), np.zeros((100, 200, 3), dtype=np.uint8))
    monkeypatch.setattr(deepface, "availability", lambda: None)
    monkeypatch.setattr(deepface, "analyze_faces", lambda image, faces: {
        "faces": [{"face_index": 0, "bbox": [50, 20, 150, 80], "landmarks": [[100, 40]]}],
        "face_count": 1, "face_index_source": "deepface"})
    result = process_image(str(image), ["deepface"])
    assert result["face_count"] == 1
    doc, _ = load_or_create(image)
    assert not get_face_section(doc, "scrfd")
    stored = get_face_section(doc, "deepface")
    assert stored["faces"][0]["bbox"] == [.25, .2, .75, .8]
    assert stored["faces"][0]["landmarks"] == [[.5, .4]]


def test_inline_sdk_failure_is_reported(tmp_path, monkeypatch):
    import click
    from meta_face.cli import _scan_inline
    from meta_face.scanner import ScanStats

    monkeypatch.setattr("meta_face.scanner.scan_directory_level", lambda *args, **kwargs: (
        ScanStats(discovered=1, enqueued=1), [tmp_path / "bad.jpg"], []))

    def fail(*args, **kwargs):
        raise RuntimeError("SDK model failed")

    monkeypatch.setattr("meta_face.jobs.process_image", fail)
    with pytest.raises(click.ClickException, match="SDK model failed"):
        _scan_inline(tmp_path, ["deepface"], ["deepface"], False, "arcface", False, False)
