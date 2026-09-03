"""Normalized storage must survive resizing, preserve confidence/depth semantics and legacy data."""

import cv2
import numpy as np
import pytest
from sidecar_rs import SidecarDocument

from meta_face.coordinate_migration import migrate_document
from meta_face.coordinates import record_to_pixels, to_normalized
from meta_face.sidecar import get_face_section, update_sidecar, write_tool_result
from meta_face.tools.analysis.crops import face_contexts_from_records, scrfd_faces_from_doc
from meta_face.tools.face_record import records_from_sidecar


def test_normalized_storage_and_resized_crop_and_overlay(tmp_path):
    path = tmp_path / "photo.png"
    image = np.zeros((200, 400, 3), dtype=np.uint8)
    image[40:120, 100:300] = [10, 50, 250]
    cv2.imwrite(str(path), image)
    record = {"bbox": [100, 40, 300, 120], "landmarks": [[150, 60]],
              "landmark_3d_68": [[150, 60, 20]], "pose": [5, 10, 15], "det_score": .9}
    scar = update_sidecar(path, lambda doc: write_tool_result(
        doc, "scrfd", {"faces": [record]}, image_size=(400, 200)))
    doc = SidecarDocument.from_path(str(scar))
    stored = get_face_section(doc, "scrfd")["faces"][0]
    assert stored["bbox"] == [.25, .2, .75, .6]
    assert stored["landmarks"] == [[.375, .3]]
    assert stored["landmark_3d_68"] == [[.375, .3, .05]]
    assert stored["pose"] == [5, 10, 15]
    resized = cv2.resize(image, (200, 50), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(str(path), resized)
    records = records_from_sidecar(path)
    assert records[0]["bbox"] == [50, 10, 150, 30]
    assert records[0]["kps"] == [[75, 15]]
    contexts = face_contexts_from_records(resized, scrfd_faces_from_doc(doc), buffer_pct=0)
    assert contexts[0].crop_bgr.shape == (20, 100, 3)
    assert np.all(contexts[0].crop_bgr == [10, 50, 250])
    from meta_face.annotate import draw_annotations

    assert draw_annotations(resized, records).shape == resized.shape


def test_named_dlib_and_detectron_keypoint_confidence():
    result = to_normalized({"faces": [{"location": [20, 100, 60, 40],
        "landmarks_named": {"left_eye": [[50, 25]]},
        "keypoints": [[100, 50, .7]], "keypoint_visibility": [.7]}]}, (200, 100))
    record = result["faces"][0]
    assert record["location"] == [.2, .5, .6, .2]
    assert record["landmarks_named"]["left_eye"] == [[.25, .25]]
    assert record["keypoints"] == [[.5, .5, .7]]
    assert record["keypoint_visibility"] == [.7]


def test_sdk_nested_landmarks_and_crop_reference():
    payload = {"faces": [{"bbox": [100, 20, 300, 60],
        "raw": {"x_0": 150, "y_0": 30, "mesh_x_2": 100, "mesh_y_2": 25,
                "mesh_z_2": 12, "Pitch": 20, "Identity_0": .8},
        "landmarks": {"x_0": 150, "y_0": 30},
        "analyses": {"Landmark106": [[100, 20]],
                     "FaceMesh": [{"landmarks": [[150, 30, 8]], "score": .98}]},
        "attributes": {"coordinates": {"unit": "percent", "space": "aligned_crop"},
                       "region": {"x": 0, "y": 0, "w": 100, "h": 100}}}]}
    result = to_normalized(payload, (400, 100))
    face = result["faces"][0]
    assert face["landmarks"] == {"x_0": .375, "y_0": .3}
    assert face["analyses"]["FaceMesh"][0]["landmarks"] == [[.375, .3, .02]]
    assert face["raw"]["Pitch"] == 20
    assert face["raw"]["Identity_0"] == .8
    assert face["raw"]["mesh_z_2"] == 12  # Py-Feat depth is already crop-normalized.
    assert face["attributes"]["region"] == {"x": 0, "y": 0, "w": 1, "h": 1}
    assert face["attributes"]["coordinates"]["space"] == "aligned_crop"
    assert face["attributes"]["coordinates"]["unit"] == "normalized"
    assert to_normalized(result, (800, 200)) == result  # No double conversion.
    assert to_normalized({"faces": [face]}, (800, 200))["faces"][0]["bbox"] == face["bbox"]


def test_legacy_rescaling_and_missing_size():
    face = {"bbox": [10, 20, 50, 60], "landmarks": [[25, 30]]}
    assert record_to_pixels(face, (200, 200), source_image_size=(100, 100))["bbox"] == [20, 40, 100, 120]
    assert record_to_pixels(face, (200, 200)) == face
    with pytest.raises(ValueError):
        to_normalized({"faces": [face]}, (0, 100))


def test_migration_preserves_pose_namespace_and_provenance():
    doc = SidecarDocument()
    doc.set("face.scrfd.faces", [{"bbox": [10, 20, 50, 60]}])
    doc.set("face.scrfd.image_size", [100, 200])
    doc.set("face.scrfd.version", "1.1.0")
    doc.set("face.scrfd.processed_at", "original")
    doc.set("pose.example.landmarks", [[1, 2, 3]])
    assert migrate_document(doc)["converted"] == ["scrfd"]
    assert "face.scrfd.coordinates" not in doc
    migrate_document(doc, write=True)
    assert doc["face.scrfd.faces"][0]["bbox"] == [.1, .1, .5, .3]
    assert doc["face.scrfd.version"] == "1.1.0"
    assert doc["face.scrfd.processed_at"] == "original"
    assert doc["pose.example.landmarks"] == [[1, 2, 3]]
    assert migrate_document(doc, write=True)["already_normalized"] == ["scrfd"]


def test_zero_faces_are_valid_existing_data():
    doc = SidecarDocument()
    write_tool_result(doc, "scrfd", {"faces": []}, image_size=(100, 100))
    assert scrfd_faces_from_doc(doc) == []


def test_out_of_frame_geometry_is_clamped_without_changing_other_units():
    payload = {"faces": [{
        "bbox": [-40, -10, 500, 150], "bbox_width": 540, "bbox_height": 160,
        "location": [-10, 500, 150, -40],
        "landmarks": [[-1, 101], [400, 0]],
        "landmarks_named": {"nose": [[500, -20]]},
        "landmark_3d_68": [[-10, 120, -200]],
        "keypoints": [{"x": -20, "y": 125, "score": .7}, [450, -5, .3]],
        "analyses": {"Landmark106": [[450, -10]], "PIPNet": [[-5, 105]],
                     "FaceMesh": [{"landmarks": [[500, -10, -100]]}]},
        "raw": {"x_0": -1, "y_0": 101, "mesh_x_2": 500, "mesh_y_2": -1,
                "mesh_z_2": -2, "FaceRectX": -40, "FaceRectY": 80,
                "FaceRectWidth": 200, "FaceRectHeight": 50},
        "facial_area": {"x": -40, "y": 80, "w": 200, "h": 50, "left_eye": [-5, 101]},
        "attributes": {"coordinates": {"unit": "pixels", "space": "aligned_crop"},
                       "image_size": [40, 20], "region": {"x": -4, "y": 10, "w": 60, "h": 15}},
        "pose": [-15, 90, 20], "gaze": [-.2, .3, 2], "embedding": [-2, .5, 3],
    }]}
    result = to_normalized(payload, (400, 100))
    face = result["faces"][0]
    assert face["coordinates"]["schema"] == 2
    assert face["bbox"] == [0, 0, 1, 1]
    assert face["bbox_width"] == face["bbox_height"] == 1
    assert face["location"] == [0, 1, 1, 0]
    assert face["landmarks"] == [[0, 1], [1, 0]]
    assert face["landmarks_named"] == {"nose": [[1, 0]]}
    assert face["landmark_3d_68"] == [[0, 1, -.5]]
    assert face["keypoints"] == [{"x": 0, "y": 1, "score": .7}, [1, 0, .3]]
    assert face["analyses"]["Landmark106"] == [[1, 0]]
    assert face["analyses"]["PIPNet"] == [[0, 1]]
    assert face["analyses"]["FaceMesh"][0]["landmarks"] == [[1, 0, -.25]]
    assert face["raw"]["x_0"] == face["raw"]["mesh_y_2"] == 0
    assert face["raw"]["y_0"] == face["raw"]["mesh_x_2"] == 1
    assert face["raw"]["mesh_z_2"] == -2
    assert [face["raw"][k] for k in ("FaceRectX", "FaceRectY", "FaceRectWidth", "FaceRectHeight")] == pytest.approx([0, .8, .4, .2])
    assert face["facial_area"]["w"] == .4
    assert face["facial_area"]["h"] == pytest.approx(.2)
    assert face["facial_area"]["left_eye"] == [0, 1]
    assert face["attributes"]["region"] == {"x": 0, "y": .5, "w": 1, "h": .5}
    for key in ("pose", "gaze", "embedding"):
        assert face[key] == payload["faces"][0][key]
    assert to_normalized(result) == result
    assert payload["faces"][0]["bbox"][0] == -40  # Inputs are never mutated.
    restored = record_to_pixels(face, (200, 50))
    assert restored["bbox"] == [0, 0, 200, 50]
    assert restored["attributes"] == face["attributes"]  # Still crop-local.


@pytest.mark.parametrize("bbox", [[-50, 10, -1, 80], [401, 10, 450, 80],
                                  [10, -50, 100, -1], [10, 101, 100, 150]])
def test_entirely_out_of_frame_face_has_no_crop(bbox):
    result = to_normalized({"faces": [{"bbox": bbox}]}, (400, 100))
    assert face_contexts_from_records(np.zeros((50, 200, 3), dtype=np.uint8), result["faces"]) == []


def test_percent_upgrade_and_clamping_need_no_original_dimensions():
    doc = SidecarDocument()
    doc.set("face.deepface.coordinates", {"schema": 1, "unit": "percent", "space": "image"})
    doc.set("face.deepface.faces", [{"bbox": [-5, 20, 105, 80], "landmarks": [[25, 150]]}])
    doc.set("face.deepface.processed_at", "original")
    legacy = get_face_section(doc, "deepface")
    from meta_face.coordinates import section_records_in_pixels

    assert section_records_in_pixels(legacy, (200, 100))[0]["bbox"] == [0, 20, 200, 80]
    assert migrate_document(doc)["converted"] == ["deepface"]
    assert doc["face.deepface.coordinates"]["unit"] == "percent"  # Preview only.
    migrate_document(doc, write=True)
    face = doc["face.deepface.faces"][0]
    assert face["bbox"] == [0, .2, 1, .8]
    assert face["landmarks"] == [[.25, 1]]
    assert "face.deepface.image_size" not in doc
    assert doc["face.deepface.processed_at"] == "original"
    assert migrate_document(doc)["already_normalized"] == ["deepface"]
    face["landmarks"] = [[-.5, 1.1]]
    doc.set("face.deepface.faces", [face])
    assert migrate_document(doc, write=True)["converted"] == ["deepface"]
    assert doc["face.deepface.faces"][0]["landmarks"] == [[0, 1]]


def test_mixed_units_and_pyfeat_mesh_landmark_columns():
    payload = {"coordinates": {"unit": "percent"}, "faces": [
        {"bbox": [10, 20, 50, 60]},
        {"coordinates": {"unit": "normalized"}, "bbox": [-1, .2, .8, 2]},
        {"coordinates": {"unit": "pixels"}, "source_image_size": [400, 100],
         "bbox": [100, 20, 200, 60],
         "landmarks": {"mesh_x_0": -5, "mesh_y_0": 110, "mesh_z_0": -3}},
    ]}
    result = to_normalized(payload)
    assert [f["bbox"] for f in result["faces"]] == [[.1, .2, .5, .6], [0, .2, .8, 1], [.25, .2, .5, .6]]
    assert result["faces"][2]["landmarks"] == {"mesh_x_0": 0, "mesh_y_0": 1, "mesh_z_0": -3}
    assert to_normalized(result) == result


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_positions_cannot_be_stored(bad):
    with pytest.raises(ValueError, match="finite"):
        to_normalized({"faces": [{"landmarks": [[bad, 20]]}]}, (400, 100))
    with pytest.raises(ValueError, match="finite"):
        to_normalized({"faces": []}, (bad, 100))


def test_writer_rejects_pixel_geometry_without_dimensions_before_writing():
    doc = SidecarDocument()
    with pytest.raises(ValueError, match="dimensions"):
        write_tool_result(doc, "scrfd", {"faces": [{"bbox": [10, 20, 50, 60]}]})
    assert not doc.entries()
    write_tool_result(doc, "scrfd", {"coordinates": {"unit": "normalized"},
                                     "faces": [{"landmarks": [[-.1, 1.2]]}]})
    assert doc["face.scrfd.faces"][0]["landmarks"] == [[0, 1]]
