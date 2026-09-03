"""Verify sidecar associations and notebook execution without model inference."""
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('tool_review', ROOT / 'notebooks/_tool_review.py')
review = importlib.util.module_from_spec(spec)
spec.loader.exec_module(review)


def test_sparse_indices_and_independent_boxes():
    entry = {'sections': {
        'scrfd': {'faces': [{'bbox': [0, 0, 10, 10]}, {'bbox': [20, 20, 30, 30]}]},
        'emotion': {'faces': [{'face_index': 1, 'emotion': 'happy'}]},
        'dlib_detect': {'faces': [{'bbox': [1, 1, 11, 11]}]},
        'cluster_dlib': {'labels': [7]},
        'unknown': {'faces': [{'face_index': 99, 'error': 'missing'}]},
    }}
    records = review.records_for(entry)
    emotion = next(r for r in records if r['tool'] == 'emotion')
    assert emotion['bbox'] == [20, 20, 30, 30]
    assert emotion['inherited']
    cluster = next(r for r in records if r['tool'] == 'cluster_dlib')
    assert cluster['bbox'] == [1, 1, 11, 11]
    assert next(r for r in records if r['tool'] == 'unknown')['bbox'] is None
    pairs = review.overlaps(records)
    independent = next(p for p in pairs if p['tool_a'] == 'scrfd'
                       and p['tool_b'] == 'dlib_detect')
    assert independent['iou'] == pytest.approx(81 / 119)
    assert not independent['reused_box']
    assert not review.valid_box([0, 0, float('nan'), 10])


def test_focused_review_shows_face_pixels_not_full_frame(tmp_path, monkeypatch):
    import matplotlib
    matplotlib.use('Agg')
    import numpy as np
    from PIL import Image

    photo = tmp_path / 'faces.jpg'
    canvas = np.zeros((80, 120, 3), dtype=np.uint8)
    canvas[20:50, 40:70] = (0, 200, 40)
    Image.fromarray(canvas).save(photo)
    entry = {
        'image': photo,
        'sections': {'scrfd': {'faces': [{'bbox': [40, 20, 70, 50], 'kps': [[50, 30]]}]}},
    }
    focus = {'tool': 'scrfd', 'index': 0, 'bbox': [40, 20, 70, 50],
             'data': {'kps': [[50, 30]]}, 'inherited': False}
    monkeypatch.setattr(review, 'load_image', lambda path: np.array(Image.open(path))[:, :, ::-1])
    fig = review.draw_review(entry, {'scrfd': (1, 0, 0)}, focus=focus)
    shown = fig.axes[0].images[0].get_array()
    left, top, right, bottom = review.face_crop_bounds([40, 20, 70, 50], (120, 80))
    assert shown.shape[0] == bottom - top
    assert shown.shape[1] == right - left
    assert shown.shape[0] < 80
    assert shown.shape[1] < 120
    assert shown[..., 1].max() >= 200
    import matplotlib.pyplot as plt
    plt.close(fig)


def test_notebook_end_to_end(tmp_path, monkeypatch):
    import re
    import nbformat
    import numpy as np
    from PIL import Image
    from meta_face.sidecar import update_sidecar, write_tool_result

    inputs = tmp_path / 'inputs'
    inputs.mkdir()
    photo = inputs / 'team.001.JPG'
    Image.fromarray(np.full((100, 120, 3), 180, dtype=np.uint8)).save(photo)
    Image.new('RGB', (20, 20)).save(inputs / 'missing.png')
    (inputs / 'corrupt.scar').write_bytes(b'invalid sidecar')

    def patch(doc):
        write_tool_result(doc, 'scrfd', {'faces': [
            {'bbox': [10, 10, 50, 65], 'kps': [[20, 25], [40, 25]], 'age': 30}]}, image_size=(120, 100))
        write_tool_result(doc, 'dlib_detect', {'faces': [{'bbox': [12, 12, 51, 65]}]}, image_size=(120, 100))
        write_tool_result(doc, 'emotion', {'faces': [{'face_index': 0, 'scores': {'happy': .8}}]})
        write_tool_result(doc, 'arcface', {'embeddings': [[.1, .2, .3]]})
    update_sidecar(photo, patch)
    update_sidecar(inputs / 'orphan.jpg', patch)
    entries = review.inventory(inputs)
    assert len(entries) == 4
    assert {'ok', 'missing image', 'missing sidecar'} <= {e['status'] for e in entries}
    assert any(e['status'].startswith('sidecar error') for e in entries)
    assert review.inventory(photo.with_suffix('.scar'))[0]['image'] == photo
    before = {p: p.read_bytes() for p in inputs.iterdir()}
    nb = nbformat.read(ROOT / 'notebooks/05_directory_tool_review.ipynb', as_version=4)
    nbformat.validate(nb)
    monkeypatch.chdir(ROOT)
    env = {}
    image_index = next(i for i, e in enumerate(entries) if e['image'] == photo)
    for cell in nb.cells:
        if cell.cell_type != 'code':
            continue
        # Override settings by assignment, independent of a user's notebook defaults.
        source = re.sub(r'(?m)^ROOT = .*$', lambda _: f'ROOT = {inputs!r}', cell.source)
        source = re.sub(r'(?m)^IMAGE_INDEX = .*$', lambda _: f'IMAGE_INDEX = {image_index}', source)
        source = re.sub(r'(?m)^EXPORT_DIR = .*$',
                        lambda _: f'EXPORT_DIR = {str(tmp_path / "export")!r}', source)
        env['PosixPath'] = Path
        exec(compile(source, '<notebook>', 'exec'), env)
    assert list((tmp_path / 'export').glob('*_overlay.png'))
    assert (tmp_path / 'export/overlapping_tools.csv').exists()
    assert before == {p: p.read_bytes() for p in inputs.iterdir()}
