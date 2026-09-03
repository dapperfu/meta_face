"""Read-only, multi-tool sidecar inspection for the directory review notebook."""
from __future__ import annotations

import colorsys
import json
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from sidecar_rs import SidecarDocument

from meta_face.imaging import is_image_path, load_image
from meta_face.coordinates import section_records_in_pixels
from meta_face.sidecar import get_face_section, list_face_tools, sidecar_path_for_media


def inventory(root, recursive=True):
    """Discover images and orphan sidecars; retain missing/error statuses."""
    root = Path(root).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(root)
    paths = sorted(root.rglob('*') if recursive else root.iterdir()) if root.is_dir() else [root]
    images = {p for p in paths if p.is_file() and is_image_path(p)}
    scars = {p for p in paths if p.is_file() and p.suffix.lower() == '.scar'}
    # Discover siblings even when the input is a single .scar file.
    for scar in scars:
        images.update(p for p in scar.parent.iterdir()
                      if p.is_file() and is_image_path(p) and p.stem == scar.stem)
    pairs = [(p, sidecar_path_for_media(p)) for p in sorted(images)]
    used = {scar for _, scar in pairs}
    pairs.extend((None, scar) for scar in sorted(scars - used))
    entries = []
    for media, scar in pairs:
        entry = dict(image=media, sidecar=scar, sections={}, raw={}, status='ok')
        if not scar.exists():
            entry['status'] = 'missing sidecar'
        else:
            try:
                doc = SidecarDocument.from_path(str(scar))
                entry['raw'] = doc.entries()
                entry['sections'] = {t: get_face_section(doc, t) for t in list_face_tools(doc)}
                if media is None:
                    basename = entry['raw'].get('_media.basename')
                    candidate = scar.parent / basename if isinstance(basename, str) else None
                    if candidate and candidate.is_file() and is_image_path(candidate):
                        entry['image'] = candidate
                    else:
                        entry['status'] = 'missing image'
            except Exception as exc:
                entry['status'] = f'sidecar error: {exc}'
        entries.append(entry)
    return entries


def valid_box(value):
    try:
        a = np.asarray(value, dtype=float)
        return (a.shape == (4,) and bool(np.isfinite(a).all())
                and a[2] > a[0] and a[3] > a[1])
    except (TypeError, ValueError):
        return False


def records_for(entry, image_size=None):
    """Preserve every record; link stored analysis indices to their source boxes."""
    sections = entry['sections']
    if image_size is None and entry.get('image') and any(
        'coordinates' in section or 'image_size' in section for section in sections.values()
    ):
        image = load_image(entry['image'])
        image_size = (image.shape[1], image.shape[0])
    if image_size is not None:
        sections = {tool: {**section, 'faces': section_records_in_pixels(section, image_size)}
                    for tool, section in sections.items()}
    records = []
    for tool, section in sections.items():
        faces = section.get('faces', [])
        if not isinstance(faces, list):
            faces = []
        source = section.get('face_index_source') or (
            'dlib_detect' if tool in {'dlib_embed', 'cluster_dlib'} else 'scrfd')
        source_faces = sections.get(source, {}).get('faces', [])
        if not isinstance(source_faces, list):
            source_faces = []
        arrays = {k: section[k] for k in ('embeddings', 'labels', 'probabilities', 'outlier_scores')
                  if isinstance(section.get(k), list)}
        count = max([len(faces)] + [len(v) for v in arrays.values()])
        for index in range(count):
            raw = faces[index] if index < len(faces) else {}
            data = dict(raw) if isinstance(raw, dict) else {'value': raw}
            for key, values in arrays.items():
                if index < len(values):
                    data[key] = values[index]
            face_index = data.get('face_index', index)
            bbox = data.get('bbox')
            inherited = False
            if not valid_box(bbox) and (arrays or 'face_index' in data):
                if isinstance(face_index, int) and 0 <= face_index < len(source_faces):
                    src = source_faces[face_index]
                    bbox = src.get('bbox') if isinstance(src, dict) else None
                    inherited = valid_box(bbox)
            records.append(dict(tool=tool, index=index, face_index=face_index, data=data,
                                bbox=bbox if valid_box(bbox) else None,
                                inherited=inherited, source=source if inherited else tool))
    return records


def tool_colors(entries):
    tools = sorted({t for e in entries for t in e['sections']})
    return {t: colorsys.hsv_to_rgb(i / max(len(tools), 1), .72, .85)
            for i, t in enumerate(tools)}


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0, x2-x1) * max(0, y2-y1)
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - intersection
    return intersection / union if union else 0.0


def overlaps(records, threshold=.3):
    """All cross-tool box pairs above threshold; not identity or consensus claims."""
    if not 0 < threshold <= 1:
        raise ValueError('IoU threshold must be in (0, 1].')
    rows = []
    for a, b in combinations(records, 2):
        if a['tool'] == b['tool'] or a['bbox'] is None or b['bbox'] is None:
            continue
        score = iou(a['bbox'], b['bbox'])
        if score >= threshold:
            rows.append(dict(tool_a=a['tool'], record_a=a['index'], tool_b=b['tool'],
                             record_b=b['index'], iou=score,
                             reused_box=a['inherited'] or b['inherited']))
    return rows


def face_crop_bounds(bbox, image_size, pad_frac=0.3):
    """Pixel crop around a box, padded and clamped to the image."""
    x1, y1, x2, y2 = (float(v) for v in bbox[:4])
    width, height = image_size
    pad = pad_frac * max(x2 - x1, y2 - y1)
    left = max(0, int(x1 - pad))
    top = max(0, int(y1 - pad))
    right = min(int(width), int(np.ceil(x2 + pad)))
    bottom = min(int(height), int(np.ceil(y2 + pad)))
    if right <= left or bottom <= top:
        return 0, 0, int(width), int(height)
    return left, top, right, bottom


def draw_review(entry, colors, selected=None, focus=None, landmarks=True):
    """Draw the full photo, or the actual face pixels when ``focus`` is set."""
    if entry['image'] is None:
        raise ValueError('This sidecar has no image to display.')
    image = load_image(entry['image'])[:, :, ::-1]
    records = [r for r in records_for(entry, (image.shape[1], image.shape[0]))
               if selected is None or r['tool'] in selected]
    origin_x = origin_y = 0
    view = image
    if focus is not None and focus.get('bbox') is not None:
        origin_x, origin_y, right, bottom = face_crop_bounds(
            focus['bbox'], (image.shape[1], image.shape[0])
        )
        view = image[origin_y:bottom, origin_x:right]
    fig, ax = plt.subplots(figsize=(13, 9))
    ax.imshow(view)
    # Thick first, thin last: exact shared boxes appear as nested colored strokes.
    boxed = [r for r in records if r['bbox'] is not None]
    ranks = {tool: i for i, tool in enumerate(sorted({r['tool'] for r in boxed}))}
    for n, r in enumerate(boxed):
        x1, y1, x2, y2 = r['bbox']
        x1, y1, x2, y2 = x1 - origin_x, y1 - origin_y, x2 - origin_x, y2 - origin_y
        color = colors[r['tool']]
        ax.add_patch(Rectangle((x1, y1), x2-x1, y2-y1, fill=False, edgecolor=color,
                               linewidth=1.5 + .7*(len(ranks)-ranks[r['tool']]),
                               linestyle='--' if r['inherited'] else '-'))
        ax.annotate(f"{r['tool']} #{r['index']}", (x1, y1), xytext=(3, -11*(n % 8)),
                    textcoords='offset points', color=color, fontsize=8,
                    bbox=dict(facecolor='white', alpha=.75, edgecolor='none'))
        if landmarks:
            for key in ('kps', 'landmarks', 'landmark_2d_106', 'landmark_3d_68'):
                value = r['data'].get(key)
                try:
                    points = np.asarray(value, dtype=float)
                    if points.ndim == 2 and points.shape[1] >= 2:
                        ax.scatter(points[:, 0] - origin_x, points[:, 1] - origin_y,
                                   s=5, color=color)
                except (ValueError, TypeError):
                    pass
    present = [t for t in entry['sections'] if selected is None or t in selected]
    if present:
        ax.legend(handles=[Line2D([0], [0], color=colors[t], label=t) for t in present],
                  loc='upper left', bbox_to_anchor=(1, 1))
    title = entry['image'].name + ' — solid: stored box; dashed: reused source box'
    if focus is not None and focus.get('bbox') is not None:
        title += f" — face crop #{focus.get('index', focus.get('face_index'))}"
    ax.set_title(title)
    ax.axis('off')
    fig.tight_layout()
    return fig


def show_metadata(value, title='Complete saved metadata'):
    """Expandable, escaped JSON with no truncation (including vectors and arrays)."""
    import html
    from IPython.display import HTML, display
    payload = html.escape(json.dumps(value, indent=2, ensure_ascii=False, default=str))
    display(HTML(f'<details><summary>{html.escape(title)}</summary>'
                 f'<pre style="max-height:600px;overflow:auto">{payload}</pre></details>'))
