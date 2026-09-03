# Annotation Notebooks Directory

## Goal

Add a `notebooks/` directory with sequentially numbered annotation example notebooks, backed by bbox buffer/crop utilities in the library and notebook display helpers. Users pass an image path and see original vs annotated full images plus per-face crops with configurable per-side buffer %.

## Confirmed decisions

- **Naming:** sequential prefixes (`01_*`, `02_*`, …)
- **Buffer:** per-side expansion — `buffer_pct=10` adds 10% of bbox width to left and right, 10% of height to top and bottom, clamped to image bounds
- **Policy vs mechanism:** bbox math in `src/meta_face/bbox.py`; matplotlib display in `notebooks/_utils.py`

## Layout

```
meta_face/
├── src/meta_face/bbox.py          # expand_bbox, crop_face
├── tests/test_bbox.py
├── notebooks/
│   ├── _utils.py                  # matplotlib display helpers
│   ├── 01_annotate_overview.ipynb
│   └── 02_face_crops_buffered.ipynb
├── Makefile                       # notebook / notebook-run targets
└── pyproject.toml                 # dev extras: jupyter, matplotlib, etc.
```

## Implementation Status

- [x] Plan saved
- [x] bbox module + tests
- [x] notebooks/_utils.py
- [x] 01_annotate_overview.ipynb
- [x] 02_face_crops_buffered.ipynb
- [x] pyproject.toml, .gitignore, Makefile, README

## Verification

1. `pytest tests/test_bbox.py` passes (no GPU)
2. `ruff check` / `ruff format` on new Python files
3. Manual: notebooks render original + annotated and buffered crops
4. `make notebook-run` executes both notebooks headlessly (requires GPU env)
