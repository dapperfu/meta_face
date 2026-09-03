.PHONY: clean build run test lint format notebook notebook-run notebook-analysis help

PY ?= python3
PIP ?= pip

help:
	@echo "Targets:"
	@echo "  make build        - Install package (pip install -e .)"
	@echo "  make run          - Show mf CLI help"
	@echo "  make test         - Run pytest"
	@echo "  make lint         - Run ruff check"
	@echo "  make format       - Run ruff format"
	@echo "  make notebook          - Launch annotation notebooks in Jupyter"
	@echo "  make notebook-run      - Execute annotation notebooks headlessly"
	@echo "  make notebook-analysis - Launch meta-analysis notebook 20 in Jupyter"
	@echo "  make clean        - Remove build artifacts"

clean:
	rm -rf build dist *.egg-info src/*.egg-info .pytest_cache .ruff_cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

build:
	$(PIP) install -e .

run: build
	mf --help

test:
	pytest

lint:
	ruff check src tests notebooks

format:
	ruff format src tests notebooks

notebook: build
	$(PY) -m jupyter notebook notebooks/01_annotate_overview.ipynb

notebook-run: build
	$(PY) -m jupyter nbconvert --to notebook --execute --inplace notebooks/01_annotate_overview.ipynb
	$(PY) -m jupyter nbconvert --to notebook --execute --inplace notebooks/02_face_crops_buffered.ipynb
	$(PY) -m jupyter nbconvert --to notebook --execute --inplace notebooks/03_face_attributes.ipynb
	$(PY) -m jupyter nbconvert --to notebook --execute --inplace notebooks/04_face_metadata_crops.ipynb

notebook-analysis: build
	$(PY) -m jupyter notebook notebooks/20_collection_overview.ipynb
