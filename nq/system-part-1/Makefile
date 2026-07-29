.PHONY: install test lint type fmt check profile manifest doctor clean

VENV ?= .venv
PY := $(VENV)/bin/python
PYTEST := $(VENV)/bin/pytest
RUFF := $(VENV)/bin/ruff
MYPY := $(VENV)/bin/mypy
RUN := $(PY) -m nqlab.cli.main

install:
	uv venv --python 3.12
	uv pip install -e ".[dev]"
	@SP=$$($(PY) -c "import sysconfig; print(sysconfig.get_paths()['purelib'])"); \
	cp scripts/sitecustomize.py "$$SP/sitecustomize.py"; \
	echo "installed sitecustomize -> $$SP (makes 'nqlab' importable)"

test:
	$(PYTEST)

lint:
	$(RUFF) check src tests

fmt:
	$(RUFF) format src tests
	$(RUFF) check --fix src tests

type:
	$(MYPY)

check: lint type test

profile:
	$(RUN) data profile

manifest:
	$(RUN) data manifest

doctor:
	$(RUN) doctor

clean:
	rm -rf .mypy_cache .pytest_cache .ruff_cache **/__pycache__
