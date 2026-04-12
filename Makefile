PYTHON ?= python3
VENV ?= .venv
PIP := $(VENV)/bin/pip
PY := $(VENV)/bin/python
CONFIG ?= configs/config.yaml

.PHONY: venv install test run-offline run-live run-web service init-db smoke

venv:
	./scripts/setup_venv.sh $(VENV)

install: venv
	./scripts/install_python_deps.sh $(VENV)

test:
	$(PY) -m pytest

run-offline:
	$(PY) -m app.main analyze-dir --config $(CONFIG) sample_audio

run-live:
	$(PY) -m app.main run-live --config $(CONFIG)

run-web:
	$(PY) -m app.main web --config $(CONFIG)

service:
	$(PY) -m app.main service --config $(CONFIG)

init-db:
	$(PY) -m app.main init-db --config $(CONFIG)

smoke:
	./scripts/smoke_test.sh $(CONFIG)

