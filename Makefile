.PHONY: setup setup-dev test compile check cpp-worker

PYTHON ?= python3
PIP_INSTALL_FLAGS ?= --disable-pip-version-check

setup:
	$(PYTHON) -m pip install $(PIP_INSTALL_FLAGS) -r requirements.txt

setup-dev:
	$(PYTHON) -m pip install $(PIP_INSTALL_FLAGS) -r requirements-dev.txt

test:
	$(PYTHON) -m pytest

compile:
	$(PYTHON) -m compileall -q express_derm_ml

check: test compile

cpp-worker:
	./scripts/build_cpp_ai_worker.sh
