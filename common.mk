PYTHON_INTERPRETER = python3.9
VIRTUALENV = .venv
PYTHON = .venv/bin/python
PIP = .venv/bin/pip
MYPY = $(VIRTUALENV)/bin/mypy
FLAKE = $(VIRTUALENV)/bin/flake8

PIP_INSTALL = ${PYTHON} -m pip install --cache-dir ../pip-cache

DIST=dist

.PHONY: build

.venv:
	virtualenv --python $(PYTHON_INTERPRETER) .venv

.dependencies: .venv
	$(PIP_INSTALL) -r requirements.txt
	touch .dependencies

build: .dependencies test static-analysis
	# This will create a python wheel
	# wheel: https://packaging.python.org/tutorials/packaging-projects/
	$(PYTHON) setup.py bdist_wheel --dist-dir ${DIST}
	mkdir -p logs

build-dev: .dependencies static-analysis
	# This will create a python wheel
	# wheel: https://packaging.python.org/tutorials/packaging-projects/
	$(PYTHON) setup.py develop

test: .dependencies
	$(PYTHON) setup.py test

static-analysis: .dependencies
	$(FLAKE) ${SOURCE_PATH} --config ../check.ini
	$(MYPY) --config-file ../check.ini $(SOURCE_PATH)

install: .dependencies test static-analysis
	$(PYTHON) setup.py install

clean:
	rm -rf ${VIRTUALENV}
	rm -rf ${DIST}
	rm -rf build
	rm -rf *.egg-info
	rm -f .dependencies
