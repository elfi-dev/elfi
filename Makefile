.PHONY: clean clean-test clean-pyc clean-build docs notebook-docs help
.DEFAULT_GOAL := help
define BROWSER_PYSCRIPT
import os, webbrowser, sys
try:
	from urllib import pathname2url
except:
	from urllib.request import pathname2url

webbrowser.open("file://" + pathname2url(os.path.abspath(sys.argv[1])))
endef
export BROWSER_PYSCRIPT

define PRINT_HELP_PYSCRIPT
import re, sys

for line in sys.stdin:
	match = re.match(r'^([a-zA-Z_-]+):.*?## (.*)$$', line)
	if match:
		target, help = match.groups()
		print("%-20s %s" % (target, help))
endef
export PRINT_HELP_PYSCRIPT
BROWSER := python -c "$$BROWSER_PYSCRIPT"

help:
	@python -c "$$PRINT_HELP_PYSCRIPT" < $(MAKEFILE_LIST)

clean: clean-build clean-pyc clean-test ## remove all build, test, coverage and Python artifacts


clean-build: ## remove build artifacts
	rm -fr build/
	rm -fr dist/
	rm -fr .eggs/
	rm -fr docs/_build
	find . -name '*.egg-info' -exec rm -fr {} +
	find . -name '*.egg' -exec rm -fr {} +

clean-pyc: ## remove Python file artifacts
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +

clean-test: ## remove test and coverage artifacts
	rm -fr .tox/
	rm -f .coverage
	rm -fr htmlcov/

lint: ## check style with flake8
	flake8 elfi tests

test: ## run tests quickly with the default Python
	PYTHONPATH=$$PYTHONPATH:. pytest --reruns 1

test-notslow: ## skips tests marked as slowtest
	PYTHONPATH=$$PYTHONPATH:. pytest -m "not slowtest"

test-all: ## run tests on every Python version with tox
	tox

coverage: ## check code coverage quickly with the default Python
	pytest --cov=elfi
	coverage report -m
	coverage html
	$(BROWSER) htmlcov/index.html

docs: ## generate Sphinx HTML documentation, including API docs
	$(MAKE) -C docs clean
	$(MAKE) notebook-docs
	$(MAKE) -C docs html
	# $(BROWSER) docs/_build/html/index.html

CONTENT_URL := https://raw.githubusercontent.com/elfi-dev/notebooks/dev/figures/

notebook-docs: ## Conver notebooks to rst docs. Assumes you have them in `notebooks` directory.
	jupyter nbconvert --to rst ../notebooks/quickstart.ipynb --output-dir docs
	sed -i '' 's|\(quickstart_files/quickstart.*\.\)|'${CONTENT_URL}'\1|g' docs/quickstart.rst

	jupyter nbconvert --to rst ../notebooks/tutorial.ipynb --output-dir docs/usage
	sed -i '' 's|\(tutorial_files/tutorial.*\.\)|'${CONTENT_URL}'\1|g' docs/usage/tutorial.rst

	jupyter nbconvert --to rst ../notebooks/adaptive_distance.ipynb --output-dir docs/usage
	sed -i '' 's|\(adaptive_distance_files/adaptive_distance.*\.\)|'${CONTENT_URL}'\1|g' docs/usage/adaptive_distance.rst	

	jupyter nbconvert --to rst ../notebooks/adaptive_threshold_selection.ipynb --output-dir docs/usage
	sed -i '' 's|\(adaptive_threshold_selection_files/adaptive_threshold_selection.*\.\)|'${CONTENT_URL}'\1|g' docs/usage/adaptive_threshold_selection.rst		

	jupyter nbconvert --to rst ../notebooks/BOLFI.ipynb --output-dir docs/usage
	sed -i '' 's|\(BOLFI_files/BOLFI.*\.\)|'${CONTENT_URL}'\1|g' docs/usage/BOLFI.rst

	jupyter nbconvert --to rst ../notebooks/parallelization.ipynb --output-dir docs/usage
	sed -i '' 's|\(parallelization_files/parallelization.*\.\)|'${CONTENT_URL}'\1|g' docs/usage/parallelization.rst

	jupyter nbconvert --to rst ../notebooks/non_python_operations.ipynb --output-dir docs/usage --output=external
	sed -i '' 's|\(external_files/external.*\.\)|'${CONTENT_URL}'\1|g' docs/usage/external.rst

# release: clean ## package and upload a release
# 	python setup.py sdist upload
# 	python setup.py bdist_wheel upload

dist: clean ## builds source and wheel package
	python setup.py sdist
	python setup.py bdist_wheel
	ls -l dist

install: clean ## install the package to the active Python's site-packages
	# python setup.py install
	pip install numpy
	pip install -e .

dev: install ## install the development requirements to the active Python's site-packages
	pip install -r requirements-dev.txt

docker-build: ## build a docker image suitable for running ELFI
	docker build --rm -t elfi .

docker: ## run a docker container with a live elfi directory and publish port 8888 for Jupyter
	docker run --rm -v ${PWD}:/elfi -w /elfi -it -p 8888:8888 elfi
	# to run Jupyter from within the container: jupyter notebook --ip 0.0.0.0 --no-browser --allow-root
	# and then from host open page: http://localhost:8888
