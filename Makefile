# simple makefile to simplify repetitive build env management tasks under posix

# caution: testing won't work on windows, see README

PYTHON ?= python

all: clean

clean-pyc:
	find . -name "*.pyc" | xargs rm -f
	find . -name "__pycache__" | xargs rm -rf

clean-so:
	find . -name "*.so" | xargs rm -f
	find . -name "*.pyd" | xargs rm -f

clean-build:
	rm -rf build

clean: clean-build clean-pyc clean-so

install:
	npm install

minify: install
	node build.js
	cp dist/brainsprite.min.js src/brainsprite/data/js/

build: clean minify
	pip install build twine
	python -m build --outdir dist/python
	twine check dist/python/*

.PHONY: src/brainsprite/data/js/brainsprite.js
src/brainsprite/data/js/brainsprite.js:
	cp src/brainsprite.js src/brainsprite/data/js

.PHONY: tests/js/*html
tests/js/*html: src/brainsprite/data/js/brainsprite.js
	tox run -e examples
	cp examples/plot_anat.html tests/js
	cp examples/plot_stat_map.html tests/js
	cp examples/plot_stat_map_radio.html tests/js
	rm -fr src/brainsprite/data/js/brainsprite.js

.PHONY: coverage
coverage: install tests/js/*html
	mkdir -p docs/build/html/_images
	npm run test
	npm i nyc -g
	nyc report --reporter=html
