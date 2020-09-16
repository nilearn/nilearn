# simple makefile to simplify repetetive build env management tasks under posix

# caution: testing won't work on windows, see README

PYTHON ?= python
CYTHON ?= cython
CTAGS ?= ctags

all: clean test doc-noplot

clean-pyc:
	find . -name "*.pyc" | xargs rm -f
	find . -name "__pycache__" | xargs rm -rf

clean-so:
	find . -name "*.so" | xargs rm -f
	find . -name "*.pyd" | xargs rm -f

clean-build:
	rm -rf build

clean-ctags:
	rm -f tags

clean: clean-build clean-pyc clean-so clean-ctags

in: inplace # just a shortcut
inplace:
	$(PYTHON) setup.py build_ext -i

test-code:
	python -m pytest --pyargs nilearn --cov=nilearn

test-doc:
	pytest --doctest-glob='*.rst' `find doc/ -name '*.rst'`
	pytest doc/_additional_doctests.txt


test-coverage:
	rm -rf coverage .coverage
	pytest --pyargs nilearn --showlocals --cov=nilearn --cov-report=html:coverage

test: test-code test-doc

trailing-spaces:
	find . -name "*.py" | xargs perl -pi -e 's/[ \t]*$$//'

cython:
	find -name "*.pyx" | xargs $(CYTHON)

ctags:
	# make tags for symbol based navigation in emacs and vim
	# Install with: sudo apt-get install exuberant-ctags
	$(CTAGS) -R *

.PHONY : doc-plot
doc-plot:
	make -C doc html

.PHONY : doc
doc:
	make -C doc html-noplot

.PHONY : pdf
pdf:
	make -C doc pdf
