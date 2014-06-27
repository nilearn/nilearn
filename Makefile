# simple makefile to simplify repetetive build env management tasks under posix

# caution: testing won't work on windows, see README

PYTHON ?= python
CYTHON ?= cython
NOSETESTS ?= nosetests
CTAGS ?= ctags

all: clean test doc-noplot

clean-pyc:
	find . -name "*.pyc" | xargs rm -f

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
	$(NOSETESTS) -s nilearn 
test-doc:
	$(NOSETESTS) -s --with-doctest --doctest-tests --doctest-extension=rst \
	--doctest-extension=inc --doctest-fixtures=_fixture doc/ \
	

test-coverage:
	rm -rf coverage .coverage
	$(NOSETESTS) -s --with-coverage --cover-html --cover-html-dir=coverage \
	--cover-package=nilearn nilearn

test: test-code test-doc

trailing-spaces:
	find . -name "*.py" | xargs perl -pi -e 's/[ \t]*$$//'

cython:
	find -name "*.pyx" | xargs $(CYTHON)

ctags:
	# make tags for symbol based navigation in emacs and vim
	# Install with: sudo apt-get install exuberant-ctags
	$(CTAGS) -R *

.PHONY : doc
doc:
	make -C doc html

.PHONY : doc-noplot
doc-noplot:
	make -C doc html-noplot

.PHONY : pdf
pdf:
	make -C doc pdf

install: 
	cd doc && make install
 

uml:
	pyreverse -o png -p sparse_models nilearn/sparse_models/estimators.py nilearn/sparse_models/cv.py nilearn/sparse_models/tv.py nilearn/sparse_models/fista.py nilearn/sparse_models/operators.py nilearn/sparse_models/common.py nilearn/sparse_models/prox_tv_l1.py nilearn/sparse_models/smooth_lasso.py nilearn/sparse_models/_cv_tricks.py
	echo "Out put images writen to *_sparse_models.png"