# Makefile for Sphinx documentation
#

#  script to open an html file a browser
define BROWSER_PYSCRIPT
import os, webbrowser, sys

from urllib.request import pathname2url

webbrowser.open("file://" + pathname2url(os.path.abspath(sys.argv[1])))
endef
export BROWSER_PYSCRIPT

BROWSER := python -c "$$BROWSER_PYSCRIPT"

# You can set these variables from the command line.
SPHINXOPTS    = -v --color
SPHINXBUILD   = sphinx-build
PAPER         =
BUILDDIR      = _build

# Internal variables.
PAPEROPT_a4     = -D latex_paper_size=a4
PAPEROPT_letter = -D latex_paper_size=letter
ALLSPHINXOPTS   = -d $(BUILDDIR)/doctrees $(PAPEROPT_$(PAPER)) $(SPHINXOPTS) .

.PHONY: help clean html dirhtml pickle json htmlhelp qthelp latex latexpdf changes linkcheck doctest

all: html-noplot

help:
	@echo "Please use \`make <target>' where <target> is one of"
	@echo "  html      to make standalone HTML files"
	@echo "  html-noplot to make standalone HTML files, without plotting anything"
	@echo "  ci-html-noplot to run html-noplot with warnings as errors"
	@echo "  check     to do a sanity check of the documentation"
	@echo "  dirhtml   to make HTML files named index.html in directories"
	@echo "  pickle    to make pickle files"
	@echo "  json      to make JSON files"
	@echo "  htmlhelp  to make HTML files and a HTML help project"
	@echo "  qthelp    to make HTML files and a qthelp project"
	@echo "  latex     to make LaTeX files, you can set PAPER=a4 or PAPER=letter"
	@echo "  latexpdf  to make LaTeX files and run them through pdflatex"
	@echo "  changes   to make an overview of all changed/added/deprecated items"
	@echo "  linkcheck to check all external links for integrity"
	@echo "  doctest   to run all doctests embedded in the documentation (if enabled)"

clean:
	-rm -rf $(BUILDDIR)/*
	-rm -rf auto_examples/
	-rm -rf generated/*
	-rm -rf modules/generated/*
	-rm -rf ../examples/**/nilearn_cache
	-rm -rf ../**/nilearn_cache
	-rm -rf modules/description/

glm_reports:
	python visual_testing/reporter_visual_inspection_suite.py

sym_links_datasets:
	# copy datasets RST from nilearn package to doc folder
	mkdir -p modules/description
	cd ../nilearn/datasets/description && cp -l *.rst ../../../doc/modules/description

sym_links:
	# Make sym-links to share the cache across various example directories
	# The following find command selects all the depth-one
	# subdirectories of ../examples/ and creates a sym-link to
	# ../examples/nilearn_cache/
	-find ../examples/ -type d -mindepth 1 -maxdepth 1 -not -path "../examples/nilearn_cache" -exec ln -sf ../nilearn_cache "{}/nilearn_cache" \;
	-mkdir -p ../examples/nilearn_cache/joblib

force_html: force html

force:
	find . -name \*.rst -exec touch {} \;

# view doc
view: _build/html/index.html
	$(BROWSER) _build/html/index.html

no_jekyll:
	mkdir -p $(BUILDDIR)/html
	touch $(BUILDDIR)/html/.nojekyll

html: no_jekyll	sym_links sym_links_datasets glm_reports
	# These two lines make the build a bit more lengthy, and the
	# the embedding of images more robust
	rm -rf $(BUILDDIR)/html/_images
	$(SPHINXBUILD) -b html $(ALLSPHINXOPTS) $(BUILDDIR)/html
	@echo
	@echo "Build finished. The HTML pages are in $(BUILDDIR)/html."

html-strict: no_jekyll sym_links sym_links_datasets glm_reports
	# Build html documentation using a strict mode: Warnings are
	# considered as errors.
	make check
	@echo
	@echo "Build finished. The HTML pages are in $(BUILDDIR)/html."

html-modified-examples-only: sym_links sym_links_datasets glm_reports
	BUILD_DEV_HTML=1 $(SPHINXBUILD) -W --keep-going -D sphinx_gallery_conf.filename_pattern=$(PATTERN) -D sphinx_gallery_conf.run_stale_examples=True -b html $(ALLSPHINXOPTS) $(BUILDDIR)/html
	@echo
	@echo "Build finished. The HTML pages are in $(BUILDDIR)/html."

html-noplot: no_jekyll sym_links_datasets glm_reports
	BUILD_DEV_HTML=1 $(SPHINXBUILD) -D plot_gallery=0 -b html $(ALLSPHINXOPTS) $(BUILDDIR)/html
	@echo
	@echo "Build finished. The HTML pages are in $(BUILDDIR)/html."

html-noplot-noreport: sym_links_datasets
	BUILD_DEV_HTML=1 $(SPHINXBUILD) -D plot_gallery=0 -b html $(ALLSPHINXOPTS) $(BUILDDIR)/html
	@echo
	@echo "Build finished. The HTML pages are in $(BUILDDIR)/html."

ci-html-noplot: sym_links_datasets glm_reports
	BUILD_DEV_HTML=1 $(SPHINXBUILD) -W --keep-going -D plot_gallery=0 -b html $(ALLSPHINXOPTS) $(BUILDDIR)/html
	@echo
	@echo "Build finished. The HTML pages are in $(BUILDDIR)/html."

dirhtml: no_jekyll
	$(SPHINXBUILD) -b dirhtml $(ALLSPHINXOPTS) $(BUILDDIR)/dirhtml
	@echo
	@echo "Build finished. The HTML pages are in $(BUILDDIR)/dirhtml."

pickle:
	$(SPHINXBUILD) -b pickle $(ALLSPHINXOPTS) $(BUILDDIR)/pickle
	@echo
	@echo "Build finished; now you can process the pickle files."

json:
	$(SPHINXBUILD) -b json $(ALLSPHINXOPTS) $(BUILDDIR)/json
	@echo
	@echo "Build finished; now you can process the JSON files."

htmlhelp:
	$(SPHINXBUILD) -b htmlhelp $(ALLSPHINXOPTS) $(BUILDDIR)/htmlhelp
	@echo
	@echo "Build finished; now you can run HTML Help Workshop with the" \
	      ".hhp project file in $(BUILDDIR)/htmlhelp."

qthelp:
	$(SPHINXBUILD) -b qthelp $(ALLSPHINXOPTS) $(BUILDDIR)/qthelp
	@echo
	@echo "Build finished; now you can run "qcollectiongenerator" with the" \
	      ".qhcp project file in $(BUILDDIR)/qthelp, like this:"
	@echo "# qcollectiongenerator $(BUILDDIR)/qthelp/nilearn.qhcp"
	@echo "To view the help file:"
	@echo "# assistant -collectionFile $(BUILDDIR)/qthelp/nilearn.qhc"

latex:
	$(SPHINXBUILD) -b latex $(ALLSPHINXOPTS) $(BUILDDIR)/latex
	@echo
	@echo "Build finished; the LaTeX files are in $(BUILDDIR)/latex."
	@echo "Run \`make' in that directory to run these through (pdf)latex" \
	      "(use \`make latexpdf' here to do that automatically)."

latexpdf:
	$(SPHINXBUILD) -b latex $(ALLSPHINXOPTS) $(BUILDDIR)/latex
	@echo "Running LaTeX files through pdflatex..."
	make -C $(BUILDDIR)/latex all-pdf
	@echo "pdflatex finished; the PDF files are in $(BUILDDIR)/latex."

changes:
	$(SPHINXBUILD) -b changes $(ALLSPHINXOPTS) $(BUILDDIR)/changes
	@echo
	@echo "The overview file is in $(BUILDDIR)/changes."

linkcheck:
	$(SPHINXBUILD) -b linkcheck $(ALLSPHINXOPTS) $(BUILDDIR)/linkcheck
	@echo
	@echo "Link check complete; look for any errors in the above output " \
	      "or in $(BUILDDIR)/linkcheck/output.txt."

doctest:
	$(SPHINXBUILD) -b doctest $(ALLSPHINXOPTS) $(BUILDDIR)/doctest
	@echo "Testing of doctests in the sources finished, look at the " \
	      "results in $(BUILDDIR)/doctest/output.txt."

zip: html
	mkdir -p _build/nilearn ;
	cp -r _build/html _build/nilearn ;
	zip -r _build/nilearn.zip _build/nilearn

check:
	rm -rf _build/doctrees
	rm -rf $(BUILDDIR)/html/_images
	BUILD_DEV_HTML=1 $(SPHINXBUILD) -W --keep-going -T -n -b html $(ALLSPHINXOPTS) $(BUILDDIR)/html

install: sym_links_datasets glm_reports
	@echo $(VERSIONTAG)
	rm -rf _build/doctrees _build/nilearn.github.io
	# first clone the nilearn.github.io repo because it may ask
	# for password and we don't want to delay this long build in
	# the middle of it
	# --depth 1 is a speed optimization since we don't need the
	# history prior to the last commit
	git clone --depth 1 git@github.com:nilearn/nilearn.github.io.git _build/nilearn.github.io
	touch _build/nilearn.github.io/.nojekyll
	rm -Rf _build/nilearn.github.io/stable;
	mkdir _build/nilearn.github.io/stable;
	make html
	cd _build/ && \
	cp -r html/* nilearn.github.io/stable/ && \
	cd nilearn.github.io && \
	git add -A && \
	git add .nojekyll && \
	git commit -a -m 'Make install' && \
	git tag $(VERSIONTAG) && \
	git push --tags
