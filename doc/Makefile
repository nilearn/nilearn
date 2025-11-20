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

.PHONY: help clean html dirhtml pickle json htmlhelp qthelp latex latexpdf changes linkcheck doctest reports dummy_reports

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
	-rm -rf modules/generated_reports/*.html

sym_links_datasets:
	# copy datasets RST from nilearn package to doc folder
	-rm -rf modules/description/
	mkdir -p modules/description
	cd ../nilearn/datasets/description && cp -l *.rst ../../../doc/modules/description

sym_links:
	# Make sym-links to share the cache across various example directories
	# The following find command selects all the depth-one
	# subdirectories of ../examples/ and creates a sym-link to
	# ../examples/nilearn_cache/
	-find ../examples/ -mindepth 1 -maxdepth 1 -type d -not -path "../examples/nilearn_cache" -exec ln -sf ../nilearn_cache "{}/nilearn_cache" \;
	-mkdir -p ../examples/nilearn_cache/joblib

.PHONY: ci.rst
ci.rst:
	python get_ci_workflows_description.py

force_html: force html

force:
	find . -name \*.rst -exec touch {} \;

# view doc
view: _build/html/index.html
	$(BROWSER) _build/html/index.html

dummy_reports:
	mkdir -p _build/html/
	marimo export html  visual_testing/notebook_reports_maskers.py -- --build_type=partial > reports/notebook_reports_maskers.html
	marimo export html  visual_testing/notebook_reports_glm.py -- --build_type=partial > reports/notebook_reports_glm.html

reports:
	marimo export html  visual_testing/notebook_reports_maskers.py -- --build_type=full > reports/notebook_reports_maskers.html
	marimo export html  visual_testing/notebook_reports_glm.py -- --build_type=full > reports/notebook_reports_glm.html


html: ci.rst sym_links sym_links_datasets reports
	rm -rf $(BUILDDIR)/html/_images
	$(SPHINXBUILD) -b html $(ALLSPHINXOPTS) $(BUILDDIR)/html
	@echo
	@echo "Build finished. The HTML pages are in $(BUILDDIR)/html."

html-strict: ci.rst sym_links sym_links_datasets reports
	# Build html documentation using a strict mode: Warnings are
	# considered as errors.
	make check
	@echo
	@echo "Build finished. The HTML pages are in $(BUILDDIR)/html."

html-modified-examples-only-reports: ci.rst sym_links sym_links_datasets reports
	BUILD_DEV_HTML=1 $(SPHINXBUILD) -W --keep-going -D sphinx_gallery_conf.filename_pattern=$(PATTERN) -D sphinx_gallery_conf.run_stale_examples=True -b html $(ALLSPHINXOPTS) $(BUILDDIR)/html
	@echo
	@echo "Build finished. The HTML pages are in $(BUILDDIR)/html."

html-modified-examples-only: ci.rst sym_links sym_links_datasets dummy_reports
	BUILD_DEV_HTML=1 $(SPHINXBUILD) -W --keep-going -D sphinx_gallery_conf.filename_pattern=$(PATTERN) -D sphinx_gallery_conf.run_stale_examples=True -b html $(ALLSPHINXOPTS) $(BUILDDIR)/html
	@echo
	@echo "Build finished. The HTML pages are in $(BUILDDIR)/html."

html-noplot-reports: ci.rst sym_links_datasets reports
	BUILD_DEV_HTML=1 $(SPHINXBUILD) -D plot_gallery=0 -b html $(ALLSPHINXOPTS) $(BUILDDIR)/html
	@echo
	@echo "Build finished. The HTML pages are in $(BUILDDIR)/html."

html-noplot: ci.rst sym_links_datasets dummy_reports
	BUILD_DEV_HTML=1 $(SPHINXBUILD) -W --keep-going -D plot_gallery=0 -b html $(ALLSPHINXOPTS) $(BUILDDIR)/html
	@echo
	@echo "Build finished. The HTML pages are in $(BUILDDIR)/html."

ci-html-noplot-reports: ci.rst sym_links_datasets reports
	BUILD_DEV_HTML=1 $(SPHINXBUILD) -W --keep-going -D plot_gallery=0 -b html $(ALLSPHINXOPTS) $(BUILDDIR)/html
	@echo
	@echo "Build finished. The HTML pages are in $(BUILDDIR)/html."

ci-html-noplot: ci.rst sym_links_datasets dummy_reports
	BUILD_DEV_HTML=1 $(SPHINXBUILD) -W --keep-going -D plot_gallery=0 -b html $(ALLSPHINXOPTS) $(BUILDDIR)/html
	@echo
	@echo "Build finished. The HTML pages are in $(BUILDDIR)/html."

dirhtml:
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

changes:
	$(SPHINXBUILD) -b changes $(ALLSPHINXOPTS) $(BUILDDIR)/changes
	@echo
	@echo "The overview file is in $(BUILDDIR)/changes."

linkcheck: sym_links sym_links_datasets dummy_reports
	$(SPHINXBUILD) -D plot_gallery=0 -b linkcheck $(ALLSPHINXOPTS) $(BUILDDIR)/linkcheck
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
