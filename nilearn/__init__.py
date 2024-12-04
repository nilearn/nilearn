"""
Machine Learning module for NeuroImaging in python.
---------------------------------------------------

Documentation is available in the docstrings and online at
https://nilearn.github.io.

Contents
--------

Nilearn aims at simplifying the use of the scikit-learn package
in the context of neuroimaging.
It provides specific input/output functions, algorithms and
visualization tools.

Submodules
---------

connectome              --- Set of tools for computing functional
                            connectivity matrices
                            and for sparse multi-subjects learning
                            of Gaussian graphical models
datasets                --- Utilities to download NeuroImaging datasets
decoding                --- Decoding tools and algorithms
decomposition           --- Includes a subject level variant of the ICA
                            algorithm called Canonical ICA
glm                     --- Analyzing fMRI data using GLMs
image                   --- Set of functions defining mathematical operations
                            working on Niimg-like objects
maskers                 --- Includes scikit-learn transformers.
masking                 --- Utilities to compute and operate on brain masks
interfaces              --- Includes tools to preprocess neuro-imaging data
                            from various common interfaces like fMRIPrep.
mass_univariate         --- Defines a Massively Univariate Linear Model
                            estimated with OLS and permutation test
plotting                --- Plotting code for nilearn
region                  --- Set of functions for extracting region-defined
                            signals, clustering methods,
                            connected regions extraction
reporting               --- Implements functions useful
                            to report analysis results
signal                  --- Set of preprocessing functions for time series
"""

import gzip

try:
    from ._version import __version__
except ImportError:
    __version__ = "0+unknown"


# Monkey-patch gzip to have faster reads on large gzip files
if hasattr(gzip.GzipFile, "max_read_chunk"):
    gzip.GzipFile.max_read_chunk = 100 * 1024 * 1024  # 100Mb

# Boolean controlling the default globbing technique when using check_niimg
# and the os.path.expanduser usage in CacheMixin.
# Default value it True, set it to False to completely deactivate this
# behavior.
EXPAND_PATH_WILDCARDS = True

# list all submodules available in nilearn and version
__all__ = [
    "__version__",
    "connectome",
    "datasets",
    "decoding",
    "decomposition",
    "image",
    "interfaces",
    "maskers",
    "masking",
    "mass_univariate",
    "plotting",
    "regions",
    "signal",
    "surface",
]
