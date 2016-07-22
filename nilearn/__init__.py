"""
Machine Learning module for NeuroImaging in python
--------------------------------------------------

Documentation is available in the docstrings and online at
http://nilearn.github.io.

Contents
--------
Nilearn aims at simplifying the use of the scikit-learn package in the context of
neuroimaging. It provides specific input/output functions, algorithms and
visualization tools.

Submodules
---------
datasets                --- Utilities to download NeuroImaging datasets
decoding                --- Decoding tools and algorithms
decomposition           --- Includes a subject level variant of the ICA
                            algorithm called Canonical ICA
connectome              --- Set of tools for computing functional connectivity matrices
                            and for sparse multi-subjects learning of Gaussian graphical models
image                   --- Set of functions defining mathematical operations
                            working on Niimg-like objects
input_data              --- includes scikit-learn tranformers and tools to
                            preprocess neuro-imaging data
masking                 --- Utilities to compute and operate on brain masks
mass_univariate         --- Defines a Massively Univariate Linear Model
                            estimated with OLS and permutation test
plotting                --- Plotting code for nilearn
region                  --- Set of functions for extracting region-defined
                            signals
signal                  --- Set of preprocessing functions for time series
"""

import gzip

from .version import _check_module_dependencies, __version__

_check_module_dependencies()

# Monkey-patch gzip to have faster reads on large gzip files
if hasattr(gzip.GzipFile, 'max_read_chunk'):
    gzip.GzipFile.max_read_chunk = 100 * 1024 * 1024  # 100Mb

# Boolean controlling the default globbing technique when using check_niimg
# and the os.path.expanduser usage in CacheMixin.
# Default value it True, set it to False to completely deactivate this
# behavior.
EXPAND_PATH_WILDCARDS = True

# Boolean controlling whether the joblib caches should be
# flushed if the version of certain modules changes (eg nibabel, as it
# does not respect the backward compatibility in some of its internal
# structures
# This  is used in nilearn._utils.cache_mixin
CHECK_CACHE_VERSION = True

# list all submodules available in nilearn and version
__all__ = ['datasets', 'decoding', 'decomposition', 'connectome',
           'image', 'input_data', 'masking', 'mass_univariate', 'plotting',
           'region', 'signal', '__version__']
