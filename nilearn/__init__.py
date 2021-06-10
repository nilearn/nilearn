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
                            signals, clustering methods, connected regions extraction
signal                  --- Set of preprocessing functions for time series
"""

import gzip
import os
import pkg_resources
import warnings

from distutils.version import LooseVersion

from .version import _check_module_dependencies, __version__

# Workaround issue discovered in intel-openmp 2019.5:
# https://github.com/ContinuumIO/anaconda-issues/issues/11294
#
# see also https://github.com/scikit-learn/scikit-learn/pull/15020
os.environ.setdefault("KMP_INIT_AT_FORK", "FALSE")


def _nibabel2_deprecation_warning():
    msg = ('Support for Nibabel 2.x is deprecated and will stop '
           'in release 0.9.0. Please consider upgrading to '
           'Nibabel 3.x.')
    warnings.filterwarnings('once', message=msg)
    warnings.warn(message=msg,
                  category=FutureWarning,
                  stacklevel=3)


def _nibabel_deprecation_warnings():
    """Give a deprecation warning is the version of
    Nibabel is < 3.0.0.
    """
    # Nibabel should be installed or we would
    # have had an error when calling
    # _check_module_dependencies
    dist = pkg_resources.get_distribution('nibabel')
    nib_version = LooseVersion(dist.version)
    if nib_version < '3.0':
        _nibabel2_deprecation_warning()


_check_module_dependencies()
_nibabel_deprecation_warnings()

# Temporary work around to address formatting issues in doc tests
# with NumPy 1.14. NumPy had made more consistent str/repr formatting
# of numpy arrays. Hence we print the options to old versions.
import numpy as np
if LooseVersion(np.__version__) >= LooseVersion("1.14"):
    # See issue #1600 in nilearn for reason to add try and except
    try:
        from ._utils.testing import are_tests_running
        if are_tests_running():
            np.set_printoptions(legacy='1.13')
    except ImportError:
        pass

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
           'regions', 'signal', 'stats', 'surface',
           'parcellations', '__version__']

