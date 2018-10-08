"""
functional MRI module for NeuroImaging in python
--------------------------------------------------

Documentation is available in the docstrings and online at
http://nistats.github.io.

Contents
--------
Nistats is a Python module for fast and easy functional MRI statistical
analysis.

Submodules
---------
datasets                --- Utilities to download NeuroImaging datasets
hemodynamic_models      --- Hemodyanmic response function specification
design_matrix           --- Design matrix creation for fMRI analysis
experimental_paradigm   --- Experimental paradigm files checks and utils
model                   --- Statistical tests on likelihood models
regression              --- Standard regression models
first_level_model       --- API for first level fMRI model estimation
second_level_model      --- API for second level fMRI model estimation
contrasts               --- API for contrast computation and manipulations
thresholding            --- Utilities for cluster-level statistical results
utils                   --- Miscelaneous utilities
"""

import gzip
import six
import warnings

from .version import _check_module_dependencies, __version__


def py2_deprecation_warning():
    warnings.simplefilter('once')
    py2_warning = ('Python2 support is deprecated and will be removed in '
                   'a future release. Consider switching to Python3.')
    if six.PY2:
        warnings.warn(message=py2_warning,
                      category=DeprecationWarning,
                      stacklevel=4,
                      )

_check_module_dependencies()


__all__ = ['__version__', 'datasets', 'design_matrix']
py2_deprecation_warning()
