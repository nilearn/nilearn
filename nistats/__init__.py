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
reporting               --- Utilities for creating reports & plotting data
utils                   --- Miscellaneous utilities
"""

import sys
import warnings

from .version import _check_module_dependencies, __version__


def _py2_deprecation_warning():
    py2_warning = ('Python2 support is deprecated and will be removed in '
                   'a future release. Consider switching to Python3.')
    warnings.filterwarnings('once', message=py2_warning)
    warnings.warn(message=py2_warning,
                  category=DeprecationWarning,
                  stacklevel=3,
                  )

def _py34_deprecation_warning():
    py34_warning = ('Python 3.4 support is deprecated and will be removed in '
                   'a future release. Consider switching to Python 3.6 or 3.7.'
                   )
    warnings.filterwarnings('once', message=py34_warning)
    warnings.warn(message=py34_warning,
                  category=DeprecationWarning,
                  stacklevel=3,
                  )


def _python_deprecation_warnings():
    if sys.version_info.major == 2:
        _py2_deprecation_warning()
    elif sys.version_info.major == 3 and sys.version_info.minor == 4:
        _py34_deprecation_warning()


_check_module_dependencies()
_python_deprecation_warnings()

__all__ = ['__version__', 'datasets', 'design_matrix']
