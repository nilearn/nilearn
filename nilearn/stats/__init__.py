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

# __all__ = ['datasets', 'design_matrix']

from .contrasts import (
    compute_contrast,
    compute_fixed_effects,
    Contrast,
    expression_to_contrast_vector,
)
from .model import (
    LikelihoodModelResults,
    TContrastResults,
    FContrastResults,
)
from .regression import (
    OLSModel,
    ARModel,
    RegressionResults,
    SimpleRegressionResults,
)
from .thresholding import (
    fdr_threshold,
    cluster_level_inference,
    map_threshold,
)

__all__ = [
    'compute_contrast',
    'compute_fixed_effects',
    'Contrast',
    'expression_to_contrast_vector',
    'LikelihoodModelResults',
    'TContrastResults',
    'FContrastResults',
    'OLSModel',
    'ARModel',
    'RegressionResults',
    'SimpleRegressionResults',
    'fdr_threshold',
    'cluster_level_inference',
    'map_threshold',
]
