"""
Analysing fMRI data using GLMs.

Note that the nilearn.glm module is experimental.
     It may change in any future (>0.7.0) release of Nilearn.
"""
from warnings import warn
from nilearn.glm.contrasts import (
    compute_contrast,
    compute_fixed_effects,
    Contrast,
    expression_to_contrast_vector,
)
from nilearn.glm.model import (
    LikelihoodModelResults,
    TContrastResults,
    FContrastResults,
)
from nilearn.glm.regression import (
    OLSModel,
    ARModel,
    RegressionResults,
    SimpleRegressionResults,
)
from nilearn.glm.thresholding import (
    fdr_threshold,
    cluster_level_inference,
    threshold_stats_img,
)

from nilearn.glm import first_level
from nilearn.glm import second_level

__all__ = [
    'first_level',
    'second_level',
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
    'threshold_stats_img',
    'first_level',
    'second_level',
]

warn('The nilearn.glm module is experimental. '
     'It may change in any future release of Nilearn.', FutureWarning)
