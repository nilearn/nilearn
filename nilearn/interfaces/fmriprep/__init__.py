"""The :mod:`nilearn.interfaces.fmriprep` module includes tools to preprocess \
neuroimaging data and access :term:`fMRIPrep` generated confounds.
"""

from nilearn.interfaces.fmriprep.load_confounds import load_confounds
from nilearn.interfaces.fmriprep.load_confounds_strategy import (
    load_confounds_strategy,
)

__all__ = [
    "load_confounds",
    "load_confounds_strategy",
]
