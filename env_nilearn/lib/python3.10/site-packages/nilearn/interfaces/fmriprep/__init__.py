"""The :mod:`nilearn.interfaces.fmriprep` module includes tools to preprocess \
neuroimaging data and access :term:`fMRIPrep` generated confounds.
"""

from .load_confounds import load_confounds
from .load_confounds_strategy import load_confounds_strategy

__all__ = [
    "load_confounds",
    "load_confounds_strategy",
]
