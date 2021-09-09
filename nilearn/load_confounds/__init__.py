"""
Loading fMRIprep confounds into python.

Note that the nilearn.load_confounds module is experimental.
It may change in any future (>0.8.0) release of Nilearn.
"""
from warnings import warn
from .parser import Confounds


__all__ = ["Confounds"]

warn('The nilearn.koad_confounds module is experimental. '
     'It may change in any future release of Nilearn.', FutureWarning)
