"""
The :mod:`nilearn.decomposition` module includes a subject level
variant of the ICA called Canonical ICA.
"""
from .canica import CanICA
from .multi_pca import session_pca

__all__ = ['CanICA']
