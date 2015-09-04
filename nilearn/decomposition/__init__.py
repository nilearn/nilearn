"""
The :mod:`nilearn.decomposition` module includes a subject level
variant of the ICA called Canonical ICA.
"""
from .canica import CanICA

__all__ = ['CanICA']
