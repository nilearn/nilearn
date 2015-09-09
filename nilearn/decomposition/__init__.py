"""
The :mod:`nilearn.decomposition` module includes a subject level
variant of the ICA called Canonical ICA.
"""
from .canica import CanICA
from .dict_learning import DictLearning
from .sparse_pca import SparsePCA

__all__ = ['CanICA', 'DictLearning', 'SparsePCA']
