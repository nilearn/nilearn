"""
The :mod:`nilearn.decomposition` module includes a subject level
variant of the ICA called Canonnical ICA.
"""
from .canica import CanICA
from .dict_learning import DictLearning

__all__ = ['CanICA', 'DictLearning']
