"""The :mod:`nilearn.decomposition` module includes a subject level \
variant of the :term:`ICA` called Canonical :term:`ICA`.
"""

from nilearn.decomposition.canica import CanICA
from nilearn.decomposition.dict_learning import DictLearning

__all__ = ["CanICA", "DictLearning"]
