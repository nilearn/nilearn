"""
The :mod:`nilearn.maskers` contains masker objects.
"""

from .base_masker import BaseMasker
from .nifti_masker import NiftiMasker
from .multi_nifti_masker import MultiNiftiMasker
from .nifti_labels_masker import NiftiLabelsMasker
from .nifti_maps_masker import NiftiMapsMasker
from .nifti_spheres_masker import NiftiSpheresMasker


__all__ = ['BaseMasker', 'NiftiMasker', 'MultiNiftiMasker',
           'NiftiLabelsMasker', 'NiftiMapsMasker', 'NiftiSpheresMasker']
