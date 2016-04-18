"""
The :mod:`nilearn.input_data` module includes scikit-learn tranformers and
tools to preprocess neuro-imaging data.
"""
from .nifti_masker import NiftiMasker
from .multi_nifti_masker import MultiNiftiMasker
from .nifti_labels_masker import NiftiLabelsMasker
from .nifti_maps_masker import NiftiMapsMasker
from .nifti_spheres_masker import NiftiSpheresMasker

__all__ = ['NiftiMasker', 'MultiNiftiMasker', 'NiftiLabelsMasker',
           'NiftiMapsMasker', 'NiftiSpheresMasker']
