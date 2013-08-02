"""
The :mod:`nilearn.input_data` module includes scikit-learn tranformers and
tools to preprocess neuro-imaging data.
"""
from .nifti_masker import NiftiMasker
from .multi_nifti_masker import MultiNiftiMasker
from .nifti_region import NiftiLabelsMasker, NiftiMapsMasker
