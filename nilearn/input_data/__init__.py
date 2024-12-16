"""
The :mod:`nilearn.input_data` module used to include masker objects.

It is deprecated since release 0.9.0 in favor of the
:mod:`~nilearn.maskers` module.
Please consider updating your code:
.. code-blocks::python
    from nilearn.input_data import NiftiMasker
becomes:
.. code-blocks::python
    from nilearn.maskers import NiftiMasker
Note that all imports that used to work will continue to do so with
a simple warning at least until release 0.13.0.
"""

import warnings

message = (
    "The import path 'nilearn.input_data' is deprecated in version 0.9. "
    "Importing from 'nilearn.input_data' will be possible at least until "
    "release 0.13.0. Please import from 'nilearn.maskers' instead."
)
warnings.warn(message, DeprecationWarning, stacklevel=1)


from .base_masker import BaseMasker
from .multi_nifti_masker import MultiNiftiMasker
from .nifti_labels_masker import NiftiLabelsMasker
from .nifti_maps_masker import NiftiMapsMasker
from .nifti_masker import NiftiMasker
from .nifti_spheres_masker import NiftiSpheresMasker

__all__ = [
    "BaseMasker",
    "MultiNiftiMasker",
    "NiftiLabelsMasker",
    "NiftiMapsMasker",
    "NiftiMasker",
    "NiftiSpheresMasker",
]
