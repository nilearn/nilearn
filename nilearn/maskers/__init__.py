"""The :mod:`nilearn.maskers` contains masker objects."""

from .base_masker import BaseMasker
from .multi_nifti_labels_masker import MultiNiftiLabelsMasker
from .multi_nifti_maps_masker import MultiNiftiMapsMasker
from .multi_nifti_masker import MultiNiftiMasker
from .nifti_labels_masker import NiftiLabelsMasker
from .nifti_maps_masker import NiftiMapsMasker
from .nifti_masker import NiftiMasker
from .nifti_spheres_masker import NiftiSpheresMasker
from .surface_labels_masker import SurfaceLabelsMasker
from .surface_maps_masker import SurfaceMapsMasker
from .surface_masker import SurfaceMasker

__all__ = [
    "BaseMasker",
    "MultiNiftiLabelsMasker",
    "MultiNiftiMapsMasker",
    "MultiNiftiMasker",
    "NiftiLabelsMasker",
    "NiftiMapsMasker",
    "NiftiMasker",
    "NiftiSpheresMasker",
    "SurfaceLabelsMasker",
    "SurfaceMapsMasker",
    "SurfaceMasker",
]
