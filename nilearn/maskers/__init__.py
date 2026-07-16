"""The :mod:`nilearn.maskers` contains masker objects."""

from nilearn.maskers.base_masker import BaseMasker
from nilearn.maskers.multi_nifti_labels_masker import MultiNiftiLabelsMasker
from nilearn.maskers.multi_nifti_maps_masker import MultiNiftiMapsMasker
from nilearn.maskers.multi_nifti_masker import MultiNiftiMasker
from nilearn.maskers.multi_surface_labels_masker import (
    MultiSurfaceLabelsMasker,
)
from nilearn.maskers.multi_surface_maps_masker import MultiSurfaceMapsMasker
from nilearn.maskers.multi_surface_masker import MultiSurfaceMasker
from nilearn.maskers.nifti_labels_masker import NiftiLabelsMasker
from nilearn.maskers.nifti_maps_masker import NiftiMapsMasker
from nilearn.maskers.nifti_masker import NiftiMasker
from nilearn.maskers.nifti_spheres_masker import NiftiSpheresMasker
from nilearn.maskers.surface_labels_masker import SurfaceLabelsMasker
from nilearn.maskers.surface_maps_masker import SurfaceMapsMasker
from nilearn.maskers.surface_masker import SurfaceMasker

__all__ = [
    "BaseMasker",
    "MultiNiftiLabelsMasker",
    "MultiNiftiMapsMasker",
    "MultiNiftiMasker",
    "MultiSurfaceLabelsMasker",
    "MultiSurfaceMapsMasker",
    "MultiSurfaceMasker",
    "NiftiLabelsMasker",
    "NiftiMapsMasker",
    "NiftiMasker",
    "NiftiSpheresMasker",
    "SurfaceLabelsMasker",
    "SurfaceMapsMasker",
    "SurfaceMasker",
]
