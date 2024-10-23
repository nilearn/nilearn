"""Kept as 'redirects' until migration is done."""

from nilearn.maskers import SurfaceLabelsMasker as SurfLabMask
from nilearn.maskers import SurfaceMasker as SurfMask


class SurfaceMasker(SurfMask):
    """Extract data from a SurfaceImage."""


class SurfaceLabelsMasker(SurfLabMask):
    """Extract data from a SurfaceImage, averaging over atlas regions."""
