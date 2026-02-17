# emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the NiBabel package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Define supported image classes and names"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .analyze import AnalyzeImage
from .brikhead import AFNIImage
from .cifti2 import Cifti2Image
from .freesurfer import MGHImage
from .gifti import GiftiImage
from .minc1 import Minc1Image
from .minc2 import Minc2Image
from .nifti1 import Nifti1Image, Nifti1Pair
from .nifti2 import Nifti2Image, Nifti2Pair
from .parrec import PARRECImage
from .spm2analyze import Spm2AnalyzeImage
from .spm99analyze import Spm99AnalyzeImage

if TYPE_CHECKING:
    from .dataobj_images import DataobjImage
    from .filebasedimages import FileBasedImage

# Ordered by the load/save priority.
all_image_classes: list[type[FileBasedImage]] = [
    Nifti1Pair,
    Nifti1Image,
    Nifti2Pair,
    Cifti2Image,
    Nifti2Image,  # Cifti2 before Nifti2
    Spm2AnalyzeImage,
    Spm99AnalyzeImage,
    AnalyzeImage,
    Minc1Image,
    Minc2Image,
    MGHImage,
    PARRECImage,
    GiftiImage,
    AFNIImage,
]

# Image classes known to require spatial axes to be first in index ordering.
# When adding an image class, consider whether the new class should be listed
# here.
KNOWN_SPATIAL_FIRST: tuple[type[FileBasedImage], ...] = (
    Nifti1Pair,
    Nifti1Image,
    Nifti2Pair,
    Nifti2Image,
    Spm2AnalyzeImage,
    Spm99AnalyzeImage,
    AnalyzeImage,
    MGHImage,
    PARRECImage,
    AFNIImage,
)


def spatial_axes_first(img: DataobjImage) -> bool:
    """True if spatial image axes for `img` always precede other axes

    Parameters
    ----------
    img : object
        Image object implementing at least ``shape`` attribute.

    Returns
    -------
    spatial_axes_first : bool
        True if image only has spatial axes (number of axes < 4) or image type
        known to have spatial axes preceding other axes.
    """
    if len(img.shape) < 4:
        return True
    return type(img) in KNOWN_SPATIAL_FIRST
