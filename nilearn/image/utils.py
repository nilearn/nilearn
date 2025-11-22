"""Image utilities available for other nilearn subpackages."""

import numpy as np

from nilearn._utils.niimg import safe_get_data
from nilearn._utils.niimg_conversions import (
    check_niimg,
)
from nilearn.surface.surface import (
    SurfaceImage,
)
from nilearn.surface.surface import get_data as get_surface_data
from nilearn.typing import NiimgLike


def get_indices_from_image(image) -> np.ndarray:
    """Return unique values in a label image."""
    if isinstance(image, NiimgLike):
        img = check_niimg(image)
        data = safe_get_data(img)
    elif isinstance(image, SurfaceImage):
        data = get_surface_data(image)
    elif isinstance(image, np.ndarray):
        data = image
    else:
        raise TypeError(
            "Image to extract indices from must be one of: "
            "Niimg-Like, SurfaceImage, numpy array. "
            f"Got {image.__class__.__name__}"
        )

    labels_present = np.unique(data)

    return labels_present[np.isfinite(labels_present)]
