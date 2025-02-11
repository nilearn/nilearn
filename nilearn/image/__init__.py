"""Mathematical operations working on Niimg-like objects.

Like, for example, a (3+)D block of data, and an affine.
"""

from .image import (
    binarize_img,
    clean_img,
    concat_imgs,
    copy_img,
    crop_img,
    get_data,
    high_variance_confounds,
    index_img,
    iter_img,
    largest_connected_component_img,
    load_img,
    math_img,
    mean_img,
    new_img_like,
    smooth_img,
    swap_img_hemispheres,
    threshold_img,
)
from .resampling import (
    coord_transform,
    reorder_img,
    resample_img,
    resample_to_img,
)

__all__ = [
    "binarize_img",
    "clean_img",
    "concat_imgs",
    "coord_transform",
    "copy_img",
    "crop_img",
    "get_data",
    "high_variance_confounds",
    "index_img",
    "iter_img",
    "largest_connected_component_img",
    "load_img",
    "math_img",
    "mean_img",
    "new_img_like",
    "reorder_img",
    "resample_img",
    "resample_to_img",
    "smooth_img",
    "swap_img_hemispheres",
    "threshold_img",
]
