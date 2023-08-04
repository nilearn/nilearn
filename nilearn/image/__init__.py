"""Mathematical operations working on Niimg-like objects.

Like, for example, a (3+)D block of data, and an affine.
"""
from .._utils.niimg import copy_img
from .._utils.niimg_conversions import concat_niimgs as concat_imgs
from .image import (
    binarize_img,
    clean_img,
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
    "resample_img",
    "resample_to_img",
    "high_variance_confounds",
    "smooth_img",
    "crop_img",
    "mean_img",
    "reorder_img",
    "swap_img_hemispheres",
    "concat_imgs",
    "copy_img",
    "index_img",
    "iter_img",
    "new_img_like",
    "threshold_img",
    "math_img",
    "binarize_img",
    "load_img",
    "clean_img",
    "get_data",
    "largest_connected_component_img",
    "coord_transform",
]
