"""
Mathematical operations working on niimgs like -a (3+n)-D block of data,
and an affine.
"""
from .resampling import resample_img
from .image import high_variance_confounds, smooth_img, crop_img, \
            mean_img, swap_img_hemispheres

__all__ = ['resample_img', 'high_variance_confounds', 'smooth_img',
           'crop_img', 'mean_img', 'reorder_img',
           'swap_img_hemispheres']

