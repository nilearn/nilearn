"""
Mathematical operations working on Niimg-like objects like -a (3+n)-D block of
data, and an affine.
"""
from .resampling import resample_img, reorder_img
from .image import high_variance_confounds, smooth_img, crop_img, \
    mean_img, swap_img_hemispheres, index_img, iter_img
from .._utils.niimg_conversions import concat_niimgs as concat_imgs

__all__ = ['resample_img', 'high_variance_confounds', 'smooth_img',
           'crop_img', 'mean_img', 'reorder_img',
           'swap_img_hemispheres', 'concat_imgs',
           'index_img', 'iter_img']
