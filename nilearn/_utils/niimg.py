"""
Neuroimaging file input and output.
"""
# Author: Gael Varoquaux, Alexandre Abraham, Philippe Gervais
# License: simplified BSD

import copy
import gc
import collections

import numpy as np
import nibabel

from .compat import _basestring, get_affine


def _safe_get_data(img, ensure_finite=False):
    """ Get the data in the image without having a side effect on the
        Nifti1Image object

    Parameters
    ----------
    img: Nifti image/object
        Image to get data.

    ensure_finite: bool
        If True, non-finite values such as (NaNs and infs) found in the
        image will be replaced by zeros.

    Returns
    -------
    data: numpy array
        get_data() return from Nifti image.
    """
    if hasattr(img, '_data_cache') and img._data_cache is None:
        # By loading directly dataobj, we prevent caching if the data is
        # memmaped. Preventing this side-effect can save memory in some cases.
        img = copy.deepcopy(img)
    # typically the line below can double memory usage
    # that's why we invoke a forced call to the garbage collector
    gc.collect()

    data = img.get_data()
    if ensure_finite:
        non_finite_mask = np.logical_not(np.isfinite(data))
        if non_finite_mask.sum() > 0: # any non_finite_mask values?
            data[non_finite_mask] = 0

    return data


def _get_target_dtype(dtype, target_dtype):
    """Returns a new dtype if conversion is needed

    Parameters
    ----------

    dtype: dtype
        Data type of the original data

    target_dtype: {None, dtype, "auto"}
        If None, no conversion is required. If a type is provided, the
        function will check if a conversion is needed. The "auto" mode will
        automatically convert to int32 if dtype is discrete and float32 if it
        is continuous.

    Returns
    -------

    dtype: dtype
        The data type toward which the original data should be converted.
    """

    if target_dtype is None:
        return None
    if target_dtype == 'auto':
        if dtype.kind == 'i':
            target_dtype = np.int32
        else:
            target_dtype = np.float32
    if target_dtype == dtype:
        return None
    return target_dtype


def load_niimg(niimg, dtype=None):
    """Load a niimg, check if it is a nibabel SpatialImage and cast if needed

    Parameters:
    -----------

    niimg: Niimg-like object
        See http://nilearn.github.io/manipulating_images/input_output.html
        Image to load.

    dtype: {dtype, "auto"}
        Data type toward which the data should be converted. If "auto", the
        data will be converted to int32 if dtype is discrete and float32 if it
        is continuous.

    Returns:
    --------
    img: image
        A loaded image object.
    """
    from ..image import new_img_like  # avoid circular imports

    if isinstance(niimg, _basestring):
        # data is a filename, we load it
        niimg = nibabel.load(niimg)
    elif not isinstance(niimg, nibabel.spatialimages.SpatialImage):
        raise TypeError("Data given cannot be loaded because it is"
                        " not compatible with nibabel format:\n"
                        + short_repr(niimg))

    dtype = _get_target_dtype(niimg.get_data().dtype, dtype)

    if dtype is not None:
        niimg = new_img_like(niimg, niimg.get_data().astype(dtype),
                             get_affine(niimg))
    return niimg


def copy_img(img):
    """Copy an image to a nibabel.Nifti1Image.

    Parameters
    ----------
    img: image
        nibabel SpatialImage object to copy.

    Returns
    -------
    img_copy: image
        copy of input (data, affine and header)
    """
    from ..image import new_img_like  # avoid circular imports

    if not isinstance(img, nibabel.spatialimages.SpatialImage):
        raise ValueError("Input value is not an image")
    return new_img_like(img, _safe_get_data(img).copy(), get_affine(img).copy(),
                        copy_header=True)


def _repr_niimgs(niimgs):
    """ Pretty printing of niimg or niimgs.
    """
    if isinstance(niimgs, _basestring):
        return niimgs
    if isinstance(niimgs, collections.Iterable):
        return '[%s]' % ', '.join(_repr_niimgs(niimg) for niimg in niimgs)
    # Nibabel objects have a 'get_filename'
    try:
        filename = niimgs.get_filename()
        if filename is not None:
            return "%s('%s')" % (niimgs.__class__.__name__,
                                 filename)
        else:
            return "%s(\nshape=%s,\naffine=%s\n)" % \
                   (niimgs.__class__.__name__,
                    repr(niimgs.shape),
                    repr(get_affine(niimgs)))
    except:
        pass
    return repr(niimgs)


def short_repr(niimg):
    """Gives a shorten version on niimg representation
    """
    this_repr = _repr_niimgs(niimg)
    if len(this_repr) > 20:
        # Shorten the repr to have a useful error message
        this_repr = this_repr[:18] + '...'
    return this_repr
