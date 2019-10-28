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

from .compat import _basestring


def _get_data(img):
    # copy-pasted from https://github.com/nipy/nibabel/blob/de44a105c1267b07ef9e28f6c35b31f851d5a005/nibabel/dataobj_images.py#L204
    # get_data is removed from nibabel because:
    # see https://github.com/nipy/nibabel/wiki/BIAP8
    if img._data_cache is not None:
        return img._data_cache
    data = np.asanyarray(img._dataobj)
    img._data_cache = data
    return data


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
        nilearn.image.get_data return from Nifti image.
    """
    if hasattr(img, '_data_cache') and img._data_cache is None:
        # By loading directly dataobj, we prevent caching if the data is
        # memmaped. Preventing this side-effect can save memory in some cases.
        img = copy.deepcopy(img)
    # typically the line below can double memory usage
    # that's why we invoke a forced call to the garbage collector
    gc.collect()

    data = _get_data(img)
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

    dtype = _get_target_dtype(_get_data(niimg).dtype, dtype)

    if dtype is not None:
        # Copyheader and set dtype in header if header exists
        if niimg.header is not None:
            niimg = new_img_like(niimg, _get_data(niimg).astype(dtype),
                                niimg.affine, copy_header=True)
            niimg.header.set_data_dtype(dtype)        
        else:
            niimg = new_img_like(niimg, _get_data(niimg).astype(dtype),
                                niimg.affine)

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
    return new_img_like(img, _safe_get_data(img).copy(), img.affine.copy(),
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
                    repr(niimgs.affine))
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


def img_data_dtype(niimg):
    """Determine type of data contained in image

    Based on the information contained in ``niimg.dataobj``, determine the
    dtype of ``np.array(niimg.dataobj).dtype``.
    """

    dataobj = niimg.dataobj

    # Neuroimages that scale data should be interpreted as floating point
    if nibabel.is_proxy(dataobj) and (dataobj.slope, dataobj.inter) != (1.0, 0.0):
        return np.float_

    # ArrayProxy gained the dtype attribute in nibabel 2.2
    if hasattr(dataobj, 'dtype'):
        return dataobj.dtype

    return niimg.get_data_dtype()
