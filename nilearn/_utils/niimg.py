"""
Neuroimaging file input and output.
"""
# Author: Gael Varoquaux, Alexandre Abraham, Philippe Gervais
# License: simplified BSD

import copy
import gc
import collections
from distutils.version import LooseVersion

import numpy as np
import nibabel

from .numpy_conversions import as_ndarray
from .compat import _basestring


def _safe_get_data(img):
    """ Get the data in the image without having a side effect on the
        Nifti1Image object
    """
    if hasattr(img, '_data_cache') and img._data_cache is None:
        # By loading directly dataobj, we prevent caching if the data is
        # memmaped. Preventing this side-effect can save memory in some cases.
        img = copy.deepcopy(img)
    # typically the line below can double memory usage
    # that's why we invoke a forced call to the garbage collector
    gc.collect()
    return img.get_data()


def _get_data_dtype(img):
    """Returns the dtype of an image.
    If the image is non standard (no get_data_dtype member), this function
    relies on the data itself.
    """
    try:
        return img.get_data_dtype()
    except AttributeError:
        return img.get_data().dtype


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
        See http://nilearn.github.io/building_blocks/manipulating_mr_images.html#niimg.
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
    if isinstance(niimg, _basestring):
        # data is a filename, we load it
        niimg = nibabel.load(niimg)
    elif not isinstance(niimg, nibabel.spatialimages.SpatialImage):
        raise TypeError("Data given cannot be loaded because it is"
                        " not compatible with nibabel format:\n"
                        + short_repr(niimg))

    dtype = _get_target_dtype(_get_data_dtype(niimg), dtype)

    if dtype is not None:
        niimg = new_img_like(niimg, niimg.get_data().astype(dtype),
                             niimg.get_affine())
    return niimg


def new_img_like(ref_img, data, affine, copy_header=False):
    """Create a new image of the same class as the reference image

    Parameters
    ----------
    ref_img: image
        Reference image. The new image will be of the same type.

    data: numpy array
        Data to be stored in the image

    affine: 4x4 numpy array
        Transformation matrix

    copy_header: boolean, optional
        Indicated if the header of the reference image should be used to
        create the new image

    Returns
    -------

    new_img: image
        An image which has the same type as the reference image.
    """
    if data.dtype == bool:
        default_dtype = np.int8
        if (LooseVersion(nibabel.__version__) >= LooseVersion('1.2.0') and
                isinstance(ref_img, nibabel.freesurfer.mghformat.MGHImage)):
            default_dtype = np.uint8
        data = as_ndarray(data, dtype=default_dtype)
    header = None
    if copy_header:
        header = copy.copy(ref_img.get_header())
        header['scl_slope'] = 0.
        header['scl_inter'] = 0.
        header['glmax'] = 0.
        header['cal_max'] = np.max(data) if data.size > 0 else 0.
        header['cal_max'] = np.min(data) if data.size > 0 else 0.
    return ref_img.__class__(data, affine, header=header)


def copy_img(img):
    """Copy an image to a nibabel.Nifti1Image.

    Parameters
    ==========
    img: image
        nibabel SpatialImage object to copy.

    Returns
    =======
    img_copy: image
        copy of input (data, affine and header)
    """
    if not isinstance(img, nibabel.spatialimages.SpatialImage):
        raise ValueError("Input value is not an image")
    return new_img_like(img, img.get_data().copy(), img.get_affine().copy(),
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
                    repr(niimgs.get_affine()))
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
