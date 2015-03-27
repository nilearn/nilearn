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


def load_niimg(niimg, dtype=None):
    """Load a niimg, check if it is a nibabel SpatialImage and cast if needed

    Parameters:
    -----------

    niimg: Niimg-like object
        See http://nilearn.github.io/building_blocks/manipulating_mr_images.html#niimg.
        Image to load.

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
        data = as_ndarray(data, dtype=np.int8)
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
