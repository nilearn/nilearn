import copy
import gc
import collections
from six import string_types

import numpy as np

import nibabel

from .numpy_conversions import as_ndarray


def _get_shape(img):
    # Use the fact that Nifti1Image has a shape attribute that is
    # faster than loading the data from disk
    if hasattr(img, 'shape'):
        shape = img.shape
    else:
        shape = img.get_data().shape
    return shape


def _safe_get_data(img):
    """ Get the data in the image without having a side effect on the
        Nifti1Image object
    """
    if hasattr(img, '_data_cache') and img._data_cache is None:
        # Copy locally the Nift1Image object to avoid the side effect of data
        # loading
        img = copy.deepcopy(img)
    # typically the line below can double memory usage
    # that's why we invoke a forced call to the garbage collector
    gc.collect()
    return img.get_data()


def is_img(obj):
    """ Check for get_data and get_affine method in an object

    Parameters
    ----------
    obj: any object
        Tested object

    Returns
    -------
    is_img: boolean
        True if get_data and get_affine methods are present and callable,
        False otherwise.
    """

    # We use a try/except here because this is the way hasattr works
    try:
        get_data = getattr(obj, "get_data")
        get_affine = getattr(obj, "get_affine")
        return callable(get_data) and callable(get_affine)
    except AttributeError:
        return False


# XXX add dtype option for masks
def load_img(img):
    """Load a niimg and check if it has required methods
    """
    if isinstance(img, string_types):
        # data is a filename, we load it
        img = nibabel.load(img)
    elif not is_img(img):
        raise TypeError("Data given cannot be converted to a nifti"
                        " image: this object -'%s'- does not expose"
                        " get_data or get_affine methods"
                        % short_repr(img))
    return img


def new_img_like(ref_img, data, affine, copy_header=False):
    """Create a new image of the same class as the reference image

    Parameters
    ----------
    ref_img: Niimg-like image
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

    new_img: nibabel.SpatialImage
        An image which has the same type as the reference image.
    """
    # XXX If the niimg is a list of 3D images, we don't need to load them all
    # But this is a bit ugly
    if hasattr(ref_img, "__iter__") and not isinstance(ref_img, string_types):
        ref_img = next(ref_img)
    ref_img = load_img(ref_img)
    header = ref_img.get_header() if copy_header else None
    # XXX Nifti can't handle boolean, is this the case of other types?
    data = np.asarray(data)
    if data.dtype == bool:
        data = as_ndarray(data, dtype=np.int8)
    return ref_img.__class__(data, affine, header=header)


def copy_img(img):
    """Copy an image to a nibabel.Nifti1Image.

    Parameters
    ==========
    img: image
        nibabel.Nifti1Image object to copy.

    Returns
    =======
    img_copy: nibabel.Nifti1Image
        copy of input (data and affine)
    """
    if not is_img(img):
        raise ValueError("Input value is not an image")
    return nibabel.Nifti1Image(img.get_data().copy(),
                               img.get_affine().copy())


def _repr_niimgs(niimgs):
    """ Pretty printing of niimg or niimgs.
    """
    if isinstance(niimgs, string_types):
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
                    repr(_get_shape(niimgs)),
                    repr(niimgs.get_affine()))
    except:
        pass
    return repr(niimgs)


def short_repr(niimg):
    """Gives a shorten version on niimg representation
    """
    this_repr = repr(niimg)
    if len(this_repr) > 20:
        # Shorten the repr to have a useful error message
        this_repr = this_repr[:18] + '...'
    return this_repr
