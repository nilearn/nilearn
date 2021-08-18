"""
Neuroimaging file input and output.
"""
# Author: Gael Varoquaux, Alexandre Abraham, Philippe Gervais
# License: simplified BSD

import copy
import gc
import collections.abc
from warnings import warn

import numpy as np
import nibabel

from pathlib import Path


def _get_data(img):
    # copy-pasted from https://github.com/nipy/nibabel/blob/de44a105c1267b07ef9e28f6c35b31f851d5a005/nibabel/dataobj_images.py#L204
    # get_data is removed from nibabel because:
    # see https://github.com/nipy/nibabel/wiki/BIAP8
    if img._data_cache is not None:
        return img._data_cache
    data = np.asanyarray(img._dataobj)
    img._data_cache = data
    return data


def _safe_get_data(img, ensure_finite=False, copy_data=False):
    """ Get the data in the image without having a side effect on the
        Nifti1Image object

    Parameters
    ----------
    img: Nifti image/object
        Image to get data.

    ensure_finite: bool
        If True, non-finite values such as (NaNs and infs) found in the
        image will be replaced by zeros.

    copy_data: bool, default is False
        If true, the returned data is a copy of the img data.

    Returns
    -------
    data: numpy array
        nilearn.image.get_data return from Nifti image.
    """
    if copy_data:
        img = copy.deepcopy(img)

    # typically the line below can double memory usage
    # that's why we invoke a forced call to the garbage collector
    gc.collect()

    data = _get_data(img)
    if ensure_finite:
        non_finite_mask = np.logical_not(np.isfinite(data))
        if non_finite_mask.sum() > 0:  # any non_finite_mask values?
            warn(
                "Non-finite values detected. "
                "These values will be replaced with zeros."
            )
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

    if isinstance(niimg, str):
        # data is a filename, we load it
        niimg = nibabel.load(niimg)
    elif not isinstance(niimg, nibabel.spatialimages.SpatialImage):
        raise TypeError("Data given cannot be loaded because it is"
                        " not compatible with nibabel format:\n"
                        + _repr_niimgs(niimg, shorten=True))

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


def _is_binary_niimg(niimg):
    """Returns whether a given niimg is binary or not.

    Parameters
    ----------
    niimg: Niimg-like object
        See http://nilearn.github.io/manipulating_images/input_output.html
        Image to test.

    Returns
    -------
    is_binary: Boolean
        True if binary, False otherwise.

    """
    niimg = load_niimg(niimg)
    data = _safe_get_data(niimg, ensure_finite=True)
    unique_values = np.unique(data)
    if len(unique_values) != 2:
        return False
    return sorted(list(unique_values)) == [0,1]


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
    return new_img_like(img, _safe_get_data(img, copy_data=True), img.affine.copy(),
                        copy_header=True)


def _repr_niimgs(niimgs, shorten=True):
    """ Pretty printing of niimg or niimgs.

    Parameters
    ----------
    niimgs: image or collection of images
        nibabel SpatialImage to repr.

    shorten: boolean, optional, default is True
        If True, filenames with more than 20 characters will be
        truncated, and lists of more than 3 file names will be
        printed with only first and last element.

    Returns
    -------
    repr: str
        String representation of the image.
    """
    # Maximum number of elements to be displayed
    # Note: should be >= 3 to make sense...
    list_max_display = 3
    # Simple string case
    if isinstance(niimgs, (str, Path)):
        return _short_repr(niimgs, shorten=shorten)
    # Collection case
    if isinstance(niimgs, collections.abc.Iterable):
        if shorten and len(niimgs) > list_max_display:
            return '[%s]' % ',\n         ...\n '.join(_repr_niimgs(niimg, shorten=shorten) for niimg in [niimgs[0], niimgs[-1]])
        elif len(niimgs) > list_max_display:
            return '[%s]' % ',\n '.join(_repr_niimgs(niimg, shorten=shorten) for niimg in niimgs)
        else:
            return '[%s]' % ', '.join(_repr_niimgs(niimg, shorten=shorten) for niimg in niimgs)
   # Nibabel objects have a 'get_filename'
    try:
        filename = niimgs.get_filename()
        if filename is not None:
            return "%s('%s')" % (niimgs.__class__.__name__,
                                 _short_repr(filename,
                                             shorten=shorten))
        else:
            # No shortening in this case
            return "%s(\nshape=%s,\naffine=%s\n)" % \
                   (niimgs.__class__.__name__,
                    repr(niimgs.shape),
                    repr(niimgs.affine))
    except:
        pass
    return _short_repr(repr(niimgs), shorten=shorten)


def _short_repr(niimg_rep, shorten=True, truncate=20):
    """Gives a shorter version of niimg representation
    """
    # Make sure truncate has a reasonable value
    truncate = max(truncate, 10)
    path_to_niimg = Path(niimg_rep)
    if not shorten:
        return str(path_to_niimg)
    # If the name of the file itself is larger than
    # truncate, then shorten the name only
    if len(path_to_niimg.name) > truncate:
        return path_to_niimg.name[: (truncate - 2)] + '...'
    # Else add some folder structure if available
    else:
        rep = path_to_niimg.name
        if len(path_to_niimg.parts) > 1:
            for p in path_to_niimg.parts[::-1][1:]:
                if len(rep) + len(p) < truncate - 3:
                    rep = str(Path(p, rep))
                else:
                    rep = str(Path("...", rep))
                    break
        return rep


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
