"""
Conversion utilities.
"""
# Author: Gael Varoquaux, Alexandre Abraham, Philippe Gervais
# License: simplified BSD

import collections

import numpy as np

import nibabel


def is_a_niimg(obj):
    """ Check for get_data and get_affine method in an object

    Parameters
    ----------
    obj: any object
        Tested object

    Returns
    -------
    is_niimg: boolean
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


def _get_shape(niimg):
    # Use the fact that Nifti1Image has a shape attribute that is
    # faster than loading the data from disk
    if hasattr(niimg, 'shape'):
        shape = niimg.shape
    else:
        shape = niimg.get_data().shape
    return shape


def _repr_niimgs(niimgs):
    """ Pretty printing of niimg or niimgs.
    """
    if isinstance(niimgs, basestring):
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


def copy_niimg(niimg):
    """Copy a niimg to a nibabel.Nifti1Image.

    Parameters
    ==========
    niimg: niimg
        Nifti image to copy.

    Returns
    =======
    niimg_copy: nibabel.Nifti1Image
        copy of input (data and affine)
    """
    if not is_a_niimg(niimg):
        raise ValueError("input value is not a niimg")
    return nibabel.Nifti1Image(niimg.get_data().copy(),
                               niimg.get_affine().copy())


def check_niimg(niimg):
    """Check that niimg is a proper niimg. Turn filenames into objects.

    Parameters
    ----------
    niimg: string or object
        If niimg is a string, consider it as a path to Nifti image and
        call nibabel.load on it. If it is an object, check if get_data()
        and get_affine() methods are present, raise TypeError otherwise.

    Returns
    -------
    result: nifti-like
       result can be nibabel.Nifti1Image or the input, as-is. It is guaranteed
       that the returned object has get_data() and get_affine() methods.

    Notes
    -----
    In NiLearn, special care has been taken to make image manipulation easy.
    This method is a kind of pre-requisite for any data processing method in
    NiLearn because it checks if data have a correct format and loads them if
    necessary.

    Its application is idempotent.
    """

    if isinstance(niimg, basestring):
        # data is a filename, we load it
        result = nibabel.load(niimg)
    elif hasattr(niimg, "__iter__"):
        return concat_niimgs(niimg)
    else:
        # it is an object, it should have get_data and get_affine methods
        if not is_a_niimg(niimg):
            raise TypeError("Given data does not expose"
                            " get_data or get_affine methods")
        result = niimg
    return result


def concat_niimgs(niimgs, dtype=np.float32):
    """Concatenate a list of niimgs

    Parameters
    ----------
    niimgs: iterable of niimgs
        niimgs to concatenate.

    Returns
    -------
    concatenated: nibabel.Nifti1Image
        A single niimg.
    """

    first_niimg = check_niimg(iter(niimgs).next())
    affine = first_niimg.get_affine()
    first_data = first_niimg.get_data()
    first_data_shape = first_data.shape
    # Using fortran order makes concatenation much faster than with C order,
    # because the voxels for a given image are grouped together in memory.
    data = np.ndarray(first_data_shape + (len(niimgs),),
                      order="F", dtype=dtype)
    data[..., 0] = first_data
    del first_data, first_niimg

    for index, iter_niimg in enumerate(niimgs):
        if index == 0:
            continue
        niimg = check_niimg(iter_niimg)
        if not np.array_equal(niimg.get_affine(), affine):
            if (isinstance(iter_niimg, basestring)):
                i_error = "image " + iter_niimg
            else:
                i_error = "image #" + str(index)

            raise ValueError("Affine of %s is different"
                             " from reference affine"
                             "\nReference affine:\n%s\n"
                             "Wrong affine:\n%s"
                             % (i_error,
                             repr(affine), repr(niimg.get_affine())))
        this_data = niimg.get_data()
        if this_data.shape != first_data_shape:
            if (isinstance(iter_niimg, basestring)):
                i_error = "image " + iter_niimg
            else:
                i_error = "image #" + str(index)
            raise ValueError("Shape of %s is different from first image shape."
                             % i_error)
        data[..., index] = this_data
    return nibabel.Nifti1Image(data, affine)


def check_niimgs(niimgs, accept_3d=False):
    """ Check that an object is a list of niimg and load it if necessary

    Parameters
    ----------
    niimgs: (iterable of)* strings or objects
        If niimgs is an iterable, checks if data is really 4D. Then,
        considering that it is a list of niimg and load them one by one.
        If niimg is a string, consider it as a path to Nifti image and
        call nibabel.load on it. If it is an object, check if get_data
        and get_affine methods are present, raise an Exception otherwise.

   accept_3d (boolean)
       If True, consider a 3D image as a 4D one with last dimension equals
       to 1.

    Returns
    -------
    niimg: nibabel.Nifti1Image
        One 4D image. If 3D images were provided as input, this is the
        concatenation of all of them.

    Notes
    -----
    This function is the equivalent of check_niimg() for niimages with a
    session level.

    Its application is idempotent.
    """
    # Initialization:
    # If given data is a list, we count the number of levels to check
    # dimensionality and make a consistent error message.
    depth = 0
    first_img = niimgs
    if accept_3d and (isinstance(first_img, basestring)
                      or not isinstance(first_img, collections.Iterable)):
        niimg = check_niimg(niimgs)
        if len(_get_shape(niimg)) == 3:
            niimg = nibabel.Nifti1Image(niimg.get_data()[..., np.newaxis],
                                        niimg.get_affine())
        return niimg

    # Use hasattr() instead of isinstance to workaround a Python 2.6/2.7 bug
    # See http://bugs.python.org/issue7624
    while hasattr(first_img, "__iter__") \
            and not isinstance(first_img, basestring):
        first_img = iter(first_img).next()
        depth += 1

    # First image is supposed to be a path or a Nifti-like element
    first_img = check_niimg(first_img)

    # Check dimension and depth
    dim = len(_get_shape(first_img))

    if (dim + depth) != 4:
        # Detailed error message that tells exactly the user what
        # was provided and what should have been provided.
        raise TypeError("Data must be either a 4D Nifti image or a"
                        " list of 3D Nifti images. You provided a %s%dD"
                        " image(s)." % ('list of ' * depth, dim))

    # Now, we load data as we know its format
    if dim == 4:
        niimg = check_niimg(niimgs)
    else:
        niimg = concat_niimgs(niimgs)
    return niimg
