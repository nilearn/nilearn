"""
Validation and conversion utilities.
"""

import collections
import itertools

import nibabel
import numpy as np


def _check_niimg_methods(object):
    # We use a try/except here because this is the way hasattr works
    try:
        get_data = getattr(object, "get_data")
        get_affine = getattr(object, "get_affine")
        return callable(get_data) and callable(get_affine)
    except Exception:
        return False


def check_niimg(niimg):
    if isinstance(niimg, basestring):
        # data is a filename, we load it
        result = nibabel.load(niimg)
    else:
        # it is an object, it should have get_data and get_affine methods
        if not _check_niimg_methods(niimg):
            raise TypeError("Given data does not expose"
                " get_data or get_affine methods")
        result = niimg
    return result


def collapse_niimgs(niimgs, compute_sessions=False):
    data = []
    if compute_sessions:
        sessions = []
    first_niimg = iter(niimgs).next()
    affine = first_niimg.get_affine()
    for index, iter_niimg in enumerate(niimgs):
        niimg = check_niimg(iter_niimg)
        if not np.array_equal(niimg.get_affine(), affine):
            s_error = ""
            if compute_sessions:
                s_error = " of session #" + str(index)
            if (isinstance(iter_niimg, basestring)):
                i_error = "image " + iter_niimg
            else:
                i_error = "image #" + str(index)

            raise ValueError("Affine of %s%s is different"
                    " from reference affine"
                    "\nReference affine:\n%s\n"
                    "Wrong affine:\n%s"
                    % (i_error, s_error,
                    repr(affine), repr(niimg.get_affine())))
        data.append(niimg.get_data())
        if compute_sessions:
            sessions += list(itertools.repeat(index, niimg.get_data().shape[-1]))
    if compute_sessions:
        return data, affine, sessions
    return data, affine


def check_niimgs(niimgs):
    # Initialization: 
    # If given data is a list, we count the number of levels to check
    # dimensionality and make a consistent error message.
    depth = 0
    first_img = niimgs
    while isinstance(first_img, collections.Iterable) \
            and not isinstance(first_img, basestring):
        first_img = iter(first_img).next()
        depth += 1

    # First Image is supposed to be a path or a Nifti like element
    first_img = check_niimg(first_img)

    # Check dimension and depth
    dim = len(first_img.get_data().shape)

    if (dim + depth) != 4:
        # Very detailed error message that tells exactly the user what 
        # was provided and what should have been provided.
        raise ValueError("Data must be either a 4D Nifti image or a"
                " list of 3D Nifti images. You provided a %s%dD image(s)."
                % ('list of ' * depth, dim))

    # Now, we load data as we know its format
    if dim == 4:
        niimg = check_niimg(niimgs)
    else:
        niimg = collapse_niimgs(niimgs)
    return niimg 
