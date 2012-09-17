import types
import itertools

import nibabel
import numpy as np


def _check_nifti_methods(object):
    # We use a try/except here because this is the way hasattr works
    try:
        get_data = getattr(object, "get_data")
        get_affine = getattr(object, "get_affine")
        return callable(get_data) and callable(get_affine)
    except Exception:
        return False


def check_nifti_image(niimg):
    if isinstance(niimg, types.StringTypes):
        # data is a filename, we load it
        result = nibabel.load(niimg)
    else:
        # it is an object, it should have get_data and get_affine methods
        if not _check_nifti_methods(niimg):
            raise AttributeError("Given data does not expose"
                " get_data or get_affine methods")
        result = niimg
    return result


def collapse_nifti_images(niimgs, compute_sessions=False):
    data = []
    if compute_sessions:
        sessions = []
    first_niimg = iter(niimgs).next()
    affine = first_niimg.get_affine()
    for index, iter_niimg in enumerate(niimgs):
        niimg = check_nifti_image(iter_niimg)
        if not np.array_equal(niimg.get_affine(), affine):
            s_error = ""
            if compute_sessions:
                s_error = " of session #" + str(index)
            if (isinstance(iter_niimg, types.StringTypes)):
                i_error = "image " + iter_niimg
            else:
                i_error = "image #" + str(index)

            raise ValueError("Affine of %s%s is different"
                    " from reference affine"
                    "\nReference affine:\n%s\n"
                    "Wrong affine:\n%s"
                    % (i_error, s_error,
                    repr(affine), repr(niimg.get_affine())))
        data.append(niimg)
        if compute_sessions:
            sessions += list(itertools.repeat(index, niimg.get_data().shape[-1]))
    if compute_sessions:
        return data, affine, sessions
    return data, affine
