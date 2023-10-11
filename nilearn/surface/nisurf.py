"""
Surface file input and output utils
"""

import numpy as np

import collections.abc
from copy import deepcopy
from .._utils.exceptions import DimensionError
from .._utils.niimg import _get_target_dtype
from .._utils import as_ndarray
from .surface import load_surf_data


def check_nisurf_1d(nisurf, dtype=None):
    """Check that nisurf is a proper 1D niimg-like object and load it.

    Parameters
    ----------
    nisurf: Nisurf-data-like object
        See http://nilearn.github.io/manipulating_images/input_output.html
        If nisurf is a string, consider it as a path to Surface file and
        call surface.load_surface_data on it.
        If it is an object, check to make sure it is a 1D Numpy array.

    dtype: {dtype, "auto"}
        Data type toward which the data should be converted. If "auto", the
        data will be converted to int32 if dtype is discrete and float32 if it
        is continuous.

    Returns
    -------
    result: 1D Numpy array
        Result is a 1D Numpy array representing the surface. 

    """

    # Load the data w/ load_surf_data, which implements most checks
    data = load_surf_data(nisurf)

    # Ensure is 1D
    if len(data.shape) != 1:
        raise DimensionError(len(data.shape), 1)

    # Proccess passed dtype
    new_dtype = _get_target_dtype(data.dtype, dtype)
    data = as_ndarray(data, dtype=new_dtype)

    return data


def check_nisurf_2d(nisurf, atleast_2d=False, dtype=None):
    """Check that nisurf is a proper 2D niimg-like object and load it.

    Parameters
    ----------
    nisurf: Nisurf-data-like object
        See http://nilearn.github.io/manipulating_images/input_output.html
        If nisurf is a string, consider it as a path to a 2D Surface file and
        call surface.load_surface_data on it.
        If it is a Numpy array, check to make sure it really is a 2D Numpy
        array.
        If it is an iterable, and not a Numpy array,
        load each piece seperately, and concatenate.

    atleast_2d : boolean, optional
        Indicates if a 1d Numpy array should be turned into a single-scan
        2D Numpy array.

    dtype: {dtype, "auto"}
        Data type toward which the data should be converted. If "auto", the
        data will be converted to int32 if dtype is discrete and float32 if it
        is continuous.

    Returns
    -------
    result: 1D Numpy array
        Result is a 2D Numpy array as
        (number of vertex, number of surfaces)
    """

    # Handle different cases for 2D surface input
    if isinstance(nisurf, str) or isinstance(nisurf, np.ndarray):
        # If str, or already ndarray call load_surf_data
        data = load_surf_data(nisurf)
    elif isinstance(nisurf, collections.abc.Iterable):
        # If iterable, try to load each piece seperately
        for s in range(len(nisurf)):
            data_part = load_surf_data(nisurf[s])

            if len(data_part.shape) == 1:
                data_part = data_part[:, np.newaxis]
            if s == 0:
                data = data_part
            elif s > 0:
                try:
                    data = np.concatenate((data, data_part), axis=1)
                except ValueError:
                    raise ValueError('When more than one file is input, all '
                                     'files must contain data with the same '
                                     'shape in axis=0')
    else:
        raise ValueError('The input type is not recognized. '
                         'Valid inputs are a 2D Numpy array '
                         'a list of 1D Numpy arrays, a valid file '
                         'with a 2D surface or a list of valid files '
                         'with 1D surfaces.')

    # Convert to 2D if specified
    if atleast_2d and len(data.shape) == 1:
        data = data[:, np.newaxis]
    # Ensure is 2D
    if len(data.shape) != 2:
        raise DimensionError(len(data.shape), 2)

    # Proccess passed dtype
    new_dtype = _get_target_dtype(data.dtype, dtype)
    data = as_ndarray(data, dtype=new_dtype)

    return data


def _repr_nisurfs_data(nisurfs):
    """ Pretty printing of nisurf-data or nisurfs-data
    """
    if isinstance(nisurfs, str):
        return nisurfs
    if isinstance(nisurfs, np.ndarray):
        return "Surface(shape=%s)" % \
               (repr(nisurfs.shape))
    if isinstance(nisurfs, collections.abc.Iterable):
        return ('[%s]' % ', '.join(_repr_nisurfs_data(nisurf)
                for nisurf in nisurfs))

    return repr(nisurfs)


def _load_surf_mask(mask_surf, allow_empty=False):
    """Check that a mask is valid, ie with two values including 0 and load it.

    Parameters
    ----------
    mask_surf: Nisurf-data-like object
        See http://nilearn.github.io/manipulating_images/input_output.html
        The mask to check

    allow_empty: boolean, optional
        Allow loading an empty mask (full of 0 values)

    Returns
    -------
    mask: numpy.ndarray
        boolean version of the mask
    """
    mask = check_nisurf_1d(mask_surf)
    values = np.unique(mask)

    if len(values) == 1:
        # We accept a single value if it is not 0 (full true mask).
        if values[0] == 0 and not allow_empty:
            raise ValueError(
                'The mask is invalid as it is empty: it masks all data.')
    elif len(values) == 2:
        # If there are 2 different values, one of them must be 0 (background)
        if 0 not in values:
            raise ValueError('Background of the mask must be represented with'
                             '0. Given mask contains: %s.' % values)
    elif len(values) != 2:
        # If there are more than 2 values, the mask is invalid
        raise ValueError('Given mask is not made of 2 values: %s'
                         '. Cannot interpret as true or false'
                         % values)

    mask = as_ndarray(mask, dtype=bool)
    return mask


def _safe_get_data(surf, copy=False, ensure_finite=False):
    """ Analog to niimg _safe_get_data.
    Right now, as there is no wrapper object for surfaces,
    this will just allow retriving a copy / a version w/o
    NaNs + infs.

    In the future, if a wrapper object for Nisurf-data
    gets added, w/ support for caching / other NiftiImage features,
    then this function can be adapted.

    Parameters
    ----------
    surf: 1D or 2D Nisurf-data-like
        The surf to get data for

    copy : bool, optional
        If a deepcopy of the surface / Numpy array should be
        returned.

    ensure_finite: bool, optional
        If True, non-finite values such as (NaNs and infs) found in the
        surf will be replaced by zeros. If copy is False, but a
        NaN / Inf val is found, create a copy anyways.

    Returns
    -------
    data: numpy array
        nilearn.image.get_data return from Nifti image.
    """

    if copy:
        surf_copy = deepcopy(surf)
    else:
        surf_copy = surf

    non_finite_mask = np.logical_not(np.isfinite(surf_copy))
    if non_finite_mask.sum() > 0:

        # If copy was False, but replacing a NaN/Inf value
        # Assume that we do not want to change the original array
        # and make a copy now.
        if not copy:
            surf_copy = deepcopy(surf_copy)

        surf_copy[non_finite_mask] = 0

    return surf_copy
