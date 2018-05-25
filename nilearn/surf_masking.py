"""
Utilities to compute a brain mask from EPI images
"""
# Author: Sylvain Takerkart
# License: simplified BSD
import warnings

import numpy as np
from scipy import ndimage
#from nibabel import Nifti1Image
#from sklearn.externals.joblib import Parallel, delayed
#
from . import _utils
#from ._utils.ndimage import largest_connected_component
#from ._utils.niimg_conversions import _safe_get_data
#from ._utils.cache_mixin import cache


class SurfMaskWarning(UserWarning):
    "A class to always raise warnings"


warnings.simplefilter("always", SurfMaskWarning)




def _get_mesh_coords(gii_mesh):
    ''' Read coordinates of all nodse of a gifti mesh object

    Parameters
    ----------
    gii_mesh: gifti-like mesh object
        The mesh to check

    Returns
    -------
    mesh_coords: numpy.ndarray
        array containing the xyz coordinates of the nodes of the mesh
    '''

    mesh_coords = gii_mesh.darrays[0].data
    return mesh_coords


def _load_surfmask_tex(surfmask_tex, allow_empty=False):
    ''' Check that a mask is valid, ie with two values including 0 and load it.

    Parameters
    ----------
    surfmask_tex: gifti-like texture
        The mask to check

    allow_empty: boolean, optional
        Allow loading an empty mask (full of 0 values)

    Returns
    -------
    surfmask: numpy.ndarray
        boolean version of the mask
    '''
    surfmask = surfmask_tex.darrays[0].data
    values = np.unique(surfmask)

    if len(values) == 1:
        # We accept a single value if it is not 0 (full true mask).
        if values[0] == 0 and not allow_empty:
            raise ValueError('Given mask is invalid because it masks all data')
    elif len(values) == 2:
        # If there are 2 different values, one of them must be 0 (background)
        if not 0 in values:
            raise ValueError('Background of the mask must be represented with'
                             '0. Given mask contains: %s.' % values)
    elif len(values) != 2:
        # If there are more than 2 values, the mask is invalid
        raise ValueError('Given mask is not made of 2 values: %s'
                         '. Cannot interpret as true or false'
                         % values)

    surfmask = _utils.as_ndarray(surfmask, dtype=bool)
    return surfmask


def _apply_surfmask_fmri(giimgs, surfmask):
    """Same as apply_mask().

    The only difference with apply_mask is that some costly checks on mask_img
    are not performed: mask_img is assumed to contain only two different
    values (this is checked for in apply_mask, not in this function).
    """

    return giimgs[surfmask,:].T

















