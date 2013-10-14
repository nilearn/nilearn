"""
Utilities to resample a Nifti Image
"""
# Author: Gael Varoquaux, Alexandre Abraham
# License: simplified BSD


import numpy as np
from scipy import ndimage, linalg
from nibabel import Nifti1Image

from .. import _utils


def to_matrix_vector(transform):
    """Split an homogeneous transform into its matrix and vector components.

    The transformation must be represented in homogeneous coordinates.
    It is split into its linear transformation matrix and translation vector
    components.

    This function does not normalize the matrix. This means that for it to be
    the inverse of from_matrix_vector, transform[-1, -1] must equal 1, and
    transform[-1, :-1] must equal 0.

    Parameters
    ----------
    transform: numpy.ndarray
        Homogeneous transform matrix. Example: a (4, 4) transform representing
        linear transformation and translation in 3 dimensions.

    Returns
    -------
    matrix, vector: numpy.ndarray
        The matrix and vector components of the transform matrix.  For
        an (N, N) transform, matrix will be (N-1, N-1) and vector will be
        a 1D array of shape (N-1,).

    See Also
    --------
    from_matrix_vector
    """

    ndimin = transform.shape[0] - 1
    ndimout = transform.shape[1] - 1
    matrix = transform[0:ndimin, 0:ndimout]
    vector = transform[0:ndimin, ndimout]
    return matrix, vector


def from_matrix_vector(matrix, vector):
    """Combine a matrix and vector into a homogeneous transform.

    Combine a rotation matrix and translation vector into a transform
    in homogeneous coordinates.

    Parameters
    ----------
    matrix: numpy.ndarray
        An (N, N) array representing the rotation matrix.

    vector: numpy.ndarray
        A (1, N) array representing the translation.

    Returns
    -------
    xform: numpy.ndarray
        An (N+1, N+1) transform matrix.

    See Also
    --------
    nilearn.resampling.to_matrix_vector
    """

    nin, nout = matrix.shape
    t = np.zeros((nin + 1, nout + 1), matrix.dtype)
    t[0:nin, 0:nout] = matrix
    t[nin,   nout] = 1.
    t[0:nin, nout] = vector
    return t


def get_bounds(shape, affine):
    """Return the world-space bounds occupied by an array given an affine.

    The coordinates returned correspond to the **center** of the corner voxels.

    Parameters
    ==========
    shape: tuple
        shape of the array. Must have 3 integer values.

    affine: numpy.ndarray
        affine giving the linear transformation between voxel coordinates
        and world-space coordinates.

    Returns
    =======
    coord: list of tuples
        coord[i] is a 2-tuple giving minimal and maximal coordinates along
        i-th axis.
    """
    adim, bdim, cdim = shape
    adim -= 1
    bdim -= 1
    cdim -= 1
    # form a collection of vectors for each 8 corners of the box
    box = np.array([[0.,   0,    0,    1],
                    [adim, 0,    0,    1],
                    [0,    bdim, 0,    1],
                    [0,    0,    cdim, 1],
                    [adim, bdim, 0,    1],
                    [adim, 0,    cdim, 1],
                    [0,    bdim, cdim, 1],
                    [adim, bdim, cdim, 1]]).T
    box = np.dot(affine, box)[:3]
    return zip(box.min(axis=-1), box.max(axis=-1))


def resample_img(niimg, target_affine=None, target_shape=None,
                 interpolation='continuous', copy=True, order="F"):
    """ Resample a Nifti Image

    Parameters
    ----------
    niimg: nilearn nifti image
        Path to a nifti file or nifti-like object

    target_affine: numpy.ndarray, optional
        If specified, the image is resampled corresponding to this new affine.
        target_affine can be a 3x3 or a 4x4 matrix

    target_shape: tuple or list, optional
        If specified, the image will be resized to match this new shape.
        len(target_shape) must be equal to 3.
        A target_affine has to be specified jointly with target_shape.

    interpolation: str, optional
        Can be continuous' (default) or 'nearest'. Indicate the resample method

    copy: bool, optional
        If True, guarantees that output array has no memory in common with
        input array.
        In all cases, input images are never modified by this function.

    order: "F" or "C"
        Data ordering in output array. This function is slightly faster with
        Fortran ordering.

    Returns
    =======
    resampled: nibabel.Nifti1Image
        input image, resampled to have respectively target_shape and
        target_affine as shape and affine.
    """
    # Do as many checks as possible before loading data, to avoid potentially
    # costly calls before raising an exception.
    if target_shape is not None and target_affine is None:
        raise ValueError("If target_shape is specified, target_affine should"
                         " be specified too.")

    if target_shape is not None and not len(target_shape) == 3:
        raise ValueError('The shape specified should be the shape of'
                         'the 3D grid, and thus of length 3. %s was specified'
                         % str(target_shape))

    if interpolation == 'continuous':
        interpolation_order = 3
    elif interpolation == 'nearest':
        interpolation_order = 0
    else:
        raise ValueError("interpolation must be either 'continuous' "
                         "or 'nearest'")

    if isinstance(niimg, basestring):
        # Avoid a useless copy
        input_niimg_is_string = True
    else:
        input_niimg_is_string = False

    # noop cases
    niimg = _utils.check_niimg(niimg)

    if target_affine is None and target_shape is None:
        if copy and not input_niimg_is_string:
            niimg = _utils.copy_niimg(niimg)
        return niimg
    if target_affine is not None:
        target_affine = np.asarray(target_affine)

    shape = _utils._get_shape(niimg)
    affine = niimg.get_affine()

    if (np.all(np.array(target_shape) == shape[:3]) and
            np.allclose(target_affine, affine)):
        if copy and not input_niimg_is_string:
            niimg = _utils.copy_niimg(niimg)
        return niimg

    # We now know that some resampling must be done.
    # The value of "copy" is of no importance: output is always a separate
    # array.
    data = niimg.get_data()

    if target_shape is None:
        target_shape = data.shape[:3]
    target_shape = list(target_shape)

    if target_affine.shape[0] == 3:
        # We have a 3D affine, we need to find out the offset and
        # shape to keep the same bounding box in the new space
        affine4d = np.eye(4)
        affine4d[:3, :3] = target_affine
        transform_affine = np.dot(linalg.inv(affine4d), affine)
        # The bounding box in the new world, if no offset is given
        (xmin, xmax), (ymin, ymax), (zmin, zmax) = \
            get_bounds(data.shape[:3], transform_affine)

        offset = np.array((xmin, ymin, zmin))
        offset = np.dot(target_affine, offset)
        target_affine = from_matrix_vector(target_affine, offset[:3])
        target_shape = (int(np.ceil(xmax - xmin)) + 1,
                        int(np.ceil(ymax - ymin)) + 1,
                        int(np.ceil(zmax - zmin)) + 1, )

    if np.all(target_affine == affine):
        # Small trick to be more numerically stable
        transform_affine = np.eye(4)
    else:
        transform_affine = np.dot(linalg.inv(affine), target_affine)
    A, b = to_matrix_vector(transform_affine)
    A_inv = linalg.inv(A)
    # If A is diagonal, ndimage.affine_transform is clever enough to use a
    # better algorithm.
    if np.all(np.diag(np.diag(A)) == A):
        A = np.diag(A)
    else:
        b = np.dot(A, b)

    data_shape = list(data.shape)
    # For images with dimensions larger than 3D:
    if len(data_shape) > 3:
        # Iter in a set of 3D volumes, as the interpolation problem is
        # separable in the extra dimensions. This reduces the
        # computational cost
        other_shape = data_shape[3:]
        resampled_data = np.ndarray(list(target_shape) + other_shape,
                                    order=order)

        all_img = (slice(None), ) * 3

        for ind in np.ndindex(*other_shape):
            img = data[all_img + ind]
            resampled_data[all_img + ind] = \
                                   ndimage.affine_transform(img, A,
                                                    offset=np.dot(A_inv, b),
                                                    output_shape=target_shape,
                                                    order=interpolation_order)
    else:
        resampled_data = ndimage.affine_transform(data, A,
                                                  offset=np.dot(A_inv, b),
                                                  output_shape=target_shape,
                                                  order=interpolation_order)
    return Nifti1Image(resampled_data, target_affine)
