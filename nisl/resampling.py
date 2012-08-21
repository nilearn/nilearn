import numpy as np
from scipy import ndimage


def to_matrix_vector(transform):
    """Split a transform into it's matrix and vector components.

    The tranformation must be represented in homogeneous coordinates
    and is split into it's rotation matrix and translation vector
    components.

    Parameters
    ----------
    transform : ndarray
        Transform matrix in homogeneous coordinates.  Example, a 4x4
        transform representing rotations and translations in 3
        dimensions.

    Returns
    -------
    matrix, vector : ndarray
        The matrix and vector components of the transform matrix.  For
        an NxN transform, matrix will be N-1xN-1 and vector will be
        1xN-1.

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
    matrix : ndarray
        An NxN array representing the rotation matrix.
    vector : ndarray
        A 1xN array representing the translation.

    Returns
    -------
    xform : ndarray
        An N+1xN+1 transform matrix.

    See Also
    --------
    to_matrix_vector

    """

    nin, nout = matrix.shape
    t = np.zeros((nin + 1, nout + 1), matrix.dtype)
    t[0:nin, 0:nout] = matrix
    t[nin,   nout] = 1.
    t[0:nin, nout] = vector
    return t


def get_bounds(shape, affine):
    """ Return the world-space bounds occupied by an array given an affine.
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


def resample(data, affine, new_affine=None, new_shape=None,
                                    interpolation='continuous', copy=True):
    if copy:
        import copy
        data = copy.copy(data)
        affine = copy.copy(affine)
    if new_affine is None and new_shape is None:
        return (data, affine)
    if new_affine is None:
        new_affine = np.eye(4)
    if new_shape is None:
        new_shape = data.shape[:3]
    new_shape = list(new_shape)
    if new_affine.shape[0] == 3:
        # We have a 3D affine, we need to find out the offset and
        # shape to keep the same bounding box in the new space
        affine4d = np.eye(4)
        affine4d[:3, :3] = new_affine
        transform_affine = np.dot(np.linalg.inv(affine4d),
                                    affine,
                                 )
        # The bounding box in the new world, if no offset is given
        (xmin, xmax), (ymin, ymax), (zmin, zmax) = get_bounds(
                                                    data.shape[:3],
                                                    transform_affine,
                                                    )

        offset = np.array((xmin, ymin, zmin))
        offset = np.dot(new_affine, offset)
        new_affine = from_matrix_vector(new_affine, offset[:3])
        new_shape = (np.ceil(xmax - xmin) + 1,
                     np.ceil(ymax - ymin) + 1,
                     np.ceil(zmax - zmin) + 1, )
    if not len(new_shape) == 3:
        raise ValueError('The shape specified should be the shape '
            'the 3D grid, and thus of length 3. %s was specified'
            % new_shape)

    # Determine interpolation order
    if interpolation == 'continuous':
        interpolation_order = 3
    elif interpolation == 'nearest':
        interpolation_order = 0
    else:
        raise ValueError("interpolation must be either 'continuous' "
                         "or 'nearest'")

    if np.all(new_affine == affine):
        # Small trick to be more numericaly stable
        transform_affine = np.eye(4)
    else:
        transform_affine = np.dot(np.linalg.inv(affine), new_affine)
    A, b = to_matrix_vector(transform_affine)
    A_inv = np.linalg.inv(A)
    # If A is diagonal, ndimage.affine_transform is clever-enough
    # to use a better algorithm
    if np.all(np.diag(np.diag(A)) == A):
        A = np.diag(A)
    else:
        b = np.dot(A, b)
    # For images with dimensions larger than 3D:
    data_shape = list(data.shape)
    if len(data_shape) > 3:
        # Iter in a set of 3D volumes, as the interpolation problem is
        # separable in the extra dimensions. This reduces the
        # computational cost
        data = np.reshape(data, data_shape[:3] + [-1])
        data = np.rollaxis(data, 3)
        resampled_data = [ndimage.affine_transform(slice, A,
                                            offset=np.dot(A_inv, b),
                                            output_shape=new_shape,
                                            order=interpolation_order)
                            for slice in data]
        resampled_data = np.concatenate([d[..., np.newaxis]
                                        for d in resampled_data],
                                        axis=3)
        resampled_data = np.reshape(resampled_data, list(new_shape) +
                                        list(data_shape[3:]))
    else:
        resampled_data = ndimage.affine_transform(data, A,
                                            offset=np.dot(A_inv, b),
                                            output_shape=new_shape,
                                            order=interpolation_order)
    return (resampled_data, new_affine)
