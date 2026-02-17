# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Utility routines for working with points and affine transforms"""

from functools import reduce

import numpy as np


class AffineError(ValueError):
    """Errors in calculating or using affines"""

    # Inherits from ValueError to keep compatibility with ValueError previously
    # raised in append_diag
    pass


def apply_affine(aff, pts, inplace=False):
    """Apply affine matrix `aff` to points `pts`

    Returns result of application of `aff` to the *right* of `pts`.  The
    coordinate dimension of `pts` should be the last.

    For the 3D case, `aff` will be shape (4,4) and `pts` will have final axis
    length 3 - maybe it will just be N by 3. The return value is the
    transformed points, in this case::

        res = np.dot(aff[:3,:3], pts.T) + aff[:3,3:4]
        transformed_pts = res.T

    This routine is more general than 3D, in that `aff` can have any shape
    (N,N), and `pts` can have any shape, as long as the last dimension is for
    the coordinates, and is therefore length N-1.

    Parameters
    ----------
    aff : (N, N) array-like
        Homogeneous affine, for 3D points, will be 4 by 4. Contrary to first
        appearance, the affine will be applied on the left of `pts`.
    pts : (..., N-1) array-like
        Points, where the last dimension contains the coordinates of each
        point.  For 3D, the last dimension will be length 3.
    inplace : bool, optional
        If True, attempt to apply the affine directly to ``pts``.
        If False, or in-place application fails, a freshly allocated
        array will be returned.

    Returns
    -------
    transformed_pts : (..., N-1) array
        transformed points

    Examples
    --------
    >>> aff = np.array([[0,2,0,10],[3,0,0,11],[0,0,4,12],[0,0,0,1]])
    >>> pts = np.array([[1,2,3],[2,3,4],[4,5,6],[6,7,8]])
    >>> apply_affine(aff, pts) #doctest: +ELLIPSIS
    array([[14, 14, 24],
           [16, 17, 28],
           [20, 23, 36],
           [24, 29, 44]]...)

    Just to show that in the simple 3D case, it is equivalent to:

    >>> (np.dot(aff[:3,:3], pts.T) + aff[:3,3:4]).T #doctest: +ELLIPSIS
    array([[14, 14, 24],
           [16, 17, 28],
           [20, 23, 36],
           [24, 29, 44]]...)

    But `pts` can be a more complicated shape:

    >>> pts = pts.reshape((2,2,3))
    >>> apply_affine(aff, pts) #doctest: +ELLIPSIS
    array([[[14, 14, 24],
            [16, 17, 28]],
    <BLANKLINE>
           [[20, 23, 36],
            [24, 29, 44]]]...)
    """
    aff = np.asarray(aff)
    pts = np.asarray(pts)
    shape = pts.shape
    pts = pts.reshape((-1, shape[-1]))
    # rzs == rotations, zooms, shears
    rzs = aff[:-1, :-1]
    trans = aff[:-1, -1]

    if inplace:
        try:
            np.dot(pts, rzs.T, out=pts)
        except ValueError:
            inplace = False
        else:
            pts += trans[None, :]
    if not inplace:
        pts = pts @ rzs.T + trans[None, :]

    return pts.reshape(shape)


def to_matvec(transform):
    """Split a transform into its matrix and vector components

    The transformation must be represented in homogeneous coordinates and is
    split into its rotation matrix and translation vector components.

    Parameters
    ----------
    transform : array-like
        NxM transform matrix in homogeneous coordinates representing an affine
        transformation from an (N-1)-dimensional space to an (M-1)-dimensional
        space. An example is a 4x4 transform representing rotations and
        translations in 3 dimensions. A 4x3 matrix can represent a
        2-dimensional plane embedded in 3 dimensional space.

    Returns
    -------
    matrix : (N-1, M-1) array
        Matrix component of `transform`
    vector : (M-1,) array
        Vector component of `transform`

    See Also
    --------
    from_matvec

    Examples
    --------
    >>> aff = np.diag([2, 3, 4, 1])
    >>> aff[:3,3] = [9, 10, 11]
    >>> to_matvec(aff)
    (array([[2, 0, 0],
           [0, 3, 0],
           [0, 0, 4]]), array([ 9, 10, 11]))
    """
    transform = np.asarray(transform)
    ndimin = transform.shape[0] - 1
    ndimout = transform.shape[1] - 1
    matrix = transform[0:ndimin, 0:ndimout]
    vector = transform[0:ndimin, ndimout]
    return matrix, vector


def from_matvec(matrix, vector=None):
    """Combine a matrix and vector into an homogeneous affine

    Combine a rotation / scaling / shearing matrix and translation vector into
    a transform in homogeneous coordinates.

    Parameters
    ----------
    matrix : array-like
        An NxM array representing the the linear part of the transform.
        A transform from an M-dimensional space to an N-dimensional space.
    vector : None or array-like, optional
        None or an (N,) array representing the translation. None corresponds to
        an (N,) array of zeros.

    Returns
    -------
    xform : array
        An (N+1, M+1) homogeneous transform matrix.

    See Also
    --------
    to_matvec

    Examples
    --------
    >>> from_matvec(np.diag([2, 3, 4]), [9, 10, 11])
    array([[ 2,  0,  0,  9],
           [ 0,  3,  0, 10],
           [ 0,  0,  4, 11],
           [ 0,  0,  0,  1]])

    The `vector` argument is optional:

    >>> from_matvec(np.diag([2, 3, 4]))
    array([[2, 0, 0, 0],
           [0, 3, 0, 0],
           [0, 0, 4, 0],
           [0, 0, 0, 1]])
    """
    matrix = np.asarray(matrix)
    nin, nout = matrix.shape
    t = np.zeros((nin + 1, nout + 1), matrix.dtype)
    t[0:nin, 0:nout] = matrix
    t[nin, nout] = 1.0
    if vector is not None:
        t[0:nin, nout] = vector
    return t


def append_diag(aff, steps, starts=()):
    """Add diagonal elements `steps` and translations `starts` to affine

    Typical use is in expanding 4x4 affines to larger dimensions.  Nipy is the
    main consumer because it uses NxM affines, whereas we generally only use
    4x4 affines; the routine is here for convenience.

    Parameters
    ----------
    aff : 2D array
        N by M affine matrix
    steps : scalar or sequence
        diagonal elements to append.
    starts : scalar or sequence
        elements to append to last column of `aff`, representing translations
        corresponding to the `steps`. If empty, expands to a vector of zeros
        of the same length as `steps`

    Returns
    -------
    aff_plus : 2D array
        Now P by Q where L = ``len(steps)`` and P == N+L, Q=N+L

    Examples
    --------
    >>> aff = np.eye(4)
    >>> aff[:3,:3] = np.arange(9).reshape((3,3))
    >>> append_diag(aff, [9, 10], [99,100])
    array([[  0.,   1.,   2.,   0.,   0.,   0.],
           [  3.,   4.,   5.,   0.,   0.,   0.],
           [  6.,   7.,   8.,   0.,   0.,   0.],
           [  0.,   0.,   0.,   9.,   0.,  99.],
           [  0.,   0.,   0.,   0.,  10., 100.],
           [  0.,   0.,   0.,   0.,   0.,   1.]])
    """
    aff = np.asarray(aff)
    steps = np.atleast_1d(steps)
    starts = np.atleast_1d(starts)
    n_steps = len(steps)
    if len(starts) == 0:
        starts = np.zeros(n_steps, dtype=steps.dtype)
    elif len(starts) != n_steps:
        raise AffineError('Steps should have same length as starts')
    old_n_out, old_n_in = aff.shape[0] - 1, aff.shape[1] - 1
    # make new affine
    aff_plus = np.zeros((old_n_out + n_steps + 1, old_n_in + n_steps + 1), dtype=aff.dtype)
    # Get stuff from old affine
    aff_plus[:old_n_out, :old_n_in] = aff[:old_n_out, :old_n_in]
    aff_plus[:old_n_out, -1] = aff[:old_n_out, -1]
    # Add new diagonal elements
    for i, el in enumerate(steps):
        aff_plus[old_n_out + i, old_n_in + i] = el
    # Add translations for new affine, plus last 1
    aff_plus[old_n_out:, -1] = list(starts) + [1]
    return aff_plus


def dot_reduce(*args):
    r"""Apply numpy dot product function from right to left on arrays

    For passed arrays :math:`A, B, C, ... Z` returns :math:`A \dot B \dot C ...
    \dot Z` where "." is the numpy array dot product.

    Parameters
    ----------
    \*\*args : arrays
        Arrays that can be passed to numpy ``dot`` function

    Returns
    -------
    dot_product : array
        If there are N arguments, result of ``arg[0].dot(arg[1].dot(arg[2].dot
        ...  arg[N-2].dot(arg[N-1])))...``
    """
    return reduce(lambda x, y: np.dot(y, x), args[::-1])


def voxel_sizes(affine):
    r"""Return voxel size for each input axis given `affine`

    The `affine` is the mapping between array (voxel) coordinates and mm
    (world) coordinates.

    The voxel size for the first voxel (array) axis is the distance moved in
    world coordinates when moving one unit along the first voxel (array) axis.
    This is the distance between the world coordinate of voxel (0, 0, 0) and
    the world coordinate of voxel (1, 0, 0).  The world coordinate vector of
    voxel coordinate vector (0, 0, 0) is given by ``v0 = affine.dot((0, 0, 0,
    1)[:3]``.  The world coordinate vector of voxel vector (1, 0, 0) is
    ``v1_ax1 = affine.dot((1, 0, 0, 1))[:3]``.  The final 1 in the voxel
    vectors and the ``[:3]`` at the end are because the affine works on
    homogeneous coordinates.  The translations part of the affine is ``trans =
    affine[:3, 3]``, and the rotations, zooms and shearing part of the affine
    is ``rzs = affine[:3, :3]``. Because of the final 1 in the input voxel
    vector, ``v0 == rzs.dot((0, 0, 0)) + trans``, and ``v1_ax1 == rzs.dot((1,
    0, 0)) + trans``, and the difference vector is ``rzs.dot((0, 0, 0)) -
    rzs.dot((1, 0, 0)) == rzs.dot((1, 0, 0)) == rzs[:, 0]``.  The distance
    vectors in world coordinates between (0, 0, 0) and (1, 0, 0), (0, 1, 0),
    (0, 0, 1) are given by ``rzs.dot(np.eye(3)) = rzs``.  The voxel sizes are
    the Euclidean lengths of the distance vectors.  So, the voxel sizes are
    the Euclidean lengths of the columns of the affine (excluding the last row
    and column of the affine).

    Parameters
    ----------
    affine : 2D array-like
        Affine transformation array.  Usually shape (4, 4), but can be any 2D
        array.

    Returns
    -------
    vox_sizes : 1D array
        Voxel sizes for each input axis of affine.  Usually 1D array length 3,
        but in general has length (N-1) where input `affine` is shape (M, N).
    """
    top_left = affine[:-1, :-1]
    return np.sqrt(np.sum(top_left**2, axis=0))


def obliquity(affine):
    r"""Estimate the *obliquity* an affine's axes represent

    The term *obliquity* is defined here as the rotation of those axes with
    respect to the cardinal axes.
    This implementation is inspired by `AFNI's implementation
    <https://github.com/afni/afni/blob/b6a9f7a21c1f3231ff09efbd861f8975ad48e525/src/thd_coords.c#L660-L698>`_.
    For further details about *obliquity*, check `AFNI's documentation
    <https://sscc.nimh.nih.gov/sscc/dglen/Obliquity>`_.

    Parameters
    ----------
    affine : 2D array-like
        Affine transformation array.  Usually shape (4, 4), but can be any 2D
        array.

    Returns
    -------
    angles : 1D array-like
        The *obliquity* of each axis with respect to the cardinal axes, in radians.

    """
    vs = voxel_sizes(affine)
    best_cosines = np.abs(affine[:-1, :-1] / vs).max(axis=1)
    return np.arccos(best_cosines)


def rescale_affine(affine, shape, zooms, new_shape=None):
    """Return a new affine matrix with updated voxel sizes (zooms)

    This function preserves the rotations and shears of the original
    affine, as well as the RAS location of the central voxel of the
    image.

    Parameters
    ----------
    affine : (N, N) array-like
        NxN transform matrix in homogeneous coordinates representing an affine
        transformation from an (N-1)-dimensional space to an (N-1)-dimensional
        space. An example is a 4x4 transform representing rotations and
        translations in 3 dimensions.
    shape : (N-1,) array-like
        The extent of the (N-1) dimensions of the original space
    zooms : (N-1,) array-like
        The size of voxels of the output affine
    new_shape : (N-1,) array-like, optional
        The extent of the (N-1) dimensions of the space described by the
        new affine. If ``None``, use ``shape``.

    Returns
    -------
    affine : (N, N) array
        A new affine transform with the specified voxel sizes

    """
    shape = np.asarray(shape)
    new_shape = np.array(new_shape if new_shape is not None else shape)

    s = voxel_sizes(affine)
    rzs_out = affine[:3, :3] * zooms / s

    # Using xyz = A @ ijk, determine translation
    centroid = apply_affine(affine, (shape - 1) // 2)
    t_out = centroid - rzs_out @ ((new_shape - 1) // 2)
    return from_matvec(rzs_out, t_out)
