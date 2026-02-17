# emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the NiBabel package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Utilities for calculating and applying affine orientations"""

import numpy as np
import numpy.linalg as npl

from .deprecated import deprecate_with_version


class OrientationError(Exception):
    pass


def io_orientation(affine, tol=None):
    """Orientation of input axes in terms of output axes for `affine`

    Valid for an affine transformation from ``p`` dimensions to ``q``
    dimensions (``affine.shape == (q + 1, p + 1)``).

    The calculated orientations can be used to transform associated
    arrays to best match the output orientations. If ``p`` > ``q``, then
    some of the output axes should be considered dropped in this
    orientation.

    Parameters
    ----------
    affine : (q+1, p+1) ndarray-like
       Transformation affine from ``p`` inputs to ``q`` outputs.  Usually this
       will be a shape (4,4) matrix, transforming 3 inputs to 3 outputs, but
       the code also handles the more general case
    tol : {None, float}, optional
       threshold below which SVD values of the affine are considered zero. If
       `tol` is None, and ``S`` is an array with singular values for `affine`,
       and ``eps`` is the epsilon value for datatype of ``S``, then `tol` set
       to ``S.max() * max((q, p)) * eps``

    Returns
    -------
    orientations : (p, 2) ndarray
       one row per input axis, where the first value in each row is the closest
       corresponding output axis. The second value in each row is 1 if the
       input axis is in the same direction as the corresponding output axis and
       -1 if it is in the opposite direction.  If a row is [np.nan, np.nan],
       which can happen when p > q, then this row should be considered dropped.
    """
    affine = np.asarray(affine)
    q, p = affine.shape[0] - 1, affine.shape[1] - 1
    # extract the underlying rotation, zoom, shear matrix
    RZS = affine[:q, :p]
    zooms = np.sqrt(np.sum(RZS * RZS, axis=0))
    # Zooms can be zero, in which case all elements in the column are zero, and
    # we can leave them as they are
    zooms[zooms == 0] = 1
    RS = RZS / zooms
    # Transform below is polar decomposition, returning the closest
    # shearless matrix R to RS
    P, S, Qs = npl.svd(RS, full_matrices=False)
    # Threshold the singular values to determine the rank.
    if tol is None:
        tol = S.max() * max(RS.shape) * np.finfo(S.dtype).eps
    keep = S > tol
    R = np.dot(P[:, keep], Qs[keep])
    # the matrix R is such that np.dot(R,R.T) is projection onto the
    # columns of P[:,keep] and np.dot(R.T,R) is projection onto the rows
    # of Qs[keep].  R (== np.dot(R, np.eye(p))) gives rotation of the
    # unit input vectors to output coordinates.  Therefore, the row
    # index of abs max R[:,N], is the output axis changing most as input
    # axis N changes.  In case there are ties, we choose the axes
    # iteratively, removing used axes from consideration as we go
    ornt = np.ones((p, 2), dtype=np.int8) * np.nan
    for in_ax in range(p):
        col = R[:, in_ax]
        if not np.allclose(col, 0):
            out_ax = np.argmax(np.abs(col))
            ornt[in_ax, 0] = out_ax
            assert col[out_ax] != 0
            if col[out_ax] < 0:
                ornt[in_ax, 1] = -1
            else:
                ornt[in_ax, 1] = 1
            # remove the identified axis from further consideration, by
            # zeroing out the corresponding row in R
            R[out_ax, :] = 0
    return ornt


def ornt_transform(start_ornt, end_ornt):
    """Return the orientation that transforms from `start_ornt` to `end_ornt`.

    Parameters
    ----------
    start_ornt : (n,2) orientation array
        Initial orientation.

    end_ornt : (n,2) orientation array
        Final orientation.

    Returns
    -------
    orientations : (p, 2) ndarray
       The orientation that will transform the `start_ornt` to the `end_ornt`.
    """
    start_ornt = np.asarray(start_ornt)
    end_ornt = np.asarray(end_ornt)
    if start_ornt.shape != end_ornt.shape:
        raise ValueError('The orientations must have the same shape')
    if start_ornt.shape[1] != 2:
        raise ValueError(f'Invalid shape for an orientation: {start_ornt.shape}')
    result = np.empty_like(start_ornt)
    for end_in_idx, (end_out_idx, end_flip) in enumerate(end_ornt):
        for start_in_idx, (start_out_idx, start_flip) in enumerate(start_ornt):
            if end_out_idx == start_out_idx:
                if start_flip == end_flip:
                    flip = 1
                else:
                    flip = -1
                result[start_in_idx, :] = [end_in_idx, flip]
                break
        else:
            raise ValueError(f'Unable to find out axis {end_out_idx} in start_ornt')
    return result


def apply_orientation(arr, ornt):
    """Apply transformations implied by `ornt` to the first
    n axes of the array `arr`

    Parameters
    ----------
    arr : array-like of data with ndim >= n
    ornt : (n,2) orientation array
       orientation transform. ``ornt[N,1]` is flip of axis N of the
       array implied by `shape`, where 1 means no flip and -1 means
       flip.  For example, if ``N==0`` and ``ornt[0,1] == -1``, and
       there's an array ``arr`` of shape `shape`, the flip would
       correspond to the effect of ``np.flipud(arr)``.  ``ornt[:,0]`` is
       the transpose that needs to be done to the implied array, as in
       ``arr.transpose(ornt[:,0])``

    Returns
    -------
    t_arr : ndarray
       data array `arr` transformed according to ornt
    """
    t_arr = np.asarray(arr)
    ornt = np.asarray(ornt)
    n = ornt.shape[0]
    if t_arr.ndim < n:
        raise OrientationError('Data array has fewer dimensions than orientation')
    # no coordinates can be dropped for applying the orientations
    if np.any(np.isnan(ornt[:, 0])):
        raise OrientationError('Cannot drop coordinates when applying orientation to data')
    # apply ornt transformations
    for ax, flip in enumerate(ornt[:, 1]):
        if flip == -1:
            t_arr = np.flip(t_arr, axis=ax)
    full_transpose = np.arange(t_arr.ndim)
    # ornt indicates the transpose that has occurred - we reverse it
    full_transpose[:n] = np.argsort(ornt[:, 0])
    t_arr = t_arr.transpose(full_transpose)
    return t_arr


def inv_ornt_aff(ornt, shape):
    """Affine transform reversing transforms implied in `ornt`

    Imagine you have an array ``arr`` of shape `shape`, and you apply the
    transforms implied by `ornt` (more below), to get ``tarr``.
    ``tarr`` may have a different shape ``shape_prime``.  This routine
    returns the affine that will take a array coordinate for ``tarr``
    and give you the corresponding array coordinate in ``arr``.

    Parameters
    ----------
    ornt : (p, 2) ndarray
       orientation transform. ``ornt[P, 1]` is flip of axis N of the array
       implied by `shape`, where 1 means no flip and -1 means flip.  For
       example, if ``P==0`` and ``ornt[0, 1] == -1``, and there's an array
       ``arr`` of shape `shape`, the flip would correspond to the effect of
       ``np.flipud(arr)``.  ``ornt[:,0]`` gives us the (reverse of the)
       transpose that has been done to ``arr``.  If there are any NaNs in
       `ornt`, we raise an ``OrientationError`` (see notes)
    shape : length p sequence
       shape of array you may transform with `ornt`

    Returns
    -------
    transform_affine : (p + 1, p + 1) ndarray
       An array ``arr`` (shape `shape`) might be transformed according to
       `ornt`, resulting in a transformed array ``tarr``.  `transformed_affine`
       is the transform that takes you from array coordinates in ``tarr`` to
       array coordinates in ``arr``.

    Notes
    -----
    If a row in `ornt` contains NaN, this means that the input row does not
    influence the output space, and is thus effectively dropped from the output
    space.  In that case one ``tarr`` coordinate maps to many ``arr``
    coordinates, we can't invert the transform, and we raise an error
    """
    ornt = np.asarray(ornt)
    if np.any(np.isnan(ornt)):
        raise OrientationError('We cannot invert orientation transform')
    p = ornt.shape[0]
    shape = np.array(shape)[:p]
    # ornt implies a flip, followed by a transpose.   We need the affine
    # that inverts these.  Thus we need the affine that first undoes the
    # effect of the transpose, then undoes the effects of the flip.
    # ornt indicates the transpose that has occurred to get the current
    # ordering, relative to canonical, so we just use that.
    # undo_reorder is a row permutatation matrix
    axis_transpose = [int(v) for v in ornt[:, 0]]
    undo_reorder = np.eye(p + 1)[axis_transpose + [p], :]
    undo_flip = np.diag(list(ornt[:, 1]) + [1.0])
    center_trans = -(shape - 1) / 2.0
    undo_flip[:p, p] = (ornt[:, 1] * center_trans) - center_trans
    return np.dot(undo_flip, undo_reorder)


@deprecate_with_version(
    'flip_axis is deprecated. Please use numpy.flip instead.',
    '3.2',
    '5.0',
)
def flip_axis(arr, axis=0):
    """Flip contents of `axis` in array `arr`

    Equivalent to ``np.flip(arr, axis)``.

    Parameters
    ----------
    arr : array-like
    axis : int, optional
       axis to flip.  Default `axis` == 0

    Returns
    -------
    farr : array
       Array with axis `axis` flipped
    """
    return np.flip(arr, axis)


def ornt2axcodes(ornt, labels=None):
    """Convert orientation `ornt` to labels for axis directions

    Parameters
    ----------
    ornt : (N,2) array-like
        orientation array - see io_orientation docstring
    labels : optional, None or sequence of (2,) sequences
        (2,) sequences are labels for (beginning, end) of output axis.  That
        is, if the first row in `ornt` is ``[1, 1]``, and the second (2,)
        sequence in `labels` is ('back', 'front') then the first returned axis
        code will be ``'front'``.  If the first row in `ornt` had been
        ``[1, -1]`` then the first returned value would have been ``'back'``.
        If None, equivalent to ``(('L','R'),('P','A'),('I','S'))`` - that is -
        RAS axes.

    Returns
    -------
    axcodes : (N,) tuple
        labels for positive end of voxel axes.  Dropped axes get a label of
        None.

    Examples
    --------
    >>> ornt2axcodes([[1, 1],[0,-1],[2,1]], (('L','R'),('B','F'),('D','U')))
    ('F', 'L', 'U')
    """
    if labels is None:
        labels = list(zip('LPI', 'RAS'))
    axcodes = []
    for axno, direction in np.asarray(ornt):
        if np.isnan(axno):
            axcodes.append(None)
            continue
        axint = int(np.round(axno))
        if axint != axno:
            raise ValueError(f'Non integer axis number {axno:f}')
        elif direction == 1:
            axcode = labels[axint][1]
        elif direction == -1:
            axcode = labels[axint][0]
        else:
            raise ValueError('Direction should be -1 or 1')
        axcodes.append(axcode)
    return tuple(axcodes)


def axcodes2ornt(axcodes, labels=None):
    """Convert axis codes `axcodes` to an orientation

    Parameters
    ----------
    axcodes : (N,) tuple
        axis codes - see ornt2axcodes docstring
    labels : optional, None or sequence of (2,) sequences
        (2,) sequences are labels for (beginning, end) of output axis.  That
        is, if the first element in `axcodes` is ``front``, and the second
        (2,) sequence in `labels` is ('back', 'front') then the first
        row of `ornt` will be ``[1, 1]``. If None, equivalent to
        ``(('L','R'),('P','A'),('I','S'))`` - that is - RAS axes.

    Returns
    -------
    ornt : (N,2) array-like
        orientation array - see io_orientation docstring

    Examples
    --------
    >>> axcodes2ornt(('F', 'L', 'U'), (('L','R'),('B','F'),('D','U')))
    array([[ 1.,  1.],
           [ 0., -1.],
           [ 2.,  1.]])
    """
    labels = list(zip('LPI', 'RAS')) if labels is None else labels
    allowed_labels = sum(map(list, labels), [None])
    if len(allowed_labels) != len(set(allowed_labels)):
        raise ValueError(f'Duplicate labels in {allowed_labels}')
    if not set(axcodes).issubset(allowed_labels):
        raise ValueError(f'Not all axis codes {list(axcodes)} in label set {allowed_labels}')
    n_axes = len(axcodes)
    ornt = np.ones((n_axes, 2), dtype=np.int8) * np.nan
    for code_idx, code in enumerate(axcodes):
        for label_idx, codes in enumerate(labels):
            if code is None:
                continue
            if code in codes:
                if code == codes[0]:
                    ornt[code_idx, :] = [label_idx, -1]
                else:
                    ornt[code_idx, :] = [label_idx, 1]
                break
    return ornt


def aff2axcodes(aff, labels=None, tol=None):
    """axis direction codes for affine `aff`

    Parameters
    ----------
    aff : (N,M) array-like
        affine transformation matrix
    labels : optional, None or sequence of (2,) sequences
        Labels for negative and positive ends of output axes of `aff`.  See
        docstring for ``ornt2axcodes`` for more detail
    tol : None or float
        Tolerance for SVD of affine - see ``io_orientation`` for more detail.

    Returns
    -------
    axcodes : (N,) tuple
        labels for positive end of voxel axes.  Dropped axes get a label of
        None.

    Examples
    --------
    >>> aff = [[0,1,0,10],[-1,0,0,20],[0,0,1,30],[0,0,0,1]]
    >>> aff2axcodes(aff, (('L','R'),('B','F'),('D','U')))
    ('B', 'R', 'U')
    """
    ornt = io_orientation(aff, tol)
    return ornt2axcodes(ornt, labels)
