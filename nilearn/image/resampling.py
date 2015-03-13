"""
Utilities to resample a Niimg-like object
See http://nilearn.github.io/building_blocks/manipulating_mr_images.html#niimg.
"""
# Author: Gael Varoquaux, Alexandre Abraham, Michael Eickenberg
# License: simplified BSD

import warnings
from distutils.version import LooseVersion

import numpy as np
import scipy
from scipy import ndimage, linalg
from nibabel import Nifti1Image

from .. import _utils

###############################################################################
# Affine utils

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


def coord_transform(x, y, z, affine):
    """ Convert the x, y, z coordinates from one image space to another
        space.

        Parameters
        ----------
        x : number or ndarray
            The x coordinates in the input space
        y : number or ndarray
            The y coordinates in the input space
        z : number or ndarray
            The z coordinates in the input space
        affine : 2D 4x4 ndarray
            affine that maps from input to output space.

        Returns
        -------
        x : number or ndarray
            The x coordinates in the output space
        y : number or ndarray
            The y coordinates in the output space
        z : number or ndarray
            The z coordinates in the output space

        Warning: The x, y and z have their Talairach ordering, not 3D
        numy image ordering.
    """
    coords = np.c_[np.atleast_1d(x).flat,
                   np.atleast_1d(y).flat,
                   np.atleast_1d(z).flat,
                   np.ones_like(np.atleast_1d(z).flat)].T
    x, y, z, _ = np.dot(affine, coords)
    return x.squeeze(), y.squeeze(), z.squeeze()


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


def get_mask_bounds(img):
    """ Return the world-space bounds occupied by a mask.

        Parameters
        ----------
        img: Niimg-like object
            See http://nilearn.github.io/building_blocks/manipulating_mr_images.html#niimg.
            The image to inspect. Zero values are considered as
            background.

        Returns
        --------
        xmin, xmax, ymin, ymax, zmin, zmax: floats
            The world-space bounds (field of view) occupied by the
            non-zero values in the image

        Notes
        -----

        The image should have only one connect component.

        The affine should be diagonal or diagonal-permuted, use
        reorder_img to ensure that it is the case.

    """
    img = _utils.check_niimg(img)
    mask = img.get_data()
    affine = img.get_affine()
    (xmin, xmax), (ymin, ymax), (zmin, zmax) = get_bounds(mask.shape, affine)
    slices = ndimage.find_objects(mask)
    if len(slices) == 0:
        warnings.warn("empty mask", stacklevel=2)
    else:
        x_slice, y_slice, z_slice = slices[0]
        x_width, y_width, z_width = mask.shape
        xmin, xmax = (xmin + x_slice.start*(xmax - xmin)/x_width,
                    xmin + x_slice.stop *(xmax - xmin)/x_width)
        ymin, ymax = (ymin + y_slice.start*(ymax - ymin)/y_width,
                    ymin + y_slice.stop *(ymax - ymin)/y_width)
        zmin, zmax = (zmin + z_slice.start*(zmax - zmin)/z_width,
                    zmin + z_slice.stop *(zmax - zmin)/z_width)

    return xmin, xmax, ymin, ymax, zmin, zmax


class BoundingBoxError(ValueError):
    """This error is raised when a resampling transformation is
    incompatible with the given data.

    This can happen, for example, if the field of view of a target affine
    matrix does not contain any of the original data."""
    pass


###############################################################################
# Resampling

def _resample_one_img(data, A, A_inv, b, target_shape,
                      interpolation_order, out, copy=True):
    "Internal function for resample_img, do not use"
    if data.dtype.kind in ('i', 'u'):
        # Integers are always finite
        has_not_finite = False
    else:
        not_finite = np.logical_not(np.isfinite(data))
        has_not_finite = np.any(not_finite)
    if has_not_finite:
        warnings.warn("NaNs or infinite values are present in the data "
                        "passed to resample. This is a bad thing as they "
                        "make resampling ill-defined and much slower.",
                        RuntimeWarning, stacklevel=2)
        if copy:
            # We need to do a copy to avoid modifying the input
            # array
            data = data.copy()
        #data[not_finite] = 0
        from ..masking import _extrapolate_out_mask
        data = _extrapolate_out_mask(data, np.logical_not(not_finite),
                                     iterations=2)[0]

    # See https://github.com/nilearn/nilearn/issues/346 Copying the
    # array makes it C continuous and as such the int32 index in the C
    # code is a lot less likely to overflow
    if (LooseVersion(scipy.__version__) < LooseVersion('0.14.1')):
        data = data.copy()

    # The resampling itself
    ndimage.affine_transform(data, A,
                             offset=np.dot(A_inv, b),
                             output_shape=target_shape,
                             output=out,
                             order=interpolation_order)

    # Bug in ndimage.affine_transform when out does not have native endianness
    # see https://github.com/nilearn/nilearn/issues/275
    # Bug was fixed in scipy 0.15
    if (LooseVersion(scipy.__version__) < LooseVersion('0.15') and
        not out.dtype.isnative):
        out.byteswap(True)

    if has_not_finite:
        # We need to resample the mask of not_finite values
        not_finite = ndimage.affine_transform(not_finite, A,
                                            offset=np.dot(A_inv, b),
                                            output_shape=target_shape,
                                            order=0)
        out[not_finite] = np.nan
    return out


def resample_img(img, target_affine=None, target_shape=None,
                 interpolation='continuous', copy=True, order="F"):
    """Resample a Niimg-like object

    Parameters
    ----------
    img: Niimg-like object
        See http://nilearn.github.io/building_blocks/manipulating_mr_images.html#niimg.
        Image(s) to resample.

    target_affine: numpy.ndarray, optional
        If specified, the image is resampled corresponding to this new affine.
        target_affine can be a 3x3 or a 4x4 matrix. (See notes)

    target_shape: tuple or list, optional
        If specified, the image will be resized to match this new shape.
        len(target_shape) must be equal to 3.
        If target_shape is specified, a target_affine of shape (4, 4)
        must also be given. (See notes)

    interpolation: str, optional
        Can be 'continuous' (default) or 'nearest'. Indicate the resample method

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

    Notes
    =====

    **BoundingBoxError**
    If a 4x4 transformation matrix (target_affine) is given and all of the
    transformed data points have a negative voxel index along one of the
    axis, then none of the data will be visible in the transformed image
    and a BoundingBoxError will be raised.

    If a 4x4 transformation matrix (target_affine) is given and no target
    shape is provided, the resulting image will have voxel coordinate
    (0, 0, 0) in the affine offset (4th column of target affine) and will
    extend far enough to contain all the visible data and a margin of one
    voxel.

    **3x3 transformation matrices**
    If a 3x3 transformation matrix is given as target_affine, it will be
    assumed to represent the three coordinate axes of the target space. In
    this case the affine offset (4th column of a 4x4 transformation matrix)
    as well as the target_shape will be inferred by resample_img, such that
    the resulting field of view is the closest possible (with a margin of
    1 voxel) bounding box around the transformed data.

    In certain cases one may want to obtain a transformed image with the
    closest bounding box around the data, which at the same time respects
    a voxel grid defined by a 4x4 affine transformation matrix. In this
    case, one resamples the image using this function given the target
    affine and no target shape. One then uses crop_img on the result.

    **NaNs and infinite values**
    This function handles gracefully NaNs and infinite values in the input
    data, however they make the execution of the function much slower.
    """
    # Do as many checks as possible before loading data, to avoid potentially
    # costly calls before raising an exception.
    if target_shape is not None and target_affine is None:
        raise ValueError("If target_shape is specified, target_affine should"
                         " be specified too.")

    if target_shape is not None and not len(target_shape) == 3:
        raise ValueError('The shape specified should be the shape of '
                         'the 3D grid, and thus of length 3. %s was specified'
                         % str(target_shape))

    if target_shape is not None and target_affine.shape == (3, 3):
        raise ValueError("Given target shape without anchor vector: "
                         "Affine shape should be (4, 4) and not (3, 3)")

    if interpolation == 'continuous':
        interpolation_order = 3
    elif interpolation == 'nearest':
        interpolation_order = 0
    else:
        message = ("interpolation must be either 'continuous' "
                   "or 'nearest' but it was set to '{0}'").format(interpolation)
        raise ValueError(message)

    if isinstance(img, basestring):
        # Avoid a useless copy
        input_img_is_string = True
    else:
        input_img_is_string = False

    # noop cases
    img = _utils.check_niimg(img)

    if target_affine is None and target_shape is None:
        if copy and not input_img_is_string:
            img = _utils.copy_img(img)
        return img
    if target_affine is not None:
        target_affine = np.asarray(target_affine)

    shape = _utils._get_shape(img)
    affine = img.get_affine()

    if (np.all(np.array(target_shape) == shape[:3]) and
            np.allclose(target_affine, affine)):
        if copy and not input_img_is_string:
            img = _utils.copy_img(img)
        return img

    # We now know that some resampling must be done.
    # The value of "copy" is of no importance: output is always a separate
    # array.
    data = img.get_data()

    # Get a bounding box for the transformed data
    # Embed target_affine in 4x4 shape if necessary
    if target_affine.shape == (3, 3):
        missing_offset = True
        target_affine_tmp = np.eye(4)
        target_affine_tmp[:3, :3] = target_affine
        target_affine = target_affine_tmp
    else:
        missing_offset = False
        target_affine = target_affine.copy()
    transform_affine = np.linalg.inv(target_affine).dot(affine)
    (xmin, xmax), (ymin, ymax), (zmin, zmax) = get_bounds(
        data.shape[:3], transform_affine)

    # if target_affine is (3, 3), then calculate
    # offset from bounding box and update bounding box
    # to be in the voxel coordinates of the calculated 4x4 affine
    if missing_offset:
        offset = target_affine[:3, :3].dot([xmin, ymin, zmin])
        target_affine[:3, 3] = offset
        (xmin, xmax), (ymin, ymax), (zmin, zmax) = (
            (0, xmax - xmin), (0, ymax - ymin), (0, zmax - zmin))

    # if target_shape is not given (always the case with 3x3
    # transformation matrix and sometimes the case with 4x4
    # transformation matrix), then set it to contain the bounding
    # box by a margin of 1 voxel
    if target_shape is None:
        target_shape = (int(np.ceil(xmax)) + 1,
                        int(np.ceil(ymax)) + 1,
                        int(np.ceil(zmax)) + 1)

    # Check whether transformed data is actually within the FOV
    # of the target affine
    if xmax < 0 or ymax < 0 or zmax < 0:
        raise BoundingBoxError("The field of view given "
                               "by the target affine does "
                               "not contain any of the data")

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
    # Make sure that we have a list here
    if isinstance(target_shape, np.ndarray):
        target_shape = target_shape.tolist()
    target_shape = tuple(target_shape)
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
            _resample_one_img(data[all_img + ind], A, A_inv, b, target_shape,
                      interpolation_order,
                      out=resampled_data[all_img + ind],
                      copy=not input_img_is_string)
    else:
        resampled_data = np.empty(target_shape, data.dtype)
        _resample_one_img(data, A, A_inv, b, target_shape,
                          interpolation_order,
                          out=resampled_data,
                          copy=not input_img_is_string)

    return Nifti1Image(resampled_data, target_affine)


def reorder_img(img, resample=None):
    """Returns an image with the affine diagonal (by permuting axes).
    The orientation of the new image will be RAS (Right, Anterior, Superior).
    If it is impossible to get xyz ordering by permuting the axes, a
    'ValueError' is raised.

        Parameters
        -----------
        img: Niimg-like object
            See http://nilearn.github.io/building_blocks/manipulating_mr_images.html#niimg.
            Image to reorder.

        resample: None or string in {'continuous', 'nearest'}, optional
            If resample is None (default), no resampling is performed, the
            axes are only permuted.
            Otherwise resampling is performed and 'resample' will
            be passed as the 'interpolation' argument into
            resample_img.

    """
    img = _utils.check_niimg(img)
    # The copy is needed in order not to modify the input img affine
    # see https://github.com/nilearn/nilearn/issues/325 for a concrete bug
    affine = img.get_affine().copy()
    A, b = to_matrix_vector(affine)

    if not np.all((np.abs(A) > 0.001).sum(axis=0) == 1):
        # The affine is not nearly diagonal
        if resample is None:
            raise ValueError('Cannot reorder the axes: '
                             'the image affine contains rotations')
        else:
            # Identify the voxel size using a QR decomposition of the
            # affine
            R, Q = np.linalg.qr(affine[:3, :3])
            target_affine = np.diag(np.abs(np.diag(Q))[
                                                np.abs(R).argmax(axis=1)])
            return resample_img(img, target_affine=target_affine,
                                interpolation=resample)

    axis_numbers = np.argmax(np.abs(A), axis=0)
    data = img.get_data()
    while not np.all(np.sort(axis_numbers) == axis_numbers):
        first_inversion = np.argmax(np.diff(axis_numbers)<0)
        axis1 = first_inversion + 1
        axis2 = first_inversion
        data = np.swapaxes(data, axis1, axis2)
        order = np.array((0, 1, 2, 3))
        order[axis1] = axis2
        order[axis2] = axis1
        affine = affine.T[order].T
        A, b = to_matrix_vector(affine)
        axis_numbers = np.argmax(np.abs(A), axis=0)

    # Now make sure the affine is positive
    pixdim = np.diag(A).copy()
    if pixdim[0] < 0:
        b[0] = b[0] + pixdim[0]*(data.shape[0] - 1)
        pixdim[0] = -pixdim[0]
        slice1 = slice(None, None, -1)
    else:
        slice1 = slice(None, None, None)
    if pixdim[1] < 0:
        b[1] = b[1] + 1 + pixdim[1]*(data.shape[1] - 1)
        pixdim[1] = -pixdim[1]
        slice2 = slice(None, None, -1)
    else:
        slice2 = slice(None, None, None)
    if pixdim[2] < 0:
        b[2] = b[2] + 1 + pixdim[2]*(data.shape[2] - 1)
        pixdim[2] = -pixdim[2]
        slice3 = slice(None, None, -1)
    else:
        slice3 = slice(None, None, None)
    data = data[slice1, slice2, slice3]
    affine = from_matrix_vector(np.diag(pixdim), b)

    niimg = Nifti1Image(data, affine)

    return niimg



