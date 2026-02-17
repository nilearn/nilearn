# emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the NiBabel package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Image processing functions

Image processing functions for:

    * smoothing
    * resampling
    * converting SD to and from FWHM

Smoothing and resampling routines need scipy.
"""

import numpy as np
import numpy.linalg as npl

from .optpkg import optional_package

spnd = optional_package('scipy.ndimage')[0]

from .affines import AffineError, append_diag, from_matvec, rescale_affine, to_matvec
from .imageclasses import spatial_axes_first
from .nifti1 import Nifti1Image
from .orientations import axcodes2ornt, io_orientation, ornt_transform
from .spaces import vox2out_vox

SIGMA2FWHM = np.sqrt(8 * np.log(2))


def fwhm2sigma(fwhm):
    """Convert a FWHM value to sigma in a Gaussian kernel.

    Parameters
    ----------
    fwhm : array-like
       FWHM value or values

    Returns
    -------
    sigma : array or float
       sigma values corresponding to `fwhm` values

    Examples
    --------
    >>> sigma = fwhm2sigma(6)
    >>> sigmae = fwhm2sigma([6, 7, 8])
    >>> sigma == sigmae[0]
    True
    """
    return np.asarray(fwhm) / SIGMA2FWHM


def sigma2fwhm(sigma):
    """Convert a sigma in a Gaussian kernel to a FWHM value

    Parameters
    ----------
    sigma : array-like
       sigma value or values

    Returns
    -------
    fwhm : array or float
       fwhm values corresponding to `sigma` values

    Examples
    --------
    >>> fwhm = sigma2fwhm(3)
    >>> fwhms = sigma2fwhm([3, 4, 5])
    >>> fwhm == fwhms[0]
    True
    """
    return np.asarray(sigma) * SIGMA2FWHM


def adapt_affine(affine, n_dim):
    """Adapt input / output dimensions of spatial `affine` for `n_dims`

    Adapts a spatial (4, 4) affine that is being applied to an image with fewer
    than 3 spatial dimensions, or more than 3 dimensions.  If there are more
    than three dimensions, assume an identity transformation for these
    dimensions.

    Parameters
    ----------
    affine : array-like
        affine transform. Usually shape (4, 4).  For what follows ``N, M =
        affine.shape``
    n_dims : int
        Number of dimensions of underlying array, and therefore number of input
        dimensions for affine.

    Returns
    -------
    adapted : shape (M, n_dims+1) array
        Affine array adapted to number of input dimensions.  Columns of the
        affine corresponding to missing input dimensions have been dropped,
        columns corresponding to extra input dimensions have an extra identity
        column added
    """
    affine = np.asarray(affine)
    rzs, trans = to_matvec(affine)
    # For missing input dimensions, drop columns in rzs
    rzs = rzs[:, :n_dim]
    adapted = from_matvec(rzs, trans)
    n_extra_columns = n_dim - adapted.shape[1] + 1
    if n_extra_columns > 0:
        adapted = append_diag(adapted, np.ones((n_extra_columns,)))
    return adapted


def resample_from_to(
    from_img,
    to_vox_map,
    order=3,
    mode='constant',
    cval=0.0,
    out_class=Nifti1Image,
):
    """Resample image `from_img` to mapped voxel space `to_vox_map`

    Resample using N-d spline interpolation.

    Parameters
    ----------
    from_img : object
        Object having attributes ``dataobj``, ``affine``, ``header`` and
        ``shape``. If `out_class` is not None, ``img.__class__`` should be able
        to construct an image from data, affine and header.
    to_vox_map : image object or length 2 sequence
        If object, has attributes ``shape`` giving input voxel shape, and
        ``affine`` giving mapping of input voxels to output space. If length 2
        sequence, elements are (shape, affine) with same meaning as above. The
        affine is a (4, 4) array-like.
    order : int, optional
        The order of the spline interpolation, default is 3.  The order has to
        be in the range 0-5 (see ``scipy.ndimage.affine_transform``)
    mode : str, optional
        Points outside the boundaries of the input are filled according
        to the given mode ('constant', 'nearest', 'reflect' or 'wrap').
        Default is 'constant' (see ``scipy.ndimage.affine_transform``)
    cval : scalar, optional
        Value used for points outside the boundaries of the input if
        ``mode='constant'``. Default is 0.0 (see
        ``scipy.ndimage.affine_transform``)
    out_class : None or SpatialImage class, optional
        Class of output image.  If None, use ``from_img.__class__``.

    Returns
    -------
    out_img : object
        Image of instance specified by `out_class`, containing data output from
        resampling `from_img` into axes aligned to the output space of
        ``from_img.affine``
    """
    # This check requires `shape` attribute of image
    if not spatial_axes_first(from_img):
        raise ValueError(
            f'Cannot predict position of spatial axes for Image type {type(from_img)}'
        )
    try:
        to_shape, to_affine = to_vox_map.shape, to_vox_map.affine
    except AttributeError:
        to_shape, to_affine = to_vox_map
    a_to_affine = adapt_affine(to_affine, len(to_shape))
    if out_class is None:
        out_class = from_img.__class__
    from_n_dim = len(from_img.shape)
    if from_n_dim < 3:
        raise AffineError('from_img must be at least 3D')
    a_from_affine = adapt_affine(from_img.affine, from_n_dim)
    to_vox2from_vox = npl.inv(a_from_affine).dot(a_to_affine)
    rzs, trans = to_matvec(to_vox2from_vox)
    data = spnd.affine_transform(
        from_img.dataobj, rzs, trans, to_shape, order=order, mode=mode, cval=cval
    )
    return out_class(data, to_affine, from_img.header)


def resample_to_output(
    in_img,
    voxel_sizes=None,
    order=3,
    mode='constant',
    cval=0.0,
    out_class=Nifti1Image,
):
    """Resample image `in_img` to output voxel axes (world space)

    Parameters
    ----------
    in_img : object
        Object having attributes ``dataobj``, ``affine``, ``header``. If
        `out_class` is not None, ``img.__class__`` should be able to construct
        an image from data, affine and header.
    voxel_sizes : None or sequence
        Gives the diagonal entries of ``out_img.affine` (except the trailing 1
        for the homogeneous coordinates) (``out_img.affine ==
        np.diag(voxel_sizes + [1])``). If None, return identity
        `out_img.affine`.  If scalar, interpret as vector ``[voxel_sizes] *
        len(in_img.shape)``.
    order : int, optional
        The order of the spline interpolation, default is 3.  The order has to
        be in the range 0-5 (see ``scipy.ndimage.affine_transform``).
    mode : str, optional
        Points outside the boundaries of the input are filled according to the
        given mode ('constant', 'nearest', 'reflect' or 'wrap').  Default is
        'constant' (see ``scipy.ndimage.affine_transform``).
    cval : scalar, optional
        Value used for points outside the boundaries of the input if
        ``mode='constant'``. Default is 0.0 (see
        ``scipy.ndimage.affine_transform``).
    out_class : None or SpatialImage class, optional
        Class of output image.  If None, use ``in_img.__class__``.

    Returns
    -------
    out_img : object
        Image of instance specified by `out_class`, containing data output from
        resampling `in_img` into axes aligned to the output space of
        ``in_img.affine``
    """
    if out_class is None:
        out_class = in_img.__class__
    in_shape = in_img.shape
    n_dim = len(in_shape)
    if voxel_sizes is not None:
        voxel_sizes = np.asarray(voxel_sizes)
        if voxel_sizes.ndim == 0:  # Scalar
            voxel_sizes = np.repeat(voxel_sizes, n_dim)
    # Allow 2D images by promoting to 3D.  We might want to see what a slice
    # looks like when resampled into world coordinates
    if n_dim < 3:  # Expand image to 3D, make voxel sizes match
        new_shape = in_shape + (1,) * (3 - n_dim)
        data = np.asanyarray(in_img.dataobj).reshape(new_shape)  # 2D data should be small
        in_img = out_class(data, in_img.affine, in_img.header)
        if voxel_sizes is not None and len(voxel_sizes) == n_dim:
            # Need to pad out voxel sizes to match new image dimensions
            voxel_sizes = tuple(voxel_sizes) + (1,) * (3 - n_dim)
    out_vox_map = vox2out_vox((in_img.shape, in_img.affine), voxel_sizes)
    return resample_from_to(in_img, out_vox_map, order, mode, cval, out_class)


def smooth_image(
    img,
    fwhm,
    mode='nearest',
    cval=0.0,
    out_class=Nifti1Image,
):
    """Smooth image `img` along voxel axes by FWHM `fwhm` millimeters

    Parameters
    ----------
    img : object
        Object having attributes ``dataobj``, ``affine``, ``header`` and
        ``shape``. If `out_class` is not None, ``img.__class__`` should be able
        to construct an image from data, affine and header.
    fwhm : scalar or length 3 sequence
        FWHM *in mm* over which to smooth.  The smoothing applies to the voxel
        axes, not to the output axes, but is in millimeters.  The function
        adjusts the FWHM to voxels using the voxel sizes calculated from the
        affine. A scalar implies the same smoothing across the spatial
        dimensions of the image, but 0 smoothing over any further dimensions
        such as time.  A vector should be the same length as the number of
        image dimensions.
    mode : str, optional
        Points outside the boundaries of the input are filled according
        to the given mode ('constant', 'nearest', 'reflect' or 'wrap').
        Default is 'nearest'. This is different from the default for
        ``scipy.ndimage.affine_transform``, which is 'constant'. 'nearest'
        might be a better choice when smoothing to the edge of an image where
        there is still strong brain signal, otherwise this signal will get
        blurred towards zero.
    cval : scalar, optional
        Value used for points outside the boundaries of the input if
        ``mode='constant'``. Default is 0.0 (see
        ``scipy.ndimage.affine_transform``).
    out_class : None or SpatialImage class, optional
        Class of output image.  If None, use ``img.__class__``.

    Returns
    -------
    smoothed_img : object
        Image of instance specified by `out_class`, containing data output from
        smoothing `img` data by given FWHM kernel.
    """
    # This check requires `shape` attribute of image
    if not spatial_axes_first(img):
        raise ValueError(f'Cannot predict position of spatial axes for Image type {type(img)}')
    if out_class is None:
        out_class = img.__class__
    n_dim = len(img.shape)
    # TODO: make sure time axis is last
    # Pad out fwhm from scalar, adding 0 for fourth etc (time etc) dimensions
    fwhm = np.asarray(fwhm)
    if fwhm.size == 1:
        fwhm_scalar = fwhm
        fwhm = np.zeros((n_dim,))
        fwhm[:3] = fwhm_scalar
    # Voxel sizes
    RZS = img.affine[:, :n_dim]
    vox = np.sqrt(np.sum(RZS**2, 0))
    # Smoothing in terms of voxels
    vox_fwhm = fwhm / vox
    vox_sd = fwhm2sigma(vox_fwhm)
    # Do the smoothing
    sm_data = spnd.gaussian_filter(img.dataobj, vox_sd, mode=mode, cval=cval)
    return out_class(sm_data, img.affine, img.header)


def conform(
    from_img,
    out_shape=(256, 256, 256),
    voxel_size=(1.0, 1.0, 1.0),
    order=3,
    cval=0.0,
    orientation='RAS',
    out_class=None,
):
    """Resample image to ``out_shape`` with voxels of size ``voxel_size``.

    Using the default arguments, this function is meant to replicate most parts
    of FreeSurfer's ``mri_convert --conform`` command. Specifically, this
    function:

        - Resamples data to ``output_shape``
        - Resamples voxel sizes to ``voxel_size``
        - Reorients to RAS (``mri_convert --conform`` reorients to LIA)

    Unlike ``mri_convert --conform``, this command does not:

        - Transform data to range [0, 255]
        - Cast to unsigned eight-bit integer

    Parameters
    ----------
    from_img : object
        Object having attributes ``dataobj``, ``affine``, ``header`` and
        ``shape``. If `out_class` is not None, ``img.__class__`` should be able
        to construct an image from data, affine and header.
    out_shape : sequence, optional
        The shape of the output volume. Default is (256, 256, 256).
    voxel_size : sequence, optional
        The size in millimeters of the voxels in the resampled output. Default
        is 1mm isotropic.
    order : int, optional
        The order of the spline interpolation, default is 3.  The order has to
        be in the range 0-5 (see ``scipy.ndimage.affine_transform``)
    cval : scalar, optional
        Value used for points outside the boundaries of the input if
        ``mode='constant'``. Default is 0.0 (see
        ``scipy.ndimage.affine_transform``)
    orientation : str, optional
        Orientation of output image. Default is "RAS".
    out_class : None or SpatialImage class, optional
        Class of output image.  If None, use ``from_img.__class__``.

    Returns
    -------
    out_img : object
        Image of instance specified by `out_class`, containing data output from
        resampling `from_img` into axes aligned to the output space of
        ``from_img.affine``
    """
    # Only support 3D images. This can be made more general in the future, once tests
    # are written.
    required_ndim = 3
    if from_img.ndim != required_ndim:
        raise ValueError('Only 3D images are supported.')
    elif len(out_shape) != required_ndim:
        raise ValueError(f'`out_shape` must have {required_ndim} values')
    elif len(voxel_size) != required_ndim:
        raise ValueError(f'`voxel_size` must have {required_ndim} values')

    start_ornt = io_orientation(from_img.affine)
    end_ornt = axcodes2ornt(orientation)
    transform = ornt_transform(start_ornt, end_ornt)

    # Reorient first to ensure shape matches expectations
    reoriented = from_img.as_reoriented(transform)

    out_aff = rescale_affine(reoriented.affine, reoriented.shape, voxel_size, out_shape)

    # Resample input image.
    out_img = resample_from_to(
        from_img=from_img,
        to_vox_map=(out_shape, out_aff),
        order=order,
        mode='constant',
        cval=cval,
        out_class=out_class,
    )

    return out_img
