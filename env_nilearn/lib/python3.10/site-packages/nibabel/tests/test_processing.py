# emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the NiBabel package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Testing processing module"""

import logging
from os.path import dirname
from os.path import join as pjoin

import numpy as np
import numpy.linalg as npl

from nibabel.optpkg import optional_package

spnd, have_scipy, _ = optional_package('scipy.ndimage')

import unittest

import pytest
from numpy.testing import assert_almost_equal, assert_array_equal

import nibabel as nib
from nibabel.affines import AffineError, apply_affine, from_matvec, to_matvec, voxel_sizes
from nibabel.eulerangles import euler2mat
from nibabel.nifti1 import Nifti1Image
from nibabel.nifti2 import Nifti2Image
from nibabel.orientations import aff2axcodes, inv_ornt_aff
from nibabel.processing import (
    adapt_affine,
    conform,
    fwhm2sigma,
    resample_from_to,
    resample_to_output,
    sigma2fwhm,
    smooth_image,
)
from nibabel.testing import assert_allclose_safely
from nibabel.tests.test_spaces import assert_all_in, get_outspace_params

needs_scipy = unittest.skipUnless(have_scipy, 'These tests need scipy')

DATA_DIR = pjoin(dirname(__file__), 'data')

# 3D MINC work correctly with processing, but not 4D MINC
from .test_imageclasses import MINC_3DS, MINC_4DS

# Filenames of other images that should work correctly with processing
OTHER_IMGS = (
    'anatomical.nii',
    'functional.nii',
    'example4d.nii.gz',
    'example_nifti2.nii.gz',
    'phantom_EPI_asc_CLEAR_2_1.PAR',
)


def test_sigma2fwhm():
    # Test from constant
    assert_almost_equal(sigma2fwhm(1), 2.3548200)
    assert_almost_equal(sigma2fwhm([1, 2, 3]), np.arange(1, 4) * 2.3548200)
    assert_almost_equal(fwhm2sigma(2.3548200), 1)
    assert_almost_equal(fwhm2sigma(np.arange(1, 4) * 2.3548200), [1, 2, 3])
    # direct test fwhm2sigma and sigma2fwhm are inverses of each other
    fwhm = np.arange(1.0, 5.0, 0.1)
    sigma = np.arange(1.0, 5.0, 0.1)
    assert np.allclose(sigma2fwhm(fwhm2sigma(fwhm)), fwhm)
    assert np.allclose(fwhm2sigma(sigma2fwhm(sigma)), sigma)


def test_adapt_affine():
    # Adapt affine to missing or extra input dimensions
    aff_3d = from_matvec(np.arange(9).reshape((3, 3)), [11, 12, 13])
    # For 4x4 affine, 3D image, no-op
    assert_array_equal(adapt_affine(aff_3d, 3), aff_3d)
    # For 4x4 affine, 4D image, add extra identity dimension
    assert_array_equal(
        adapt_affine(aff_3d, 4),
        [
            [0, 1, 2, 0, 11],
            [3, 4, 5, 0, 12],
            [6, 7, 8, 0, 13],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1],
        ],
    )
    # For 5x5 affine, 4D image, identity
    aff_4d = from_matvec(np.arange(16).reshape((4, 4)), [11, 12, 13, 14])
    assert_array_equal(adapt_affine(aff_4d, 4), aff_4d)
    # For 4x4 affine, 2D image, dropped column
    assert_array_equal(
        adapt_affine(aff_3d, 2),
        [
            [0, 1, 11],
            [3, 4, 12],
            [6, 7, 13],
            [0, 0, 1],
        ],
    )
    # For 4x4 affine, 1D image, 2 dropped columns
    assert_array_equal(
        adapt_affine(aff_3d, 1),
        [
            [0, 11],
            [3, 12],
            [6, 13],
            [0, 1],
        ],
    )
    # For 3x3 affine, 2D image, identity
    aff_2d = from_matvec(np.arange(4).reshape((2, 2)), [11, 12])
    assert_array_equal(adapt_affine(aff_2d, 2), aff_2d)


@needs_scipy
def test_resample_from_to(caplog):
    # Test resampling from image to image / image space
    data = np.arange(24, dtype='int32').reshape((2, 3, 4))
    affine = np.diag([-4, 5, 6, 1])
    img = Nifti1Image(data, affine)
    img.header['descrip'] = 'red shirt image'
    out = resample_from_to(img, img)
    assert_almost_equal(img.dataobj, out.dataobj)
    assert_array_equal(img.affine, out.affine)
    # Check resampling reverses effect of flipping axes
    # This will also test translations
    flip_ornt = np.array([[0, 1], [1, 1], [2, 1]])
    for axis in (0, 1, 2):
        ax_flip_ornt = flip_ornt.copy()
        ax_flip_ornt[axis, 1] = -1
        aff_flip_i = inv_ornt_aff(ax_flip_ornt, (2, 3, 4))
        flipped_img = Nifti1Image(np.flip(data, axis), np.dot(affine, aff_flip_i))
        out = resample_from_to(flipped_img, ((2, 3, 4), affine))
        assert_almost_equal(img.dataobj, out.dataobj)
        assert_array_equal(img.affine, out.affine)
    # A translation of one voxel on each axis
    trans_aff = from_matvec(np.diag([-4, 5, 6]), [4, -5, -6])
    trans_img = Nifti1Image(data, trans_aff)
    out = resample_from_to(trans_img, img)
    exp_out = np.zeros_like(data)
    exp_out[:-1, :-1, :-1] = data[1:, 1:, 1:]
    assert_almost_equal(out.dataobj, exp_out)
    out = resample_from_to(img, trans_img)
    trans_exp_out = np.zeros_like(data)
    trans_exp_out[1:, 1:, 1:] = data[:-1, :-1, :-1]
    assert_almost_equal(out.dataobj, trans_exp_out)
    # Test mode with translation of first axis only
    # Default 'constant' mode first
    trans1_aff = from_matvec(np.diag([-4, 5, 6]), [4, 0, 0])
    trans1_img = Nifti1Image(data, trans1_aff)
    out = resample_from_to(img, trans1_img)
    exp_out = np.zeros_like(data)
    exp_out[1:, :, :] = data[:-1, :, :]
    assert_almost_equal(out.dataobj, exp_out)
    # Then 'nearest' mode
    out = resample_from_to(img, trans1_img, mode='nearest')
    exp_out[0, :, :] = exp_out[1, :, :]
    assert_almost_equal(out.dataobj, exp_out)
    # Test order
    trans_p_25_aff = from_matvec(np.diag([-4, 5, 6]), [1, 0, 0])
    trans_p_25_img = Nifti1Image(data, trans_p_25_aff)
    # Surprising to me, but all points outside are set to 0, even with NN
    out = resample_from_to(img, trans_p_25_img, order=0)
    exp_out = np.zeros_like(data)
    exp_out[1:, :, :] = data[1, :, :]
    assert_almost_equal(out.dataobj, exp_out)
    out = resample_from_to(img, trans_p_25_img)
    with pytest.warns(UserWarning):  # Suppress scipy warning
        exp_out = spnd.affine_transform(data, [1, 1, 1], [-0.25, 0, 0], order=3)
    assert_almost_equal(out.dataobj, exp_out)
    # Test cval
    out = resample_from_to(img, trans_img, cval=99)
    exp_out = np.zeros_like(data) + 99
    exp_out[1:, 1:, 1:] = data[:-1, :-1, :-1]
    assert_almost_equal(out.dataobj, exp_out)
    # Out class
    out = resample_from_to(img, trans_img)
    assert out.__class__ == Nifti1Image
    # By default, type of from_img makes no difference
    n1_img = Nifti2Image(data, affine)
    with caplog.at_level(logging.CRITICAL):  # Here and below, suppress logs when changing classes
        out = resample_from_to(n1_img, trans_img)
    assert out.__class__ == Nifti1Image
    # Passed as keyword arg
    with caplog.at_level(logging.CRITICAL):
        out = resample_from_to(img, trans_img, out_class=Nifti2Image)
    assert out.__class__ == Nifti2Image
    # If keyword arg is None, use type of from_img
    out = resample_from_to(n1_img, trans_img, out_class=None)
    assert out.__class__ == Nifti2Image
    # to_img type irrelevant in all cases
    n1_trans_img = Nifti2Image(data, trans_aff)
    out = resample_from_to(img, n1_trans_img, out_class=None)
    assert out.__class__ == Nifti1Image
    # From 2D to 3D, error, the fixed affine is not invertible
    img_2d = Nifti1Image(data[:, :, 0], affine)
    with pytest.raises(AffineError):
        resample_from_to(img_2d, img)
    # 3D to 2D, we don't need to invert the fixed matrix
    out = resample_from_to(img, img_2d)
    assert_array_equal(out.dataobj, data[:, :, 0])
    # Same for tuple as to_img input
    out = resample_from_to(img, (img_2d.shape, img_2d.affine))
    assert_array_equal(out.dataobj, data[:, :, 0])
    # 4D input and output also OK
    data_4d = np.arange(24 * 5, dtype='int32').reshape((2, 3, 4, 5))
    img_4d = Nifti1Image(data_4d, affine)
    out = resample_from_to(img_4d, img_4d)
    assert_almost_equal(data_4d, out.dataobj)
    assert_array_equal(img_4d.affine, out.affine)
    # Errors trying to match 3D to 4D
    with pytest.raises(ValueError):
        resample_from_to(img_4d, img)
    with pytest.raises(ValueError):
        resample_from_to(img, img_4d)


@needs_scipy
def test_resample_to_output(caplog):
    # Test routine to sample images to output space
    # Image aligned to output axes - no-op
    data = np.arange(24, dtype='int32').reshape((2, 3, 4))
    img = Nifti1Image(data, np.eye(4))
    # Check default resampling
    img2 = resample_to_output(img)
    assert_array_equal(img2.shape, (2, 3, 4))
    assert_array_equal(img2.affine, np.eye(4))
    assert_array_equal(img2.dataobj, data)
    # Check resampling with different voxel size specifications
    for vox_sizes in (None, 1, [1, 1, 1]):
        img2 = resample_to_output(img, vox_sizes)
        assert_array_equal(img2.shape, (2, 3, 4))
        assert_array_equal(img2.affine, np.eye(4))
        assert_array_equal(img2.dataobj, data)
    img2 = resample_to_output(img, vox_sizes)
    # Check 2D works
    img_2d = Nifti1Image(data[0], np.eye(4))
    for vox_sizes in (None, 1, (1, 1), (1, 1, 1)):
        img3 = resample_to_output(img_2d, vox_sizes)
        assert_array_equal(img3.shape, (3, 4, 1))
        assert_array_equal(img3.affine, np.eye(4))
        assert_array_equal(img3.dataobj, data[0][..., None])
    # Even 1D
    img_1d = Nifti1Image(data[0, 0], np.eye(4))
    img3 = resample_to_output(img_1d)
    assert_array_equal(img3.shape, (4, 1, 1))
    assert_array_equal(img3.affine, np.eye(4))
    assert_array_equal(img3.dataobj, data[0, 0][..., None, None])
    # But 4D does not
    img_4d = Nifti1Image(data.reshape(2, 3, 2, 2), np.eye(4))
    with pytest.raises(ValueError):
        resample_to_output(img_4d)
    # Run vox2vox_out tests, checking output shape, coordinate transform
    for in_shape, in_aff, vox, out_shape, out_aff in get_outspace_params():
        # Allow for expansion of image shape from < 3D
        in_n_dim = len(in_shape)
        if in_n_dim < 3:
            in_shape = in_shape + (1,) * (3 - in_n_dim)
            if not vox is None:
                vox = vox + (1,) * (3 - in_n_dim)
            assert len(out_shape) == in_n_dim
            out_shape = out_shape + (1,) * (3 - in_n_dim)
        img = Nifti1Image(np.ones(in_shape), in_aff)
        out_img = resample_to_output(img, vox)
        assert_all_in(in_shape, in_aff, out_img.shape, out_img.affine)
        assert out_img.shape == out_shape
        assert_almost_equal(out_img.affine, out_aff)
    # Check data is as expected with some transforms
    # Flip first axis
    out_img = resample_to_output(Nifti1Image(data, np.diag([-1, 1, 1, 1])))
    assert_array_equal(out_img.dataobj, np.flipud(data))
    # Subsample voxels
    out_img = resample_to_output(Nifti1Image(data, np.diag([4, 5, 6, 1])))
    with pytest.warns(UserWarning):  # Suppress scipy warning
        exp_out = spnd.affine_transform(data, [1 / 4, 1 / 5, 1 / 6], output_shape=(5, 11, 19))
    assert_array_equal(out_img.dataobj, exp_out)
    # Unsubsample with voxel sizes
    out_img = resample_to_output(Nifti1Image(data, np.diag([4, 5, 6, 1])), [4, 5, 6])
    assert_array_equal(out_img.dataobj, data)
    # A rotation to test nearest, order, cval
    rot_3 = from_matvec(euler2mat(np.pi / 4), [0, 0, 0])
    rot_3_img = Nifti1Image(data, rot_3)
    out_img = resample_to_output(rot_3_img)
    exp_shape = (4, 4, 4)
    assert out_img.shape == exp_shape
    exp_aff = np.array(
        [
            [1, 0, 0, -2 * np.cos(np.pi / 4)],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]
    )
    assert_almost_equal(out_img.affine, exp_aff)
    rzs, trans = to_matvec(np.dot(npl.inv(rot_3), exp_aff))
    exp_out = spnd.affine_transform(data, rzs, trans, exp_shape)
    assert_almost_equal(out_img.dataobj, exp_out)
    # Order
    assert_almost_equal(
        resample_to_output(rot_3_img, order=0).dataobj,
        spnd.affine_transform(data, rzs, trans, exp_shape, order=0),
    )
    # Cval
    assert_almost_equal(
        resample_to_output(rot_3_img, cval=99).dataobj,
        spnd.affine_transform(data, rzs, trans, exp_shape, cval=99),
    )
    # Mode
    assert_almost_equal(
        resample_to_output(rot_3_img, mode='nearest').dataobj,
        spnd.affine_transform(data, rzs, trans, exp_shape, mode='nearest'),
    )
    # out_class
    img_ni1 = Nifti2Image(data, np.eye(4))
    img_ni2 = Nifti2Image(data, np.eye(4))
    # Default is Nifti1Image
    with caplog.at_level(logging.CRITICAL):  # Here and below, suppress logs when changing classes
        assert resample_to_output(img_ni2).__class__ == Nifti1Image
    # Can be overridden
    with caplog.at_level(logging.CRITICAL):
        assert resample_to_output(img_ni1, out_class=Nifti2Image).__class__ == Nifti2Image
    # None specifies out_class from input
    assert resample_to_output(img_ni2, out_class=None).__class__ == Nifti2Image


@needs_scipy
def test_smooth_image(caplog):
    # Test image smoothing
    data = np.arange(24, dtype='int32').reshape((2, 3, 4))
    aff = np.diag([-4, 5, 6, 1])
    img = Nifti1Image(data, aff)
    # Zero smoothing is no-op
    out_img = smooth_image(img, 0)
    assert_array_equal(out_img.affine, img.affine)
    assert_array_equal(out_img.shape, img.shape)
    assert_array_equal(out_img.dataobj, data)
    # Isotropic smoothing
    sd = fwhm2sigma(np.true_divide(8, [4, 5, 6]))
    exp_out = spnd.gaussian_filter(data, sd, mode='nearest')
    assert_array_equal(smooth_image(img, 8).dataobj, exp_out)
    assert_array_equal(smooth_image(img, [8, 8, 8]).dataobj, exp_out)
    with pytest.raises(ValueError):
        smooth_image(img, [8, 8])
    # Not isotropic
    mixed_sd = fwhm2sigma(np.true_divide([8, 7, 6], [4, 5, 6]))
    exp_out = spnd.gaussian_filter(data, mixed_sd, mode='nearest')
    assert_array_equal(smooth_image(img, [8, 7, 6]).dataobj, exp_out)
    # In 2D
    img_2d = Nifti1Image(data[0], aff)
    exp_out = spnd.gaussian_filter(data[0], sd[:2], mode='nearest')
    assert_array_equal(smooth_image(img_2d, 8).dataobj, exp_out)
    assert_array_equal(smooth_image(img_2d, [8, 8]).dataobj, exp_out)
    with pytest.raises(ValueError):
        smooth_image(img_2d, [8, 8, 8])
    # Isotropic in 4D has zero for last dimension in scalar case
    data_4d = np.arange(24 * 5, dtype='int32').reshape((2, 3, 4, 5))
    img_4d = Nifti1Image(data_4d, aff)
    exp_out = spnd.gaussian_filter(data_4d, list(sd) + [0], mode='nearest')
    assert_array_equal(smooth_image(img_4d, 8).dataobj, exp_out)
    # But raises error for vector case
    with pytest.raises(ValueError):
        smooth_image(img_4d, [8, 8, 8])
    # mode, cval
    exp_out = spnd.gaussian_filter(data, sd, mode='constant')
    assert_array_equal(smooth_image(img, 8, mode='constant').dataobj, exp_out)
    exp_out = spnd.gaussian_filter(data, sd, mode='constant', cval=99)
    assert_array_equal(smooth_image(img, 8, mode='constant', cval=99).dataobj, exp_out)
    # out_class
    img_ni1 = Nifti1Image(data, np.eye(4))
    img_ni2 = Nifti2Image(data, np.eye(4))
    # Default is Nifti1Image
    with caplog.at_level(logging.CRITICAL):  # Here and below, suppress logs when changing classes
        assert smooth_image(img_ni2, 0).__class__ == Nifti1Image
    # Can be overridden
    with caplog.at_level(logging.CRITICAL):
        assert smooth_image(img_ni1, 0, out_class=Nifti2Image).__class__ == Nifti2Image
    # None specifies out_class from input
    assert smooth_image(img_ni2, 0, out_class=None).__class__ == Nifti2Image


@needs_scipy
def test_spatial_axes_check(caplog):
    for fname in MINC_3DS + OTHER_IMGS:
        img = nib.load(pjoin(DATA_DIR, fname))
        with caplog.at_level(logging.CRITICAL):  # Suppress logs when changing classes
            s_img = smooth_image(img, 0)
        assert_array_equal(img.dataobj, s_img.dataobj)
        with caplog.at_level(logging.CRITICAL):
            out = resample_from_to(img, img, mode='nearest')
        assert_almost_equal(img.dataobj, out.dataobj)
        if len(img.shape) > 3:
            continue
        # Resample to output does not raise an error
        out = resample_to_output(img, voxel_sizes(img.affine))
    for fname in MINC_4DS:
        img = nib.load(pjoin(DATA_DIR, fname))
        with pytest.raises(ValueError):
            smooth_image(img, 0)
        with pytest.raises(ValueError):
            resample_from_to(img, img, mode='nearest')
        with pytest.raises(ValueError):
            resample_to_output(img, voxel_sizes(img.affine))


def assert_spm_resampling_close(from_img, our_resampled, spm_resampled):
    """Assert our resampling is close to SPM's, allowing for edge effects"""
    # To allow for differences in the way SPM and scipy.ndimage handle off-edge
    # interpolation, mask out voxels off edge
    to_img_shape = spm_resampled.shape
    to_img_affine = spm_resampled.affine
    to_vox_coords = np.indices(to_img_shape).transpose((1, 2, 3, 0))
    # Coordinates of to_img mapped to from_img
    to_to_from = npl.inv(from_img.affine).dot(to_img_affine)
    resamp_coords = apply_affine(to_to_from, to_vox_coords)
    # Places where SPM may not return default value but scipy.ndimage will (SPM
    # does not return zeros <0.05 from image edges).
    # See: https://github.com/nipy/nibabel/pull/255#issuecomment-186774173
    outside_vol = np.any(
        (resamp_coords < 0) | (np.subtract(resamp_coords, from_img.shape) > -1), axis=-1
    )
    spm_res = np.where(outside_vol, np.nan, np.array(spm_resampled.dataobj))
    assert_allclose_safely(our_resampled.dataobj, spm_res)
    assert_almost_equal(our_resampled.affine, spm_resampled.affine, 5)


@needs_scipy
def test_against_spm_resample():
    # Test resampling against images resampled with SPM12
    # anatomical.nii has a diagonal -2, 2 2 affine;
    # functional.nii has a diagonal -4, 4 4 affine;
    # These are a bit boring, so first add some rotations and translations to
    # the anatomical image affine, and then resample to the first volume in the
    # functional, and compare to the same thing in SPM.
    # See ``make_moved_anat.py`` script in this directory for input to SPM.
    anat = nib.load(pjoin(DATA_DIR, 'anatomical.nii'))
    func = nib.load(pjoin(DATA_DIR, 'functional.nii'))
    some_rotations = euler2mat(0.1, 0.2, 0.3)
    extra_affine = from_matvec(some_rotations, [3, 4, 5])
    moved_anat = nib.Nifti1Image(anat.get_fdata(), extra_affine.dot(anat.affine), anat.header)
    one_func = nib.Nifti1Image(func.dataobj[..., 0], func.affine, func.header)
    moved2func = resample_from_to(moved_anat, one_func, order=1, cval=np.nan)
    spm_moved = nib.load(pjoin(DATA_DIR, 'resampled_anat_moved.nii'))
    assert_spm_resampling_close(moved_anat, moved2func, spm_moved)
    # Next we resample the rotated anatomical image to output space, and compare
    # to the same operation done with SPM (our own version of 'reorient.m' by
    # John Ashburner).
    moved2output = resample_to_output(moved_anat, 4, order=1, cval=np.nan)
    spm2output = nib.load(pjoin(DATA_DIR, 'reoriented_anat_moved.nii'))
    assert_spm_resampling_close(moved_anat, moved2output, spm2output)


@needs_scipy
def test_conform(caplog):
    anat = nib.load(pjoin(DATA_DIR, 'anatomical.nii'))

    # Test with default arguments.
    c = conform(anat)
    assert c.shape == (256, 256, 256)
    assert c.header.get_zooms() == (1, 1, 1)
    assert c.dataobj.dtype.type == anat.dataobj.dtype.type
    assert aff2axcodes(c.affine) == ('R', 'A', 'S')
    assert isinstance(c, Nifti1Image)

    # Test with non-default arguments.
    with caplog.at_level(logging.CRITICAL):  # Suppress logs when changing classes
        c = conform(
            anat,
            out_shape=(100, 100, 200),
            voxel_size=(2, 2, 1.5),
            orientation='LPI',
            out_class=Nifti2Image,
        )
    assert c.shape == (100, 100, 200)
    assert c.header.get_zooms() == (2, 2, 1.5)
    assert c.dataobj.dtype.type == anat.dataobj.dtype.type
    assert aff2axcodes(c.affine) == ('L', 'P', 'I')
    assert isinstance(c, Nifti2Image)

    # TODO: support nD images in `conform` in the future, but for now, test that we get
    # errors on non-3D images.
    func = nib.load(pjoin(DATA_DIR, 'functional.nii'))
    with pytest.raises(ValueError):
        conform(func)
    with pytest.raises(ValueError):
        conform(anat, out_shape=(100, 100))
    with pytest.raises(ValueError):
        conform(anat, voxel_size=(2, 2))
