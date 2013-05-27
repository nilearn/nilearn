"""
Test the mask-extracting utilities.
"""
import types
import numpy as np

from numpy.testing import assert_array_equal
from nose.tools import assert_true, assert_false, assert_equal, assert_raises

from nibabel import Nifti1Image

from .. import masking
from ..masking import compute_epi_mask, compute_multi_epi_mask, \
    unmask, intersect_masks

from ..testing import write_tmp_imgs


def test_mask():
    mean_image = np.ones((9, 9))
    mean_image[3:-3, 3:-3] = 10
    mean_image[5, 5] = 100
    mean_image = Nifti1Image(mean_image, np.eye(4))
    mask1 = compute_epi_mask(mean_image, opening=False)
    mask2 = compute_epi_mask(mean_image, exclude_zeros=True,
                             opening=False)
    # With an array with no zeros, exclude_zeros should not make
    # any difference
    yield np.testing.assert_array_equal, mask1.get_data(), mask2.get_data()
    # Check that padding with zeros does not change the extracted mask
    mean_image2 = np.zeros((30, 30))
    mean_image2[:9, :9] = mean_image.get_data()
    mean_image2 = Nifti1Image(mean_image2, np.eye(4))
    mask3 = compute_epi_mask(mean_image2, exclude_zeros=True,
                             opening=False)
    yield np.testing.assert_array_equal, \
        mask1.get_data(), mask3.get_data()[:9, :9]
    # However, without exclude_zeros, it does
    mask3 = compute_epi_mask(mean_image2, opening=False)
    yield assert_false, np.allclose(mask1.get_data(), mask3.get_data()[:9, :9])


def test__smooth_array():
    """Test smoothing of images: _smooth_array()"""
    # Impulse in 3D
    data = np.zeros((40, 41, 42))
    data[20, 20, 20] = 1

    # fwhm divided by any test affine must be odd. Otherwise assertion below
    # will fail. ( 9 / 0.6 = 15 is fine)
    fwhm = 9
    test_affines = (np.eye(4), np.diag((1, 1, -1, 1)),
                    np.diag((.6, 1, .6, 1)))
    for affine in test_affines:
        filtered = masking._smooth_array(data, affine,
                                         fwhm=fwhm, copy=True)
        assert_false(np.may_share_memory(filtered, data))

        # We are expecting a full-width at half maximum of
        # fwhm / voxel_size:
        vmax = filtered.max()
        above_half_max = filtered > .5 * vmax
        for axis in (0, 1, 2):
            proj = np.any(np.any(np.rollaxis(above_half_max,
                          axis=axis), axis=-1), axis=-1)
            np.testing.assert_equal(proj.sum(),
                                    fwhm / np.abs(affine[axis, axis]))

    # Check that NaNs in the data do not propagate
    data[10, 10, 10] = np.NaN
    filtered = masking._smooth_array(data, affine, fwhm=fwhm,
                                   ensure_finite=True, copy=True)
    assert_true(np.all(np.isfinite(filtered)))

    # Check copy=False.
    for affine in test_affines:
        data = np.zeros((40, 41, 42))
        data[20, 20, 20] = 1
        masking._smooth_array(data, affine, fwhm=fwhm, copy=False)

        # We are expecting a full-width at half maximum of
        # fwhm / voxel_size:
        vmax = data.max()
        above_half_max = data > .5 * vmax
        for axis in (0, 1, 2):
            proj = np.any(np.any(np.rollaxis(above_half_max,
                          axis=axis), axis=-1), axis=-1)
            np.testing.assert_equal(proj.sum(),
                                    fwhm / np.abs(affine[axis, axis]))


def test_apply_mask():
    """ Test smoothing of timeseries extraction
    """
    # A delta in 3D
    # Standard masking
    data = np.zeros((40, 40, 40, 2))
    data[20, 20, 20] = 1
    mask = np.ones((40, 40, 40))
    for create_files in (False, True):
        for affine in (np.eye(4), np.diag((1, 1, -1, 1)),
                       np.diag((.5, 1, .5, 1))):
            data_img = Nifti1Image(data, affine)
            mask_img = Nifti1Image(mask, affine)
            with write_tmp_imgs(data_img, mask_img, create_files=create_files)\
                     as filenames:
                series = masking.apply_mask(filenames[0], filenames[1],
                                            smoothing_fwhm=9)

            series = np.reshape(series[0, :], (40, 40, 40))
            vmax = series.max()
            # We are expecting a full-width at half maximum of
            # 9mm/voxel_size:
            above_half_max = series > .5 * vmax
            for axis in (0, 1, 2):
                proj = np.any(np.any(np.rollaxis(above_half_max,
                              axis=axis), axis=-1), axis=-1)
                np.testing.assert_equal(proj.sum(),
                                        9 / np.abs(affine[axis, axis]))

    # Check that NaNs in the data do not propagate
    data[10, 10, 10] = np.NaN
    data_img = Nifti1Image(data, affine)
    mask_img = Nifti1Image(mask, affine)
    series = masking.apply_mask(data_img, mask_img, smoothing_fwhm=9)
    assert_true(np.all(np.isfinite(series)))

    # Check data shape and affine
    assert_raises(ValueError, masking.apply_mask,
                  Nifti1Image(data, affine),
                  Nifti1Image(mask[20, ...], affine))
    assert_raises(ValueError, masking.apply_mask,
                  Nifti1Image(data, affine),
                  Nifti1Image(mask, affine / 2.))


def test_unmask():
    # A delta in 3D
    shape = (10, 20, 30, 40)
    generator = np.random.RandomState(42)
    data4D = generator.rand(*shape)
    data3D = data4D[..., 0]
    mask = generator.randint(2, size=shape[:3])
    mask_img = Nifti1Image(mask, np.eye(4))
    mask = mask.astype(bool)

    masked4D = data4D[mask, :].T
    unmasked4D = data4D.copy()
    unmasked4D[-mask, :] = 0
    masked3D = data3D[mask]
    unmasked3D = data3D.copy()
    unmasked3D[-mask] = 0

    # 4D Test
    t = unmask(masked4D, mask_img).get_data()
    assert_equal(t.ndim, 4)
    assert_array_equal(t, unmasked4D)
    t = unmask([masked4D], mask_img)
    t = [t_.get_data() for t_ in t]
    assert_true(isinstance(t, types.ListType))
    assert_equal(t[0].ndim, 4)
    assert_array_equal(t[0], unmasked4D)

    # 3D Test - check both with Nifti1Image and file
    for create_files in (False, True):
        with write_tmp_imgs(mask_img, create_files=create_files) as filename:
            t = unmask(masked3D, filename).get_data()
            assert_equal(t.ndim, 3)
            assert_array_equal(t, unmasked3D)
            t = unmask([masked3D], filename)
            t = [t_.get_data() for t_ in t]
            assert_true(isinstance(t, types.ListType))
            assert_equal(t[0].ndim, 3)
            assert_array_equal(t[0], unmasked3D)

    # 5D test
    shape5D = (10, 20, 30, 40, 41)
    data5D = generator.rand(*shape5D)
    mask = generator.randint(2, size=shape5D[:-1])
    mask_img = Nifti1Image(mask, np.eye(4))
    mask = mask.astype(bool)

    masked5D = data5D[mask, :].T
    unmasked5D = data5D.copy()
    unmasked5D[-mask, :] = 0

    t = unmask(masked5D, mask_img).get_data()
    assert_equal(t.ndim, len(shape5D))
    assert_array_equal(t, unmasked5D)
    t = unmask([masked5D], mask_img)
    t = [t_.get_data() for t_ in t]
    assert_true(isinstance(t, types.ListType))
    assert_equal(t[0].ndim, len(shape5D))
    assert_array_equal(t[0], unmasked5D)

    # Error test
    dummy = generator.rand(500)
    if np.__version__ > '1.8':
        assert_raises(IndexError, unmask, dummy, mask_img)
        assert_raises(IndexError, unmask, [dummy], mask_img)
    else:
        assert_raises(ValueError, unmask, dummy, mask_img)
        assert_raises(ValueError, unmask, [dummy], mask_img)


def test_intersect_masks():
    """ Test the intersect_masks function
    """

    # Create dummy masks
    mask_a = np.zeros((4, 4), dtype=np.bool)
    mask_a[2:4, 2:4] = 1
    mask_a_img = Nifti1Image(mask_a.astype(int), np.eye(4))

    # +---+---+---+---+
    # |   |   |   |   |
    # +---+---+---+---+
    # |   |   |   |   |
    # +---+---+---+---+
    # |   |   | X | X |
    # +---+---+---+---+
    # |   |   | X | X |
    # +---+---+---+---+

    mask_b = np.zeros((4, 4), dtype=np.bool)
    mask_b[1:3, 1:3] = 1
    mask_b_img = Nifti1Image(mask_b.astype(int), np.eye(4))

    # +---+---+---+---+
    # |   |   |   |   |
    # +---+---+---+---+
    # |   | X | X |   |
    # +---+---+---+---+
    # |   | X | X |   |
    # +---+---+---+---+
    # |   |   |   |   |
    # +---+---+---+---+

    mask_c = np.zeros((4, 4), dtype=np.bool)
    mask_c[:, 2] = 1
    mask_c[0, 0] = 1
    mask_c_img = Nifti1Image(mask_c.astype(int), np.eye(4))

    # +---+---+---+---+
    # | X |   | X |   |
    # +---+---+---+---+
    # |   |   | X |   |
    # +---+---+---+---+
    # |   |   | X |   |
    # +---+---+---+---+
    # |   |   | X |   |
    # +---+---+---+---+

    mask_ab = np.zeros((4, 4), dtype=np.bool)
    mask_ab[2, 2] = 1
    mask_ab_ = intersect_masks([mask_a_img, mask_b_img], threshold=1.)
    assert_array_equal(mask_ab, mask_ab_.get_data())

    mask_abc = mask_a + mask_b + mask_c
    mask_abc_ = intersect_masks([mask_a_img, mask_b_img, mask_c_img],
                                threshold=0., connected=False)
    assert_array_equal(mask_abc, mask_abc_.get_data())

    mask_abc[0, 0] = 0
    mask_abc_ = intersect_masks([mask_a_img, mask_b_img, mask_c_img],
                                threshold=0.)
    assert_array_equal(mask_abc, mask_abc_.get_data())

    mask_abc = mask_ab
    mask_abc_ = intersect_masks([mask_a_img, mask_b_img, mask_c_img],
                                threshold=1.)
    assert_array_equal(mask_abc, mask_abc_.get_data())

    mask_abc[1, 2] = 1
    mask_abc[3, 2] = 1
    mask_abc_ = intersect_masks([mask_a_img, mask_b_img, mask_c_img])
    assert_array_equal(mask_abc, mask_abc_.get_data())


def test_compute_multi_epi_mask():
    # As it calls intersect_masks, we only test resampling here.
    # Same masks as test_intersect_masks
    mask_a = np.zeros((4, 4, 1), dtype=np.bool)
    mask_a[2:4, 2:4] = 1
    mask_a_img = Nifti1Image(mask_a.astype(int), np.eye(4))

    mask_b = np.zeros((8, 8, 1), dtype=np.bool)
    mask_b[2:6, 2:6] = 1
    mask_b_img = Nifti1Image(mask_b.astype(int), np.eye(4) / 2.)

    assert_raises(ValueError, compute_multi_epi_mask, [mask_a_img, mask_b_img])
    mask_ab = np.zeros((4, 4, 1), dtype=np.bool)
    mask_ab[2, 2] = 1
    mask_ab_ = compute_multi_epi_mask([mask_a_img, mask_b_img], threshold=1.,
                                      opening=0,
                                      target_affine=np.eye(4),
                                      target_shape=(4, 4, 1))
    assert_array_equal(mask_ab, mask_ab_.get_data())
