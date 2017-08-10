"""
Test the mask-extracting utilities.
"""
import distutils.version
import warnings
import numpy as np

from numpy.testing import assert_array_equal
from nose.tools import assert_true, assert_false, assert_equal, \
    assert_raises

from nibabel import Nifti1Image

from nilearn import masking
from nilearn.masking import (compute_epi_mask, compute_multi_epi_mask,
                             compute_background_mask, unmask, _unmask_3d,
                             _unmask_4d, intersect_masks, MaskWarning)
from nilearn._utils.testing import (write_tmp_imgs, assert_raises_regex)
from nilearn._utils.exceptions import DimensionError
from nilearn.input_data import NiftiMasker

np_version = (np.version.full_version if hasattr(np.version, 'full_version')
              else np.version.short_version)
np_version = distutils.version.LooseVersion(np_version).version

_TEST_DIM_ERROR_MSG = ("Input data has incompatible dimensionality: "
                       "Expected dimension is 3D and you provided "
                       "a %s image")


def test_compute_epi_mask():
    mean_image = np.ones((9, 9, 3))
    mean_image[3:-2, 3:-2, :] = 10
    mean_image[5, 5, :] = 11
    mean_image = Nifti1Image(mean_image, np.eye(4))
    mask1 = compute_epi_mask(mean_image, opening=False)
    mask2 = compute_epi_mask(mean_image, exclude_zeros=True,
                             opening=False)
    # With an array with no zeros, exclude_zeros should not make
    # any difference
    np.testing.assert_array_equal(mask1.get_data(), mask2.get_data())
    # Check that padding with zeros does not change the extracted mask
    mean_image2 = np.zeros((30, 30, 3))
    mean_image2[3:12, 3:12, :] = mean_image.get_data()
    mean_image2 = Nifti1Image(mean_image2, np.eye(4))
    mask3 = compute_epi_mask(mean_image2, exclude_zeros=True,
                             opening=False)
    np.testing.assert_array_equal(mask1.get_data(),
                                  mask3.get_data()[3:12, 3:12])
    # However, without exclude_zeros, it does
    mask3 = compute_epi_mask(mean_image2, opening=False)
    assert_false(np.allclose(mask1.get_data(),
                             mask3.get_data()[3:12, 3:12]))

    # Check that we get a ValueError for incorrect shape
    mean_image = np.ones((9, 9))
    mean_image[3:-3, 3:-3] = 10
    mean_image[5, 5] = 100
    mean_image = Nifti1Image(mean_image, np.eye(4))
    assert_raises(ValueError, compute_epi_mask, mean_image)

    # Check that we get a useful warning for empty masks
    mean_image = np.zeros((9, 9, 9))
    mean_image[0, 0, 1] = -1
    mean_image[0, 0, 0] = 1.2
    mean_image[0, 0, 2] = 1.1
    mean_image = Nifti1Image(mean_image, np.eye(4))
    with warnings.catch_warnings(record=True) as w:
        compute_epi_mask(mean_image, exclude_zeros=True)
    assert_equal(len(w), 1)
    assert_true(isinstance(w[0].message, masking.MaskWarning))


def test_compute_background_mask():
    for value in (0, np.nan):
        mean_image = value * np.ones((9, 9, 9))
        mean_image[3:-3, 3:-3, 3:-3] = 1
        mask = mean_image == 1
        mean_image = Nifti1Image(mean_image, np.eye(4))
        mask1 = compute_background_mask(mean_image, opening=False)
        np.testing.assert_array_equal(mask1.get_data(),
                                      mask.astype(np.int8))

    # Check that we get a ValueError for incorrect shape
    mean_image = np.ones((9, 9))
    mean_image[3:-3, 3:-3] = 10
    mean_image[5, 5] = 100
    mean_image = Nifti1Image(mean_image, np.eye(4))
    assert_raises(ValueError, compute_background_mask, mean_image)

    # Check that we get a useful warning for empty masks
    mean_image = np.zeros((9, 9, 9))
    mean_image = Nifti1Image(mean_image, np.eye(4))
    with warnings.catch_warnings(record=True) as w:
        compute_background_mask(mean_image)
    assert_equal(len(w), 1)
    assert_true(isinstance(w[0].message, masking.MaskWarning))


def test_apply_mask():
    """ Test smoothing of timeseries extraction
    """
    # A delta in 3D
    # Standard masking
    data = np.zeros((40, 40, 40, 2))
    data[20, 20, 20] = 1
    mask = np.ones((40, 40, 40))
    full_mask = np.zeros((40, 40, 40))
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
    full_mask_img = Nifti1Image(full_mask, affine)
    series = masking.apply_mask(data_img, mask_img, smoothing_fwhm=9)
    assert_true(np.all(np.isfinite(series)))

    # veriy that 4D masks are rejected
    mask_img_4d = Nifti1Image(np.ones((40, 40, 40, 2)), np.eye(4))
    assert_raises_regex(DimensionError, _TEST_DIM_ERROR_MSG % "4D",
                        masking.apply_mask, data_img, mask_img_4d)

    # Check that 3D data is accepted
    data_3d = Nifti1Image(np.arange(27).reshape((3, 3, 3)), np.eye(4))
    mask_data_3d = np.zeros((3, 3, 3))
    mask_data_3d[1, 1, 0] = True
    mask_data_3d[0, 1, 0] = True
    mask_data_3d[0, 1, 1] = True
    data_3d = masking.apply_mask(data_3d, Nifti1Image(mask_data_3d, np.eye(4)))
    assert_equal(sorted(data_3d.tolist()), [3., 4., 12.])

    # Check data shape and affine
    assert_raises_regex(DimensionError, _TEST_DIM_ERROR_MSG % "2D",
                        masking.apply_mask, data_img,
                        Nifti1Image(mask[20, ...], affine))
    assert_raises(ValueError, masking.apply_mask,
                  data_img, Nifti1Image(mask, affine / 2.))
    # Check that full masking raises error
    assert_raises(ValueError, masking.apply_mask,
                  data_img, full_mask_img)
    # Check weird values in data
    mask[10, 10, 10] = 2
    assert_raises(ValueError, masking.apply_mask,
                  data_img, Nifti1Image(mask, affine))
    mask[15, 15, 15] = 3
    assert_raises(ValueError, masking.apply_mask,
                  Nifti1Image(data, affine), mask_img)


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
    unmasked4D[np.logical_not(mask), :] = 0
    masked3D = data3D[mask]
    unmasked3D = data3D.copy()
    unmasked3D[np.logical_not(mask)] = 0

    # 4D Test, test value ordering at the same time.
    t = unmask(masked4D, mask_img, order="C").get_data()
    assert_equal(t.ndim, 4)
    assert_true(t.flags["C_CONTIGUOUS"])
    assert_false(t.flags["F_CONTIGUOUS"])
    assert_array_equal(t, unmasked4D)
    t = unmask([masked4D], mask_img, order="F")
    t = [t_.get_data() for t_ in t]
    assert_true(isinstance(t, list))
    assert_equal(t[0].ndim, 4)
    assert_false(t[0].flags["C_CONTIGUOUS"])
    assert_true(t[0].flags["F_CONTIGUOUS"])
    assert_array_equal(t[0], unmasked4D)

    # 3D Test - check both with Nifti1Image and file
    for create_files in (False, True):
        with write_tmp_imgs(mask_img, create_files=create_files) as filename:
            t = unmask(masked3D, filename, order="C").get_data()
            assert_equal(t.ndim, 3)
            assert_true(t.flags["C_CONTIGUOUS"])
            assert_false(t.flags["F_CONTIGUOUS"])
            assert_array_equal(t, unmasked3D)
            t = unmask([masked3D], filename, order="F")
            t = [t_.get_data() for t_ in t]
            assert_true(isinstance(t, list))
            assert_equal(t[0].ndim, 3)
            assert_false(t[0].flags["C_CONTIGUOUS"])
            assert_true(t[0].flags["F_CONTIGUOUS"])
            assert_array_equal(t[0], unmasked3D)

    # Error test: shape
    vec_1D = np.empty((500,), dtype=np.int)
    assert_raises(TypeError, unmask, vec_1D, mask_img)
    assert_raises(TypeError, unmask, [vec_1D], mask_img)

    vec_2D = np.empty((500, 500), dtype=np.float64)
    assert_raises(TypeError, unmask, vec_2D, mask_img)
    assert_raises(TypeError, unmask, [vec_2D], mask_img)

    # Error test: mask type
    assert_raises_regex(TypeError, 'mask must be a boolean array',
                        _unmask_3d, vec_1D, mask.astype(np.int))
    assert_raises_regex(TypeError, 'mask must be a boolean array',
                        _unmask_4d, vec_2D, mask.astype(np.float64))

    # Transposed vector
    transposed_vector = np.ones((np.sum(mask), 1), dtype=np.bool)
    assert_raises_regex(TypeError, 'X must be of shape',
                        unmask, transposed_vector, mask_img)


def test_intersect_masks_filename():
    # Create dummy masks
    mask_a = np.zeros((4, 4, 1), dtype=np.bool)
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

    mask_b = np.zeros((4, 4, 1), dtype=np.bool)
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

    with write_tmp_imgs(mask_a_img, mask_b_img, create_files=True)\
            as filenames:
        mask_ab = np.zeros((4, 4, 1), dtype=np.bool)
        mask_ab[2, 2] = 1
        mask_ab_ = intersect_masks(filenames, threshold=1.)
        assert_array_equal(mask_ab, mask_ab_.get_data())


def test_intersect_masks():
    """ Test the intersect_masks function
    """

    # Create dummy masks
    mask_a = np.zeros((4, 4, 1), dtype=np.bool)
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

    mask_b = np.zeros((4, 4, 1), dtype=np.bool)
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

    mask_c = np.zeros((4, 4, 1), dtype=np.bool)
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

    mask_ab = np.zeros((4, 4, 1), dtype=np.bool)
    mask_ab[2, 2] = 1
    mask_ab_ = intersect_masks([mask_a_img, mask_b_img], threshold=1.)
    assert_array_equal(mask_ab, mask_ab_.get_data())
    # Test intersect mask images with '>f8'. This function uses
    # largest_connected_component to check if intersect_masks passes with
    # connected=True (which is by default)
    mask_a_img_change_dtype = Nifti1Image(mask_a_img.get_data().astype('>f8'),
                                          affine=mask_a_img.get_affine())
    mask_b_img_change_dtype = Nifti1Image(mask_b_img.get_data().astype('>f8'),
                                          affine=mask_b_img.get_affine())
    mask_ab_change_type = intersect_masks([mask_a_img_change_dtype,
                                           mask_b_img_change_dtype],
                                          threshold=1.)
    assert_array_equal(mask_ab, mask_ab_change_type.get_data())

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
    # Check that an empty list of images creates a meaningful error
    assert_raises(TypeError, compute_multi_epi_mask, [])
    # As it calls intersect_masks, we only test resampling here.
    # Same masks as test_intersect_masks
    mask_a = np.zeros((4, 4, 1), dtype=np.bool)
    mask_a[2:4, 2:4] = 1
    mask_a_img = Nifti1Image(mask_a.astype(int), np.eye(4))

    mask_b = np.zeros((8, 8, 1), dtype=np.bool)
    mask_b[2:6, 2:6] = 1
    mask_b_img = Nifti1Image(mask_b.astype(int), np.eye(4) / 2.)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", MaskWarning)
        assert_raises(ValueError, compute_multi_epi_mask,
                      [mask_a_img, mask_b_img])
    mask_ab = np.zeros((4, 4, 1), dtype=np.bool)
    mask_ab[2, 2] = 1
    mask_ab_ = compute_multi_epi_mask([mask_a_img, mask_b_img], threshold=1.,
                                      opening=0,
                                      target_affine=np.eye(4),
                                      target_shape=(4, 4, 1))
    assert_array_equal(mask_ab, mask_ab_.get_data())


def test_error_shape(random_state=42, shape=(3, 5, 7, 11)):
    # open-ended `if .. elif` in masking.unmask

    rng = np.random.RandomState(random_state)

    # setup
    X = rng.randn()
    mask_img = np.zeros(shape, dtype=np.uint8)
    mask_img[rng.randn(*shape) > .4] = 1
    n_features = (mask_img > 0).sum()
    mask_img = Nifti1Image(mask_img, np.eye(4))
    n_samples = shape[0]

    X = rng.randn(n_samples, n_features, 2)
    # 3D X (unmask should raise a TypeError)
    assert_raises(TypeError, unmask, X, mask_img)

    X = rng.randn(n_samples, n_features)
    # Raises an error because the mask is 4D
    assert_raises(TypeError, unmask, X, mask_img)


def test_nifti_masker_empty_mask_warning():
    X = Nifti1Image(np.ones((2, 2, 2, 5)), np.eye(4))
    assert_raises_regex(
        ValueError,
        "The mask is invalid as it is empty: it masks all data",
        NiftiMasker(mask_strategy="epi").fit_transform, X)


def test_unmask_list(random_state=42):
    rng = np.random.RandomState(random_state)
    shape = (3, 4, 5)
    affine = np.eye(4)
    mask_data = (rng.rand(*shape) < .5)
    mask_img = Nifti1Image(mask_data.astype(np.uint8), affine)
    a = unmask(mask_data[mask_data], mask_img)
    b = unmask(mask_data[mask_data].tolist(), mask_img)  # shouldn't crash
    assert_array_equal(a.get_data(), b.get_data())
