"""
Test the mask-extracting utilities.
"""
import distutils.version
import warnings
import numpy as np
import pytest
import sklearn

from sklearn.preprocessing import StandardScaler

from numpy.testing import assert_array_equal

from nibabel import Nifti1Image

from nilearn import masking
from nilearn.image import get_data, high_variance_confounds
from nilearn.masking import (compute_epi_mask, compute_multi_epi_mask,
                             compute_background_mask, compute_brain_mask,
                             compute_multi_gray_matter_mask,
                             unmask, _unmask_3d, _unmask_4d, intersect_masks,
                             MaskWarning, _extrapolate_out_mask, _unmask_from_to_3d_array)
from nilearn._utils.testing import write_tmp_imgs
from nilearn._utils.exceptions import DimensionError
from nilearn.input_data import NiftiMasker

np_version = (np.version.full_version if hasattr(np.version, 'full_version')
              else np.version.short_version)
np_version = distutils.version.LooseVersion(np_version).version

_TEST_DIM_ERROR_MSG = ("Input data has incompatible dimensionality: "
                       "Expected dimension is 3D and you provided "
                       "a %s image")

def _simu_img():
    # Random confounds
    conf = 2 + np.random.randn(100, 6)
    # Random 4D volume
    vol = 100 + 10 * np.random.randn(5, 5, 2, 100)
    img = Nifti1Image(vol, np.eye(4))
    # Create an nifti image with the data, and corresponding mask
    mask = Nifti1Image(np.ones([5, 5, 2]), np.eye(4))
    return img, mask, conf

def _cov_conf(tseries, conf):
    conf_n = StandardScaler().fit_transform(conf)
    tseries_n = StandardScaler().fit_transform(tseries)
    cov_mat = np.dot(tseries.T, conf_n)
    return cov_mat

def _confounds_regression(standardize_signal=True, standardize_confounds=True):
    rng = np.random.RandomState(42)
    img, mask, conf = _simu_img()
    masker = NiftiMasker(standardize=standardize_signal,
                         standardize_confounds=standardize_confounds,
                         detrend=False,
                         mask_img=mask).fit()
    tseries = masker.transform(img, confounds=conf)
    if standardize_confounds:
        conf = StandardScaler(with_std=False).fit_transform(conf)
    cov_mat = _cov_conf(tseries, conf)
    return np.sum(np.abs(cov_mat))

def test_high_variance_confounds():
    rng = np.random.RandomState(42)
    img, mask, conf = _simu_img()
    hv_confounds = high_variance_confounds(img)
    masker1 = NiftiMasker(standardize=True, detrend=False,
                          high_variance_confounds=False,
                          mask_img=mask).fit()
    tseries1 = masker1.transform(img, confounds=[hv_confounds, conf])
    masker2 = NiftiMasker(standardize=True, detrend=False,
                          high_variance_confounds=True,
                          mask_img=mask).fit()
    tseries2 = masker2.transform(img, confounds=conf)
    np.testing.assert_array_equal(tseries1, tseries2)


def test_confounds_standardization():
    # Tests for confounds standardization
    # See Issue #2584
    # Code from @pbellec
    #
    eps = 10e-10

    # Signal is not standardized
    # Explicit standardization of confounds
    assert(_confounds_regression(standardize_signal=False,
                                 standardize_confounds=True) < 10. * eps)

    # Signal is z-scored with string arg
    # Explicit standardization of confounds
    assert(_confounds_regression(standardize_signal='zscore',
                                 standardize_confounds=True) < eps)

    # Signal is z-scored with boolean arg
    # Explicit standardization of confounds
    assert(_confounds_regression(standardize_signal=True,
                                 standardize_confounds=True) < eps)

    # Signal is psc standardized
    # Explicit standardization of confounds
    assert(_confounds_regression(standardize_signal='psc',
                                 standardize_confounds=True) < 10. * eps)

    # Signal is not standardized
    # Confounds are not standardized
    # In this case, the regression should fail...
    assert(_confounds_regression(standardize_signal=False,
                                 standardize_confounds=False) > 100)

    # Signal is z-scored with string arg
    # Confounds are not standardized
    # In this case, the regression should fail...
    assert(_confounds_regression(standardize_signal='zscore',
                                 standardize_confounds=False) > 100)

    # Signal is psc standardized
    # Confounds are not standardized
    # In this case, the regression should fail...
    assert(_confounds_regression(standardize_signal='psc',
                                 standardize_confounds=False) > 100)


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
    np.testing.assert_array_equal(get_data(mask1), get_data(mask2))
    # Check that padding with zeros does not change the extracted mask
    mean_image2 = np.zeros((30, 30, 3))
    mean_image2[3:12, 3:12, :] = get_data(mean_image)
    mean_image2 = Nifti1Image(mean_image2, np.eye(4))
    mask3 = compute_epi_mask(mean_image2, exclude_zeros=True,
                             opening=False)
    np.testing.assert_array_equal(get_data(mask1),
                                  get_data(mask3)[3:12, 3:12])
    # However, without exclude_zeros, it does
    mask3 = compute_epi_mask(mean_image2, opening=False)
    assert not np.allclose(get_data(mask1),
                             get_data(mask3)[3:12, 3:12])

    # Check that we get a ValueError for incorrect shape
    mean_image = np.ones((9, 9))
    mean_image[3:-3, 3:-3] = 10
    mean_image[5, 5] = 100
    mean_image = Nifti1Image(mean_image, np.eye(4))
    pytest.raises(ValueError, compute_epi_mask, mean_image)

    # Check that we get a useful warning for empty masks
    mean_image = np.zeros((9, 9, 9))
    mean_image[0, 0, 1] = -1
    mean_image[0, 0, 0] = 1.2
    mean_image[0, 0, 2] = 1.1
    mean_image = Nifti1Image(mean_image, np.eye(4))
    with pytest.warns(MaskWarning, match='Computed an empty mask'):
        compute_epi_mask(mean_image, exclude_zeros=True)


def test_compute_background_mask():
    for value in (0, np.nan):
        mean_image = value * np.ones((9, 9, 9))
        mean_image[3:-3, 3:-3, 3:-3] = 1
        mask = mean_image == 1
        mean_image = Nifti1Image(mean_image, np.eye(4))
        mask1 = compute_background_mask(mean_image, opening=False)
        np.testing.assert_array_equal(get_data(mask1),
                                      mask.astype(np.int8))

    # Check that we get a ValueError for incorrect shape
    mean_image = np.ones((9, 9))
    mean_image[3:-3, 3:-3] = 10
    mean_image[5, 5] = 100
    mean_image = Nifti1Image(mean_image, np.eye(4))
    pytest.raises(ValueError, compute_background_mask, mean_image)

    # Check that we get a useful warning for empty masks
    mean_image = np.zeros((9, 9, 9))
    mean_image = Nifti1Image(mean_image, np.eye(4))
    with warnings.catch_warnings(record=True) as w:
        compute_background_mask(mean_image)
    assert len(w) == 1
    assert isinstance(w[0].message, masking.MaskWarning)


def test_compute_brain_mask():
    image = Nifti1Image(np.ones((9, 9, 9)), np.eye(4))

    mask = compute_brain_mask(image, threshold=-1)
    mask1 = np.zeros((9, 9, 9))
    mask1[2:-2, 2:-2, 2:-2] = 1

    np.testing.assert_array_equal(mask1, get_data(mask))

    # Check that we get a useful warning for empty masks
    with pytest.warns(masking.MaskWarning):
        compute_brain_mask(image, threshold=1)

    # Check that masks obtained from same FOV are the same
    rng = np.random.RandomState(42)
    img1 = Nifti1Image(np.full((9, 9, 9), rng.uniform()), np.eye(4))
    img2 = Nifti1Image(np.full((9, 9, 9), rng.uniform()), np.eye(4))

    mask_img1 = compute_brain_mask(img1)
    mask_img2 = compute_brain_mask(img2)
    np.testing.assert_array_equal(get_data(mask_img1),
                                  get_data(mask_img2))


def test_deprecation_warning_compute_gray_matter_mask():
    img = Nifti1Image(np.ones((9, 9, 9)), np.eye(4))
    if distutils.version.LooseVersion(sklearn.__version__) < '0.22':
        with pytest.deprecated_call():
            masking.compute_gray_matter_mask(img)
    else:
        with pytest.warns(FutureWarning,
                          match="renamed to 'compute_brain_mask'"):
            masking.compute_gray_matter_mask(img)


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
    assert np.all(np.isfinite(series))

    # veriy that 4D masks are rejected
    mask_img_4d = Nifti1Image(np.ones((40, 40, 40, 2)), np.eye(4))
    with pytest.raises(DimensionError, match=_TEST_DIM_ERROR_MSG % "4D"):
        masking.apply_mask(data_img, mask_img_4d)

    # Check that 3D data is accepted
    data_3d = Nifti1Image(np.arange(27).reshape((3, 3, 3)), np.eye(4))
    mask_data_3d = np.zeros((3, 3, 3))
    mask_data_3d[1, 1, 0] = True
    mask_data_3d[0, 1, 0] = True
    mask_data_3d[0, 1, 1] = True
    data_3d = masking.apply_mask(data_3d, Nifti1Image(mask_data_3d, np.eye(4)))
    assert sorted(data_3d.tolist()) == [3., 4., 12.]

    # Check data shape and affine
    with pytest.raises(DimensionError, match=_TEST_DIM_ERROR_MSG % "2D"):
        masking.apply_mask(data_img,
                           Nifti1Image(mask[20, ...], affine))
    pytest.raises(ValueError, masking.apply_mask,
                  data_img, Nifti1Image(mask, affine / 2.))
    # Check that full masking raises error
    pytest.raises(ValueError, masking.apply_mask,
                  data_img, full_mask_img)
    # Check weird values in data
    mask[10, 10, 10] = 2
    pytest.raises(ValueError, masking.apply_mask,
                  data_img, Nifti1Image(mask, affine))
    mask[15, 15, 15] = 3
    pytest.raises(ValueError, masking.apply_mask,
                  Nifti1Image(data, affine), mask_img)


def test_unmask():
    # A delta in 3D
    shape = (10, 20, 30, 40)
    rng = np.random.RandomState(42)
    data4D = rng.uniform(size=shape)
    data3D = data4D[..., 0]
    mask = rng.randint(2, size=shape[:3])
    mask_img = Nifti1Image(mask, np.eye(4))
    mask = mask.astype(bool)

    masked4D = data4D[mask, :].T
    unmasked4D = data4D.copy()
    unmasked4D[np.logical_not(mask), :] = 0
    masked3D = data3D[mask]
    unmasked3D = data3D.copy()
    unmasked3D[np.logical_not(mask)] = 0

    # 4D Test, test value ordering at the same time.
    t = get_data(unmask(masked4D, mask_img, order="C"))
    assert t.ndim == 4
    assert t.flags["C_CONTIGUOUS"]
    assert not t.flags["F_CONTIGUOUS"]
    assert_array_equal(t, unmasked4D)
    t = unmask([masked4D], mask_img, order="F")
    t = [get_data(t_) for t_ in t]
    assert isinstance(t, list)
    assert t[0].ndim == 4
    assert not t[0].flags["C_CONTIGUOUS"]
    assert t[0].flags["F_CONTIGUOUS"]
    assert_array_equal(t[0], unmasked4D)

    # 3D Test - check both with Nifti1Image and file
    for create_files in (False, True):
        with write_tmp_imgs(mask_img, create_files=create_files) as filename:
            t = get_data(unmask(masked3D, filename, order="C"))
            assert t.ndim == 3
            assert t.flags["C_CONTIGUOUS"]
            assert not t.flags["F_CONTIGUOUS"]
            assert_array_equal(t, unmasked3D)
            t = unmask([masked3D], filename, order="F")
            t = [get_data(t_) for t_ in t]
            assert isinstance(t, list)
            assert t[0].ndim == 3
            assert not t[0].flags["C_CONTIGUOUS"]
            assert t[0].flags["F_CONTIGUOUS"]
            assert_array_equal(t[0], unmasked3D)

    # Error test: shape
    vec_1D = np.empty((500,), dtype=int)
    pytest.raises(TypeError, unmask, vec_1D, mask_img)
    pytest.raises(TypeError, unmask, [vec_1D], mask_img)

    vec_2D = np.empty((500, 500), dtype=np.float64)
    pytest.raises(TypeError, unmask, vec_2D, mask_img)
    pytest.raises(TypeError, unmask, [vec_2D], mask_img)

    # Error test: mask type
    with pytest.raises(TypeError, match='mask must be a boolean array'):
        _unmask_3d(vec_1D, mask.astype(int))
    with pytest.raises(TypeError, match='mask must be a boolean array'):
        _unmask_4d(vec_2D, mask.astype(np.float64))

    # Transposed vector
    transposed_vector = np.ones((np.sum(mask), 1), dtype=bool)
    with pytest.raises(TypeError, match='X must be of shape'):
        unmask(transposed_vector, mask_img)


def test_intersect_masks_filename():
    # Create dummy masks
    mask_a = np.zeros((4, 4, 1), dtype=bool)
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

    mask_b = np.zeros((4, 4, 1), dtype=bool)
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
        mask_ab = np.zeros((4, 4, 1), dtype=bool)
        mask_ab[2, 2] = 1
        mask_ab_ = intersect_masks(filenames, threshold=1.)
        assert_array_equal(mask_ab, get_data(mask_ab_))


def test_intersect_masks():
    """ Test the intersect_masks function
    """

    # Create dummy masks
    mask_a = np.zeros((4, 4, 1), dtype=bool)
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

    mask_b = np.zeros((4, 4, 1), dtype=bool)
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

    mask_c = np.zeros((4, 4, 1), dtype=bool)
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

    mask_ab = np.zeros((4, 4, 1), dtype=bool)
    mask_ab[2, 2] = 1
    mask_ab_ = intersect_masks([mask_a_img, mask_b_img], threshold=1.)
    assert_array_equal(mask_ab, get_data(mask_ab_))
    # Test intersect mask images with '>f8'. This function uses
    # largest_connected_component to check if intersect_masks passes with
    # connected=True (which is by default)
    mask_a_img_change_dtype = Nifti1Image(get_data(mask_a_img).astype('>f8'),
                                          affine=mask_a_img.affine)
    mask_b_img_change_dtype = Nifti1Image(get_data(mask_b_img).astype('>f8'),
                                          affine=mask_b_img.affine)
    mask_ab_change_type = intersect_masks([mask_a_img_change_dtype,
                                           mask_b_img_change_dtype],
                                          threshold=1.)
    assert_array_equal(mask_ab, get_data(mask_ab_change_type))

    mask_abc = mask_a + mask_b + mask_c
    mask_abc_ = intersect_masks([mask_a_img, mask_b_img, mask_c_img],
                                threshold=0., connected=False)
    assert_array_equal(mask_abc, get_data(mask_abc_))

    mask_abc[0, 0] = 0
    mask_abc_ = intersect_masks([mask_a_img, mask_b_img, mask_c_img],
                                threshold=0.)
    assert_array_equal(mask_abc, get_data(mask_abc_))

    mask_abc = mask_ab
    mask_abc_ = intersect_masks([mask_a_img, mask_b_img, mask_c_img],
                                threshold=1.)
    assert_array_equal(mask_abc, get_data(mask_abc_))

    mask_abc[1, 2] = 1
    mask_abc[3, 2] = 1
    mask_abc_ = intersect_masks([mask_a_img, mask_b_img, mask_c_img])
    assert_array_equal(mask_abc, get_data(mask_abc_))


def test_compute_multi_epi_mask():
    # Check that an empty list of images creates a meaningful error
    pytest.raises(TypeError, compute_multi_epi_mask, [])
    # As it calls intersect_masks, we only test resampling here.
    # Same masks as test_intersect_masks
    mask_a = np.zeros((4, 4, 1), dtype=bool)
    mask_a[2:4, 2:4] = 1
    mask_a_img = Nifti1Image(mask_a.astype(int), np.eye(4))

    mask_b = np.zeros((8, 8, 1), dtype=bool)
    mask_b[2:6, 2:6] = 1
    mask_b_img = Nifti1Image(mask_b.astype(int), np.eye(4) / 2.)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", MaskWarning)
        pytest.raises(ValueError, compute_multi_epi_mask,
                      [mask_a_img, mask_b_img])
    mask_ab = np.zeros((4, 4, 1), dtype=bool)
    mask_ab[2, 2] = 1
    mask_ab_ = compute_multi_epi_mask([mask_a_img, mask_b_img], threshold=1.,
                                      opening=0,
                                      target_affine=np.eye(4),
                                      target_shape=(4, 4, 1))
    assert_array_equal(mask_ab, get_data(mask_ab_))


def test_compute_multi_gray_matter_mask():
    pytest.raises(TypeError, compute_multi_gray_matter_mask, [])

    # Check error raised if images with different shapes are given as input
    imgs = [Nifti1Image(np.ones((9, 9, 9)), np.eye(4)),
            Nifti1Image(np.ones((9, 9, 8)), np.eye(4))]
    pytest.raises(ValueError, compute_multi_gray_matter_mask, imgs)

    # Check results are the same if affine is the same
    rng = np.random.RandomState(42)
    imgs1 = [Nifti1Image(rng.standard_normal(size=(9, 9, 9)), np.eye(4)),
             Nifti1Image(rng.standard_normal(size=(9, 9, 9)), np.eye(4))]
    mask1 = compute_multi_gray_matter_mask(imgs1)

    imgs2 = [Nifti1Image(rng.standard_normal(size=(9, 9, 9)), np.eye(4)),
             Nifti1Image(rng.standard_normal(size=(9, 9, 9)), np.eye(4))]
    mask2 = compute_multi_gray_matter_mask(imgs2)

    assert_array_equal(get_data(mask1), get_data(mask2))


def test_error_shape(random_state=42, shape=(3, 5, 7, 11)):
    # open-ended `if .. elif` in masking.unmask

    rng = np.random.RandomState(random_state)

    # setup
    X = rng.standard_normal()
    mask_img = np.zeros(shape, dtype=np.uint8)
    mask_img[rng.standard_normal(size=shape) > .4] = 1
    n_features = (mask_img > 0).sum()
    mask_img = Nifti1Image(mask_img, np.eye(4))
    n_samples = shape[0]

    X = rng.standard_normal(size=(n_samples, n_features, 2))
    # 3D X (unmask should raise a TypeError)
    pytest.raises(TypeError, unmask, X, mask_img)

    X = rng.standard_normal(size=(n_samples, n_features))
    # Raises an error because the mask is 4D
    pytest.raises(TypeError, unmask, X, mask_img)


def test_nifti_masker_empty_mask_warning():
    X = Nifti1Image(np.ones((2, 2, 2, 5)), np.eye(4))
    with pytest.raises(
            ValueError,
            match="The mask is invalid as it is empty: it masks all data"):
        NiftiMasker(mask_strategy="epi").fit_transform(X)


def test_unmask_list(random_state=42):
    rng = np.random.RandomState(random_state)
    shape = (3, 4, 5)
    affine = np.eye(4)
    mask_data = (rng.uniform(size=shape) < .5)
    mask_img = Nifti1Image(mask_data.astype(np.uint8), affine)
    a = unmask(mask_data[mask_data], mask_img)
    b = unmask(mask_data[mask_data].tolist(), mask_img)  # shouldn't crash
    assert_array_equal(get_data(a), get_data(b))


def test__extrapolate_out_mask():
    # Input data:
    initial_data = np.zeros((5, 5, 5))
    initial_data[1, 2, 2] = 1
    initial_data[2, 1, 2] = 2
    initial_data[2, 2, 1] = 3
    initial_data[3, 2, 2] = 4
    initial_data[2, 3, 2] = 5
    initial_data[2, 2, 3] = 6
    initial_mask = initial_data.copy() != 0

    # Expected result
    target_data = np.array([[[0., 0., 0., 0., 0.],
                             [0., 0., 0., 0., 0.],
                             [0., 0., 1., 0., 0.],
                             [0., 0., 0., 0., 0.],
                             [0., 0., 0., 0., 0.]],

                            [[0., 0., 0., 0., 0.],
                             [0., 0., 1.5, 0., 0.],
                             [0., 2., 1., 3.5, 0.],
                             [0., 0., 3., 0., 0.],
                             [0., 0., 0., 0., 0.]],

                            [[0., 0., 2., 0., 0.],
                             [0., 2.5, 2., 4., 0.],
                             [3., 3., 3.5, 6., 6.],
                             [0., 4., 5., 5.5, 0.],
                             [0., 0., 5., 0., 0.]],

                            [[0., 0., 0., 0., 0.],
                             [0., 0., 3., 0., 0.],
                             [0., 3.5, 4., 5., 0.],
                             [0., 0., 4.5, 0., 0.],
                             [0., 0., 0., 0., 0.]],

                            [[0., 0., 0., 0., 0.],
                             [0., 0., 0., 0., 0.],
                             [0., 0., 4., 0., 0.],
                             [0., 0., 0., 0., 0.],
                             [0., 0., 0., 0., 0.]]])
    target_mask = np.array([[[False, False, False, False, False],
                             [False, False, False, False, False],
                             [False, False, True, False, False],
                             [False, False, False, False, False],
                             [False, False, False, False, False]],

                            [[False, False, False, False, False],
                             [False, False, True, False, False],
                             [False, True, True, True, False],
                             [False, False, True, False, False],
                             [False, False, False, False, False]],

                            [[False, False, True, False, False],
                             [False, True, True, True, False],
                             [True, True, True, True, True],
                             [False, True, True, True, False],
                             [False, False, True, False, False]],

                            [[False, False, False, False, False],
                             [False, False, True, False, False],
                             [False, True, True, True, False],
                             [False, False, True, False, False],
                             [False, False, False, False, False]],

                            [[False, False, False, False, False],
                             [False, False, False, False, False],
                             [False, False, True, False, False],
                             [False, False, False, False, False],
                             [False, False, False, False, False]]])

    # Test:
    extrapolated_data, extrapolated_mask = _extrapolate_out_mask(initial_data,
                                                                 initial_mask,
                                                                 iterations=1)
    assert_array_equal(extrapolated_data, target_data)
    assert_array_equal(extrapolated_mask, target_mask)


def test_unmask_from_to_3d_array(size=5):
    rng = np.random.RandomState(42)
    for ndim in range(1, 4):
        shape = [size] * ndim
        mask = np.zeros(shape).astype(bool)
        mask[rng.uniform(size=shape) > .8] = 1
        support = rng.standard_normal(size=mask.sum())
        full = _unmask_from_to_3d_array(support, mask)
        np.testing.assert_array_equal(full.shape, shape)
        np.testing.assert_array_equal(full[mask], support)
