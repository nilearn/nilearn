"""Test the nifti_masker module.

Functions in this file only test features added by the NiftiMasker class,
not the underlying functions used (e.g. clean()). See test_masking.py and
test_signal.py for this.
"""

import warnings

import numpy as np
import pytest
from nibabel import Nifti1Image
from numpy.testing import assert_array_equal
from sklearn.utils.estimator_checks import parametrize_with_checks

from nilearn._utils import data_gen, exceptions
from nilearn._utils.class_inspect import get_params
from nilearn._utils.estimator_checks import (
    check_estimator,
    nilearn_check_estimator,
    return_expected_failed_checks,
)
from nilearn._utils.tags import SKLEARN_LT_1_6
from nilearn.image import get_data, index_img
from nilearn.maskers import NiftiMasker
from nilearn.maskers.nifti_masker import filter_and_mask

ESTIMATORS_TO_CHECK = [NiftiMasker()]

if SKLEARN_LT_1_6:

    @pytest.mark.parametrize(
        "estimator, check, name",
        check_estimator(estimators=ESTIMATORS_TO_CHECK),
    )
    def test_check_estimator_sklearn_valid(estimator, check, name):  # noqa: ARG001
        """Check compliance with sklearn estimators."""
        check(estimator)

    @pytest.mark.xfail(reason="invalid checks should fail")
    @pytest.mark.parametrize(
        "estimator, check, name",
        check_estimator(estimators=ESTIMATORS_TO_CHECK, valid=False),
    )
    def test_check_estimator_sklearn_invalid(estimator, check, name):  # noqa: ARG001
        """Check compliance with sklearn estimators."""
        check(estimator)

else:

    @parametrize_with_checks(
        estimators=ESTIMATORS_TO_CHECK,
        expected_failed_checks=return_expected_failed_checks,
    )
    def test_check_estimator_sklearn(estimator, check):
        """Check compliance with sklearn estimators."""
        check(estimator)


@pytest.mark.parametrize(
    "estimator, check, name",
    nilearn_check_estimator(estimators=ESTIMATORS_TO_CHECK),
)
def test_check_estimator_nilearn(estimator, check, name):  # noqa: ARG001
    """Check compliance with nilearn estimators rules."""
    check(estimator)


def test_detrend(img_3d_rand_eye, mask_img_1):
    """Check that detrending doesn't do something stupid with 3D images."""
    # Smoke test the fit
    masker = NiftiMasker(mask_img=mask_img_1, detrend=True)
    X = masker.fit_transform(img_3d_rand_eye)
    assert np.any(X != 0)


@pytest.mark.parametrize("y", [None, np.ones((9, 9, 9))])
def test_fit_transform(y, img_3d_rand_eye, mask_img_1):
    """Check fit_transform of BaseMasker with several input args."""
    # Smoke test the fit
    for mask_img in [mask_img_1, None]:
        masker = NiftiMasker(mask_img=mask_img)
        X = masker.fit_transform(X=img_3d_rand_eye, y=y)
        assert np.any(X != 0)


def test_fit_transform_warning(img_3d_rand_eye, mask_img_1):
    """Warn that mask creation is happening \
        when mask was provided at instantiation.
    """
    y = np.ones((9, 9, 9))
    masker = NiftiMasker(mask_img=mask_img_1)
    with pytest.warns(
        UserWarning,
        match="Generation of a mask has been requested .*"
        "while a mask was given at masker creation.",
    ):
        X = masker.fit_transform(X=img_3d_rand_eye, y=y)
        assert np.any(X != 0)


def test_resample(img_3d_rand_eye, mask_img_1):
    """Check that target_affine triggers the right resampling."""
    masker = NiftiMasker(mask_img=mask_img_1, target_affine=2 * np.eye(3))
    # Smoke test the fit
    X = masker.fit_transform(img_3d_rand_eye)
    assert np.any(X != 0)


def test_resample_to_mask_warning(img_3d_rand_eye, affine_eye):
    """Check that a warning is raised when data is \
       being resampled to mask's resolution.
    """
    # defining a mask with different fov than img
    mask = np.zeros((12, 12, 12))
    mask[3:-3, 3:-3, 3:-3] = 10
    mask = mask.astype("uint8")
    mask_img = Nifti1Image(mask, affine_eye)
    masker = NiftiMasker(mask_img=mask_img)
    with pytest.warns(
        UserWarning,
        match="imgs are being resampled to the mask_img resolution. "
        "This process is memory intensive. You might want to provide "
        "a target_affine that is equal to the affine of the imgs "
        "or resample the mask beforehand "
        "to save memory and computation time.",
    ):
        masker.fit_transform(img_3d_rand_eye)


def test_nan(affine_eye):
    """Check that the masker handles NaNs appropriately."""
    data = np.ones((9, 9, 9))
    data[0] = np.nan
    data[:, 0] = np.nan
    data[:, :, 0] = np.nan
    data[-1] = np.nan
    data[:, -1] = np.nan
    data[:, :, -1] = np.nan
    data[3:-3, 3:-3, 3:-3] = 10
    img = Nifti1Image(data, affine_eye)
    masker = NiftiMasker(mask_args={"opening": 0})
    masker.fit(img)
    mask = get_data(masker.mask_img_)
    assert mask[1:-1, 1:-1, 1:-1].all()
    assert not mask[0].any()
    assert not mask[:, 0].any()
    assert not mask[:, :, 0].any()
    assert not mask[-1].any()
    assert not mask[:, -1].any()
    assert not mask[:, :, -1].any()


def test_matrix_orientation():
    """Test if processing is performed along the correct axis."""
    # the "step" kind generate heavyside-like signals for each voxel.
    # all signals being identical, standardizing along the wrong axis
    # would leave a null signal. Along the correct axis, the step remains.
    fmri, mask = data_gen.generate_fake_fmri(shape=(40, 41, 42), kind="step")
    masker = NiftiMasker(mask_img=mask, standardize=True, detrend=True)
    timeseries = masker.fit_transform(fmri)
    assert timeseries.shape[0] == fmri.shape[3]
    assert timeseries.shape[1] == get_data(mask).sum()
    std = timeseries.std(axis=0)
    assert std.shape[0] == timeseries.shape[1]  # paranoid
    assert not np.any(std < 0.1)

    # Test inverse transform
    masker = NiftiMasker(mask_img=mask, standardize=False, detrend=False)
    masker.fit()
    timeseries = masker.transform(fmri)
    recovered = masker.inverse_transform(timeseries)
    np.testing.assert_array_almost_equal(get_data(recovered), get_data(fmri))


def test_mask_4d(shape_3d_default, affine_eye):
    """Test performance with 4D data."""
    # Dummy mask
    mask = np.zeros(shape_3d_default, dtype="int32")
    mask[3:7, 3:7, 3:7] = 1
    mask_bool = mask.astype(bool)
    mask_img = Nifti1Image(mask, affine_eye)

    # Dummy data
    shape_4d = (*shape_3d_default, 5)
    data = np.zeros(shape_4d, dtype="int32")
    data[..., 0] = 1
    data[..., 1] = 2
    data[..., 2] = 3
    data_img_4d = Nifti1Image(data, affine_eye)
    data_imgs = [
        index_img(data_img_4d, 0),
        index_img(data_img_4d, 1),
        index_img(data_img_4d, 2),
    ]

    # check whether transform is indeed selecting niimgs subset
    sample_mask = np.array([0, 2])
    masker = NiftiMasker(mask_img=mask_img)
    masker.fit()
    data_trans = masker.transform(data_imgs, sample_mask=sample_mask)
    data_trans_img = index_img(data_img_4d, sample_mask)
    data_trans_direct = get_data(data_trans_img)[mask_bool, :]
    data_trans_direct = np.swapaxes(data_trans_direct, 0, 1)

    assert_array_equal(data_trans, data_trans_direct)

    masker = NiftiMasker(mask_img=mask_img)
    masker.fit()
    data_trans2 = masker.transform(data_img_4d, sample_mask=sample_mask)

    assert_array_equal(data_trans2, data_trans_direct)

    diff_sample_mask = np.array([2, 4])
    data_trans_img_diff = index_img(data_img_4d, diff_sample_mask)
    data_trans_direct_diff = get_data(data_trans_img_diff)[mask_bool, :]
    data_trans_direct_diff = np.swapaxes(data_trans_direct_diff, 0, 1)
    masker = NiftiMasker(mask_img=mask_img)
    masker.fit()
    data_trans3 = masker.transform(data_img_4d, sample_mask=diff_sample_mask)

    assert_array_equal(data_trans3, data_trans_direct_diff)


def test_4d_single_scan(rng, shape_3d_default, affine_eye):
    """Test that list of 4D images with last dim=1 is treated as 3D."""
    shape_3d = (10, 10, 10)
    mask = np.zeros(shape_3d)
    mask[3:7, 3:7, 3:7] = 1
    mask_img = Nifti1Image(mask, affine_eye)

    shape_4d = (*shape_3d_default, 1)
    data_5d = [rng.random(shape_4d) for _ in range(5)]
    data_4d = [d[..., 0] for d in data_5d]
    data_5d = [Nifti1Image(d, affine_eye) for d in data_5d]
    data_4d = [Nifti1Image(d, affine_eye) for d in data_4d]

    masker = NiftiMasker(mask_img=mask_img)

    masker.fit()

    # Check attributes defined at fit
    assert masker.n_elements_ == np.sum(mask)

    data_trans_5d = masker.transform(data_5d)
    data_trans_4d = masker.transform(data_4d)

    assert_array_equal(data_trans_4d, data_trans_5d)


def test_sessions(affine_eye):
    """Test the sessions vector."""
    data = np.ones((40, 40, 40, 4))
    # Create a border, so that the masking work well
    data[0] = 0
    data[-1] = 0
    data[:, -1] = 0
    data[:, 0] = 0
    data[..., -1] = 0
    data[..., 0] = 0
    data[20, 20, 20] = 1
    data_img = Nifti1Image(data, affine_eye)
    masker = NiftiMasker(runs=np.ones(3, dtype=int))
    with pytest.raises(ValueError):
        masker.fit_transform(data_img)


def test_mask_strategy_errors_warnings(img_fmri):
    """Check that mask_strategy errors are raised."""
    # Error with unknown mask_strategy

    masker = NiftiMasker(mask_strategy="oops", mask_args={"threshold": 0.0})
    with pytest.raises(
        ValueError, match="Unknown value of mask_strategy 'oops'"
    ):
        masker.fit(img_fmri)


def test_compute_epi_mask(affine_eye):
    """Test that the masker class is passing parameters appropriately."""
    # Taken from test_masking.py, but used to test that the masker class
    #   is passing parameters appropriately.
    mean_image = np.ones((9, 9, 3))
    mean_image[3:-2, 3:-2, :] = 10
    mean_image[5, 5, :] = 11
    mean_image = Nifti1Image(mean_image.astype(float), affine_eye)

    masker = NiftiMasker(mask_strategy="epi", mask_args={"opening": False})
    masker.fit(mean_image)
    mask1 = masker.mask_img_

    masker2 = NiftiMasker(
        mask_strategy="epi",
        mask_args={"opening": False, "exclude_zeros": True},
    )
    masker2.fit(mean_image)
    mask2 = masker2.mask_img_

    # With an array with no zeros, exclude_zeros should not make
    # any difference
    np.testing.assert_array_equal(get_data(mask1), get_data(mask2))

    # Check that padding with zeros does not change the extracted mask
    mean_image2 = np.zeros((30, 30, 3))
    mean_image2[3:12, 3:12, :] = get_data(mean_image)
    mean_image2 = Nifti1Image(mean_image2, affine_eye)

    masker3 = NiftiMasker(
        mask_strategy="epi",
        mask_args={"opening": False, "exclude_zeros": True},
    )
    masker3.fit(mean_image2)
    mask3 = masker3.mask_img_
    np.testing.assert_array_equal(get_data(mask1), get_data(mask3)[3:12, 3:12])

    # However, without exclude_zeros, it does
    masker4 = NiftiMasker(mask_strategy="epi", mask_args={"opening": False})
    masker4.fit(mean_image2)
    mask4 = masker4.mask_img_

    assert not np.allclose(get_data(mask1), get_data(mask4)[3:12, 3:12])


@pytest.fixture
def expected_mask(mask_args):
    """Create an expected mask."""
    mask = np.zeros((9, 9, 5))
    if mask_args == {}:
        return mask

    mask[2:7, 2:7, 2] = 1
    return mask


@pytest.mark.parametrize(
    "strategy", [f"{p}-template" for p in ["whole-brain", "gm", "wm"]]
)
def test_compute_brain_mask_empty_mask_error(strategy):
    """Check masker raise error when estimated mask is empty."""
    masker = NiftiMasker(mask_strategy=strategy, mask_args={})

    img, _ = data_gen.generate_random_img((9, 9, 5))

    with pytest.raises(ValueError, match="masks all data"):
        masker.fit(img)


@pytest.mark.timeout(0)
@pytest.mark.parametrize(
    "strategy", [f"{p}-template" for p in ["whole-brain", "gm", "wm"]]
)
# We parametrize mask_args to make it accessible
# to the expected_mask fixture.
@pytest.mark.parametrize("mask_args", [{"threshold": 0.0}])
def test_compute_brain_mask(strategy, expected_mask, mask_args):
    """Check masker for template masking strategy."""
    masker = NiftiMasker(mask_strategy=strategy, mask_args=mask_args)
    img, _ = data_gen.generate_random_img((9, 9, 5))

    masker.fit(img)

    np.testing.assert_array_equal(get_data(masker.mask_img_), expected_mask)


def test_invalid_mask_arg_for_strategy():
    """Pass mask_args specific to epi strategy should not fail.

    But a warning should be thrown.
    """
    masker = NiftiMasker(
        mask_strategy="background",
        mask_args={"lower_cutoff": 0.1, "ensure_finite": False},
    )
    img, _ = data_gen.generate_random_img((9, 9, 5))

    with pytest.warns(
        UserWarning, match="The following arguments are not supported by"
    ):
        masker.fit(img)


@pytest.mark.parametrize(
    "strategy", [f"{p}-template" for p in ["whole-brain", "gm", "wm"]]
)
def test_no_warning_partial_joblib(strategy):
    """Check no warning thrown by joblib regarding masking strategy.

    Regression test for:
    https://github.com/nilearn/nilearn/issues/5527
    """
    masker = NiftiMasker(
        mask_strategy=strategy,
        mask_args={"threshold": 0.0},
        memory="nilearn_cache",
        memory_level=1,
    )
    img, _ = data_gen.generate_random_img((9, 9, 5))
    with warnings.catch_warnings(record=True) as warning_list:
        masker.fit(img)

    assert not any(
        "Cannot inspect object functools.partial" in str(x)
        for x in warning_list
    )


def test_filter_and_mask_error(affine_eye):
    """Check filter_and_mask fails if mask if 4D."""
    data = np.zeros([20, 30, 40, 5])
    mask = np.zeros([20, 30, 40, 2])
    mask[10, 15, 20, :] = 1

    data_img = Nifti1Image(data, affine_eye)
    mask_img = Nifti1Image(mask, affine_eye)

    masker = NiftiMasker()
    params = get_params(NiftiMasker, masker)

    with pytest.raises(
        exceptions.DimensionError,
        match="Input data has incompatible dimensionality: "
        "Expected dimension is 3D and you provided "
        "a 4D image.",
    ):
        filter_and_mask(data_img, mask_img, params)


def test_filter_and_mask(affine_eye):
    """Test filter_and_mask returns output with correct shape."""
    data_shape = (20, 30, 40, 5)
    mask_shape = (20, 30, 40)
    data = np.zeros(data_shape)
    mask = np.ones(mask_shape)

    data_img = Nifti1Image(data, affine_eye)
    mask_img = Nifti1Image(mask, affine_eye)

    masker = NiftiMasker()
    params = get_params(NiftiMasker, masker)
    params["clean_kwargs"] = {}

    # Test return_affine = False
    data = filter_and_mask(data_img, mask_img, params)
    assert data.shape == (data_shape[3], np.prod(np.array(mask.shape)))


def test_standardization(rng, shape_3d_default, affine_eye):
    """Check output properly standardized with 'standardize' parameter."""
    n_samples = 500

    signals = rng.standard_normal(size=(np.prod(shape_3d_default), n_samples))
    means = (
        rng.standard_normal(size=(np.prod(shape_3d_default), 1)) * 50 + 1000
    )
    signals += means
    img = Nifti1Image(
        signals.reshape((*shape_3d_default, n_samples)),
        affine_eye,
    )

    mask = Nifti1Image(np.ones(shape_3d_default), affine_eye)

    # z-score
    masker = NiftiMasker(mask, standardize="zscore_sample")
    trans_signals = masker.fit_transform(img)

    np.testing.assert_almost_equal(trans_signals.mean(0), 0)
    np.testing.assert_almost_equal(trans_signals.std(0), 1, decimal=3)

    # psc
    masker = NiftiMasker(mask, standardize="psc")
    trans_signals = masker.fit_transform(img)

    np.testing.assert_almost_equal(trans_signals.mean(0), 0)
    np.testing.assert_almost_equal(
        trans_signals,
        (signals / signals.mean(1)[:, np.newaxis] * 100 - 100).T,
    )
