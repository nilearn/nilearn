"""Test the nifti_masker module.

Functions in this file only test features added by the NiftiMasker class,
not the underlying functions used (e.g. clean()). See test_masking.py and
test_signal.py for this.
"""

# Author: Gael Varoquaux, Philippe Gervais
import shutil
import warnings
from pathlib import Path
from tempfile import mkdtemp

import numpy as np
import pytest
from nibabel import Nifti1Image
from numpy.testing import assert_array_equal

from nilearn._utils import data_gen, exceptions, testing
from nilearn._utils.class_inspect import check_estimator, get_params
from nilearn._utils.helpers import is_matplotlib_installed
from nilearn.image import get_data, index_img
from nilearn.maskers import NiftiMasker
from nilearn.maskers.nifti_masker import _filter_and_mask

extra_valid_checks = [
    "check_parameters_default_constructible",
    "check_estimators_unfitted",
    "check_get_params_invariance",
    "check_transformer_n_iter",
    "check_transformers_unfitted",
]


@pytest.mark.parametrize(
    "estimator, check, name",
    check_estimator(
        estimator=[NiftiMasker()],
        extra_valid_checks=extra_valid_checks,
    ),
)
def test_check_estimator(estimator, check, name):  # noqa: ARG001
    """Check compliance with sklearn estimators."""
    check(estimator)


@pytest.mark.xfail(reason="invalid checks should fail")
@pytest.mark.parametrize(
    "estimator, check, name",
    check_estimator(
        estimator=[NiftiMasker()],
        extra_valid_checks=extra_valid_checks,
        valid=False,
    ),
)
def test_check_estimator_invalid(estimator, check, name):  # noqa: ARG001
    """Check compliance with sklearn estimators."""
    check(estimator)


def test_auto_mask(img_3d_rand_eye):
    """Perform a smoke test on the auto-mask option."""
    masker = NiftiMasker()
    # Smoke test the fit
    masker.fit(img_3d_rand_eye)
    # Smoke test the transform
    # With a 4D img
    masker.transform([img_3d_rand_eye])
    # With a 3D img
    masker.transform(img_3d_rand_eye)

    # check exception when transform() called without prior fit()
    masker2 = NiftiMasker(mask_img=img_3d_rand_eye)
    with pytest.raises(ValueError, match="has not been fitted. "):
        masker2.transform(img_3d_rand_eye)


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
    y = np.ones((9, 9, 9))
    masker = NiftiMasker(mask_img=mask_img_1)
    with pytest.warns(
        UserWarning,
        match="Generation of a mask has been requested .*"
        "while a mask has been provided at masker creation.",
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


def test_with_files(tmp_path, img_3d_rand_eye):
    """Test standard masking with filenames."""
    filename = testing.write_imgs_to_path(img_3d_rand_eye, file_path=tmp_path)
    masker = NiftiMasker()
    masker.fit(filename)
    masker.transform(filename)


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


def test_mask_3d_error(shape_4d_default, affine_eye):
    """Raise an error if 4D mask is provided with no img to fit."""
    # Dummy mask
    data = np.zeros(shape_4d_default)
    data[5, 5, 5] = 1
    mask_img = Nifti1Image(data, affine_eye)
    masker = NiftiMasker(mask_img=mask_img)
    with pytest.raises(TypeError, match="Expected dimension is 3D"):
        masker.fit()


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

    # Check attributes defined at fit
    assert not hasattr(masker, "mask_img_")
    assert not hasattr(masker, "n_elements_")

    masker.fit()

    # Check attributes defined at fit
    assert hasattr(masker, "mask_img_")
    assert hasattr(masker, "n_elements_")
    assert masker.n_elements_ == np.sum(mask)

    data_trans_5d = masker.transform(data_5d)
    data_trans_4d = masker.transform(data_4d)

    assert_array_equal(data_trans_4d, data_trans_5d)


def test_5d(rng, shape_3d_default, mask_img_1, affine_eye):
    """Test that list of 4D images with last dim=3 raises a DimensionError."""
    shape_4d = (*shape_3d_default, 3)
    data_5d = [rng.random(shape_4d) for _ in range(5)]
    data_5d = [Nifti1Image(d, affine_eye) for d in data_5d]

    masker = NiftiMasker(mask_img=mask_img_1)
    masker.fit()

    with pytest.raises(
        exceptions.DimensionError,
        match="Input data has incompatible dimensionality: "
        "Expected dimension is 4D and you provided "
        "a list of 4D images \\(5D\\).",
    ):
        masker.transform(data_5d)


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


def test_joblib_cache(tmp_path, mask_img_1):
    """Test using joblib cache."""
    from joblib import Memory, hash

    filename = testing.write_imgs_to_path(
        mask_img_1,
        file_path=tmp_path,
        create_files=True,
    )
    masker = NiftiMasker(mask_img=filename)
    masker.fit()
    mask_hash = hash(masker.mask_img_)
    get_data(masker.mask_img_)
    assert mask_hash == hash(masker.mask_img_)

    # Test a tricky issue with memmapped joblib.memory that makes
    # imgs return by inverse_transform impossible to save
    cachedir = Path(mkdtemp())
    try:
        masker.memory = Memory(location=cachedir, mmap_mode="r", verbose=0)
        X = masker.transform(mask_img_1)
        # inverse_transform a first time, so that the result is cached
        out_img = masker.inverse_transform(X)
        out_img = masker.inverse_transform(X)
        out_img.to_filename(cachedir / "test.nii")
    finally:
        # enables to delete "filename" on windows
        del masker
        shutil.rmtree(cachedir, ignore_errors=True)


def test_fit_no_mask_no_img_error():
    """Check error is raised when no mask and no img is provided."""
    mask = NiftiMasker(mask_img=None)
    with pytest.raises(
        ValueError, match="Parameter 'imgs' must be provided to "
    ):
        mask.fit()


def test_mask_strategy_errors(img_3d_rand_eye):
    """Check that mask_strategy errors are raised."""
    # Error with unknown mask_strategy
    mask = NiftiMasker(mask_strategy="oops")
    with pytest.raises(
        ValueError, match="Unknown value of mask_strategy 'oops'"
    ):
        mask.fit(img_3d_rand_eye)
    # Warning with deprecated 'template' strategy,
    # plus an exception because there's no resulting mask
    mask = NiftiMasker(mask_strategy="template")
    with pytest.warns(
        UserWarning, match="Masking strategy 'template' is deprecated."
    ):
        mask.fit(img_3d_rand_eye)


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
@pytest.mark.parametrize("mask_args", [{}, {"threshold": 0.0}])
def test_compute_brain_mask(strategy, mask_args, expected_mask):
    """Check masker for template masking strategy."""
    img, _ = data_gen.generate_random_img((9, 9, 5))

    masker = NiftiMasker(mask_strategy=strategy, mask_args=mask_args)
    masker.fit(img)

    np.testing.assert_array_equal(get_data(masker.mask_img_), expected_mask)


def test_filter_and_mask_error(affine_eye):
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
        _filter_and_mask(data_img, mask_img, params)


def test_filter_and_mask(affine_eye):
    data = np.zeros([20, 30, 40, 5])
    mask = np.ones([20, 30, 40])

    data_img = Nifti1Image(data, affine_eye)
    mask_img = Nifti1Image(mask, affine_eye)

    masker = NiftiMasker()
    params = get_params(NiftiMasker, masker)
    params["clean_kwargs"] = {}

    # Test return_affine = False
    data = _filter_and_mask(data_img, mask_img, params)
    assert data.shape == (5, 24000)


def test_dtype(shape_3d_default):
    data_32 = np.zeros(shape_3d_default, dtype=np.float32)
    data_64 = np.zeros(shape_3d_default, dtype=np.float64)
    data_32[2:-2, 2:-2, 2:-2] = 10
    data_64[2:-2, 2:-2, 2:-2] = 10

    affine_32 = np.eye(4, dtype=np.float32)
    affine_64 = np.eye(4, dtype=np.float64)

    img_32 = Nifti1Image(data_32, affine_32)
    img_64 = Nifti1Image(data_64, affine_64)

    masker_1 = NiftiMasker(dtype="auto")
    assert masker_1.fit_transform(img_32).dtype == np.float32
    assert masker_1.fit_transform(img_64).dtype == np.float32

    masker_2 = NiftiMasker(dtype="float64")
    assert masker_2.fit_transform(img_32).dtype == np.float64
    assert masker_2.fit_transform(img_64).dtype == np.float64


def test_standardization(rng, shape_3d_default, affine_eye):
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


def test_nifti_masker_io_shapes(rng, shape_3d_default, affine_eye):
    """Ensure that NiftiMasker handles 1D/2D/3D/4D data appropriately.

    transform(4D image) --> 2D output, no warning
    transform(3D image) --> 2D output, DeprecationWarning
    inverse_transform(2D array) --> 4D image, no warning
    inverse_transform(1D array) --> 3D image, no warning
    inverse_transform(2D array with wrong shape) --> ValueError
    """
    n_volumes = 5
    shape_4d = (*shape_3d_default, n_volumes)

    img_4d, mask_img = data_gen.generate_random_img(
        shape_4d,
        affine=affine_eye,
    )
    img_3d, _ = data_gen.generate_random_img(
        shape_3d_default, affine=affine_eye
    )
    n_regions = np.sum(mask_img.get_fdata().astype(bool))
    data_1d = rng.random(n_regions)
    data_2d = rng.random((n_volumes, n_regions))

    masker = NiftiMasker(mask_img)
    masker.fit()

    # DeprecationWarning *should* be raised for 3D inputs
    with pytest.deprecated_call(match="Starting in version 0.12"):
        test_data = masker.transform(img_3d)
        assert test_data.shape == (1, n_regions)

    # DeprecationWarning should *not* be raised for 4D inputs
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "error",
            message="Starting in version 0.12",
            category=DeprecationWarning,
        )
        test_data = masker.transform(img_4d)
        assert test_data.shape == (n_volumes, n_regions)

    # DeprecationWarning should *not* be raised for 1D inputs
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "error",
            message="Starting in version 0.12",
            category=DeprecationWarning,
        )
        test_img = masker.inverse_transform(data_1d)
        assert test_img.shape == shape_3d_default

    # DeprecationWarning should *not* be raised for 2D inputs
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "error",
            message="Starting in version 0.12",
            category=DeprecationWarning,
        )
        test_img = masker.inverse_transform(data_2d)
        assert test_img.shape == shape_4d

    with pytest.raises(TypeError):
        masker.inverse_transform(data_2d.T)


@pytest.mark.skipif(
    is_matplotlib_installed(),
    reason="Test requires matplotlib not to be installed.",
)
def test_nifti_masker_reporting_mpl_warning():
    """Raise warning after exception if matplotlib is not installed."""
    with warnings.catch_warnings(record=True) as warning_list:
        result = NiftiMasker().generate_report()

    assert len(warning_list) == 1
    assert issubclass(warning_list[0].category, ImportWarning)
    assert result == [None]
