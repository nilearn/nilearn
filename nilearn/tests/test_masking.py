"""Test the mask-extracting utilities."""

import warnings

import numpy as np
import pytest
from nibabel import Nifti1Image
from numpy.testing import assert_array_equal
from sklearn.preprocessing import StandardScaler

from nilearn import masking

# Authors: Ana Luisa Pinho, Jerome Dockes, NicolasGensollen
from nilearn._utils import data_gen
from nilearn._utils.exceptions import DimensionError
from nilearn._utils.testing import write_imgs_to_path
from nilearn.conftest import _rng
from nilearn.image import get_data, high_variance_confounds
from nilearn.maskers import NiftiMasker
from nilearn.masking import (
    _MaskWarning,
    _unmask_3d,
    _unmask_4d,
    compute_background_mask,
    compute_brain_mask,
    compute_epi_mask,
    compute_multi_brain_mask,
    compute_multi_epi_mask,
    extrapolate_out_mask,
    intersect_masks,
    unmask,
    unmask_from_to_3d_array,
)

np_version = (
    np.version.full_version
    if hasattr(np.version, "full_version")
    else np.version.short_version
)

_TEST_DIM_ERROR_MSG = (
    "Input data has incompatible dimensionality: "
    "Expected dimension is 3D and you provided "
    "a %s image"
)


def _simu_img():
    # Random confounds
    rng = _rng()
    conf = 2 + rng.standard_normal((100, 6))
    # Random 4D volume
    vol = 100 + 10 * rng.standard_normal((5, 5, 2, 100))
    img = Nifti1Image(vol, np.eye(4))
    # Create an nifti image with the data, and corresponding mask
    mask = Nifti1Image(np.ones([5, 5, 2]), np.eye(4))
    return img, mask, conf


def _cov_conf(tseries, conf):
    conf_n = StandardScaler().fit_transform(conf)
    _ = StandardScaler().fit_transform(tseries)
    cov_mat = np.dot(tseries.T, conf_n)
    return cov_mat


def _confounds_regression(
    standardize_signal="zscore_sample", standardize_confounds=True
):
    img, mask, conf = _simu_img()
    masker = NiftiMasker(
        standardize=standardize_signal,
        standardize_confounds=standardize_confounds,
        detrend=False,
        mask_img=mask,
    ).fit()
    tseries = masker.transform(img, confounds=conf)
    if standardize_confounds:
        conf = StandardScaler(with_std=False).fit_transform(conf)
    cov_mat = _cov_conf(tseries, conf)
    return np.sum(np.abs(cov_mat))


def test_high_variance_confounds():
    img, mask, conf = _simu_img()
    hv_confounds = high_variance_confounds(img)
    masker1 = NiftiMasker(
        standardize="zscore_sample",
        detrend=False,
        high_variance_confounds=False,
        mask_img=mask,
    ).fit()
    tseries1 = masker1.transform(img, confounds=[hv_confounds, conf])
    masker2 = NiftiMasker(
        standardize="zscore_sample",
        detrend=False,
        high_variance_confounds=True,
        mask_img=mask,
    ).fit()
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
    assert (
        _confounds_regression(
            standardize_signal=False, standardize_confounds=True
        )
        < 10.0 * eps
    )

    # Signal is z-scored with string arg
    # Explicit standardization of confounds
    assert (
        _confounds_regression(
            standardize_signal="zscore_sample", standardize_confounds=True
        )
        < eps
    )

    # Signal is psc standardized
    # Explicit standardization of confounds
    assert (
        _confounds_regression(
            standardize_signal="psc", standardize_confounds=True
        )
        < 10.0 * eps
    )

    # Signal is not standardized
    # Confounds are not standardized
    # In this case, the regression should fail...
    assert (
        _confounds_regression(
            standardize_signal=False, standardize_confounds=False
        )
        > 100
    )

    # Signal is z-scored with string arg
    # Confounds are not standardized
    # In this case, the regression should fail...
    assert (
        _confounds_regression(
            standardize_signal="zscore_sample", standardize_confounds=False
        )
        > 100
    )

    # Signal is psc standardized
    # Confounds are not standardized
    # In this case, the regression should fail...
    assert (
        _confounds_regression(
            standardize_signal="psc", standardize_confounds=False
        )
        > 100
    )


def test_compute_epi_mask(affine_eye):
    mean_image = np.ones((9, 9, 3))
    mean_image[3:-2, 3:-2, :] = 10
    mean_image[5, 5, :] = 11
    mean_image = Nifti1Image(mean_image, affine_eye)
    mask1 = compute_epi_mask(mean_image, opening=False, verbose=1)
    mask2 = compute_epi_mask(mean_image, exclude_zeros=True, opening=False)
    # With an array with no zeros, exclude_zeros should not make
    # any difference
    np.testing.assert_array_equal(get_data(mask1), get_data(mask2))
    # Check that padding with zeros does not change the extracted mask
    mean_image2 = np.zeros((30, 30, 3))
    mean_image2[3:12, 3:12, :] = get_data(mean_image)
    mean_image2 = Nifti1Image(mean_image2, affine_eye)
    mask3 = compute_epi_mask(mean_image2, exclude_zeros=True, opening=False)
    np.testing.assert_array_equal(get_data(mask1), get_data(mask3)[3:12, 3:12])
    # However, without exclude_zeros, it does
    mask3 = compute_epi_mask(mean_image2, opening=False)
    assert not np.allclose(get_data(mask1), get_data(mask3)[3:12, 3:12])

    # Check that we get a ValueError for incorrect shape
    mean_image = np.ones((9, 9))
    mean_image[3:-3, 3:-3] = 10
    mean_image[5, 5] = 100
    mean_image = Nifti1Image(mean_image, affine_eye)
    with pytest.raises(ValueError):
        compute_epi_mask(mean_image)

    # Check that we get a useful warning for empty masks
    mean_image = np.zeros((9, 9, 9))
    mean_image[0, 0, 1] = -1
    mean_image[0, 0, 0] = 1.2
    mean_image[0, 0, 2] = 1.1
    mean_image = Nifti1Image(mean_image, affine_eye)
    with pytest.warns(_MaskWarning, match="Computed an empty mask"):
        compute_epi_mask(mean_image, exclude_zeros=True)


def test_compute_background_mask(affine_eye):
    for value in (0, np.nan):
        mean_image = value * np.ones((9, 9, 9))
        mean_image[3:-3, 3:-3, 3:-3] = 1
        mask = mean_image == 1
        mean_image = Nifti1Image(mean_image, affine_eye)
        mask1 = compute_background_mask(mean_image, opening=False, verbose=1)
        np.testing.assert_array_equal(get_data(mask1), mask.astype(np.int8))

    # Check that we get a ValueError for incorrect shape
    mean_image = np.ones((9, 9))
    mean_image[3:-3, 3:-3] = 10
    mean_image[5, 5] = 100
    mean_image = Nifti1Image(mean_image, affine_eye)
    with pytest.raises(ValueError):
        compute_background_mask(mean_image)

    # Check that we get a useful warning for empty masks
    mean_image = np.zeros((9, 9, 9))
    mean_image = Nifti1Image(mean_image, affine_eye)
    with pytest.warns(_MaskWarning, match="Computed an empty mask"):
        compute_background_mask(mean_image)


def test_compute_brain_mask():
    img, _ = data_gen.generate_mni_space_img(res=8, random_state=0)
    brain_mask = compute_brain_mask(img, threshold=0.2, verbose=1)
    gm_mask = compute_brain_mask(img, threshold=0.2, mask_type="gm")
    wm_mask = compute_brain_mask(img, threshold=0.2, mask_type="wm")
    brain_data, gm_data, wm_data = map(
        get_data, (brain_mask, gm_mask, wm_mask)
    )
    # Check that whole-brain mask is non-empty
    assert (brain_data != 0).any()
    for subset in gm_data, wm_data:
        # Test that gm and wm masks are included in the whole-brain mask
        assert (
            np.logical_and(brain_data, subset) == subset.astype(bool)
        ).all()
        # Test that gm and wm masks are non-empty
        assert (subset != 0).any()
    # Test that gm and wm masks have empty intersection
    assert (np.logical_and(gm_data, wm_data) == 0).all()
    # Check that we get a useful warning for empty masks
    with pytest.warns(masking._MaskWarning):
        compute_brain_mask(img, threshold=1)
    # Check that masks obtained from same FOV are the same
    img1, _ = data_gen.generate_mni_space_img(res=8, random_state=1)
    mask_img1 = compute_brain_mask(img1, verbose=1, threshold=0.2)
    assert (brain_data == get_data(mask_img1)).all()
    # Check that error is raised if mask type is unknown
    with pytest.raises(ValueError, match="Unknown mask type foo."):
        compute_brain_mask(img, verbose=1, mask_type="foo")


def test_apply_mask(affine_eye, tmp_path):
    """Test smoothing of timeseries extraction."""
    # A delta in 3D
    # Standard masking
    data = np.zeros((40, 40, 40, 2))
    data[20, 20, 20] = 1
    mask = np.ones((40, 40, 40))
    full_mask = np.zeros((40, 40, 40))
    for create_files in (False, True):
        for affine in (
            affine_eye,
            np.diag((1, 1, -1, 1)),
            np.diag((0.5, 1, 0.5, 1)),
        ):
            data_img = Nifti1Image(data, affine)
            mask_img = Nifti1Image(mask, affine)
            filenames = write_imgs_to_path(
                data_img,
                mask_img,
                file_path=tmp_path,
                create_files=create_files,
            )
            series = masking.apply_mask(
                filenames[0], filenames[1], smoothing_fwhm=9
            )

            series = np.reshape(series[0, :], (40, 40, 40))
            vmax = series.max()
            # We are expecting a full-width at half maximum of
            # 9mm/voxel_size:
            above_half_max = series > 0.5 * vmax
            for axis in (0, 1, 2):
                proj = np.any(
                    np.any(np.rollaxis(above_half_max, axis=axis), axis=-1),
                    axis=-1,
                )
                np.testing.assert_equal(
                    proj.sum(), 9 / np.abs(affine[axis, axis])
                )

    # Check that NaNs in the data do not propagate
    data[10, 10, 10] = np.nan
    data_img = Nifti1Image(data, affine)
    mask_img = Nifti1Image(mask, affine)
    full_mask_img = Nifti1Image(full_mask, affine)
    series = masking.apply_mask(data_img, mask_img, smoothing_fwhm=9)
    assert np.all(np.isfinite(series))

    # veriy that 4D masks are rejected
    mask_img_4d = Nifti1Image(np.ones((40, 40, 40, 2)), affine_eye)
    with pytest.raises(DimensionError, match=_TEST_DIM_ERROR_MSG % "4D"):
        masking.apply_mask(data_img, mask_img_4d)

    # Check that 3D data is accepted
    data_3d = Nifti1Image(
        np.arange(27, dtype="int32").reshape((3, 3, 3)), affine_eye
    )
    mask_data_3d = np.zeros((3, 3, 3))
    mask_data_3d[1, 1, 0] = True
    mask_data_3d[0, 1, 0] = True
    mask_data_3d[0, 1, 1] = True
    data_3d = masking.apply_mask(
        data_3d, Nifti1Image(mask_data_3d, affine_eye)
    )
    assert sorted(data_3d.tolist()) == [3.0, 4.0, 12.0]

    # Check data shape and affine
    with pytest.raises(DimensionError, match=_TEST_DIM_ERROR_MSG % "2D"):
        masking.apply_mask(data_img, Nifti1Image(mask[20, ...], affine))
    with pytest.raises(ValueError):
        masking.apply_mask(data_img, Nifti1Image(mask, affine / 2.0))
    # Check that full masking raises error
    with pytest.raises(ValueError):
        masking.apply_mask(data_img, full_mask_img)
    # Check weird values in data
    mask[10, 10, 10] = 2
    with pytest.raises(ValueError):
        masking.apply_mask(data_img, Nifti1Image(mask, affine))
    mask[15, 15, 15] = 3
    with pytest.raises(ValueError):
        masking.apply_mask(Nifti1Image(data, affine), mask_img)


def test_unmask(rng, affine_eye, tmp_path):
    # A delta in 3D
    shape = (10, 20, 30, 40)
    data4D = rng.uniform(size=shape)
    data3D = data4D[..., 0]
    mask = rng.integers(2, size=shape[:3], dtype="int32")
    mask_img = Nifti1Image(mask, affine_eye)
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
        filename = write_imgs_to_path(
            mask_img,
            file_path=tmp_path,
            create_files=create_files,
        )
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
    with pytest.raises(TypeError):
        unmask(vec_1D, mask_img)
    with pytest.raises(TypeError):
        unmask([vec_1D], mask_img)

    vec_2D = np.empty((500, 500), dtype=np.float64)
    with pytest.raises(TypeError):
        unmask(vec_2D, mask_img)
    with pytest.raises(TypeError):
        unmask([vec_2D], mask_img)

    # Error test: mask type
    with pytest.raises(TypeError, match="mask must be a boolean array"):
        _unmask_3d(vec_1D, mask.astype(int))
    with pytest.raises(TypeError, match="mask must be a boolean array"):
        _unmask_4d(vec_2D, mask.astype(np.float64))

    # Transposed vector
    transposed_vector = np.ones((np.sum(mask), 1), dtype=bool)
    with pytest.raises(TypeError, match="X must be of shape"):
        unmask(transposed_vector, mask_img)


def test_intersect_masks_filename(affine_eye, tmp_path):
    # Create dummy masks
    mask_a = np.zeros((4, 4, 1), dtype=bool)
    mask_a[2:4, 2:4] = 1
    mask_a_img = Nifti1Image(mask_a.astype("int32"), affine_eye)

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
    mask_b_img = Nifti1Image(mask_b.astype("int32"), affine_eye)

    # +---+---+---+---+
    # |   |   |   |   |
    # +---+---+---+---+
    # |   | X | X |   |
    # +---+---+---+---+
    # |   | X | X |   |
    # +---+---+---+---+
    # |   |   |   |   |
    # +---+---+---+---+

    filenames = write_imgs_to_path(
        mask_a_img, mask_b_img, file_path=tmp_path, create_files=True
    )
    mask_ab = np.zeros((4, 4, 1), dtype=bool)
    mask_ab[2, 2] = 1
    mask_ab_ = intersect_masks(filenames, threshold=1.0)
    assert_array_equal(mask_ab, get_data(mask_ab_))


def test_intersect_masks(affine_eye):
    """Test the intersect_masks function."""
    # Create dummy masks
    mask_a = np.zeros((4, 4, 1), dtype=bool)
    mask_a[2:4, 2:4] = 1
    mask_a_img = Nifti1Image(mask_a.astype("int32"), affine_eye)

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
    mask_b_img = Nifti1Image(mask_b.astype("int32"), affine_eye)

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
    mask_c_img = Nifti1Image(mask_c.astype("int32"), affine_eye)

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
    mask_ab_ = intersect_masks([mask_a_img, mask_b_img], threshold=1.0)
    assert_array_equal(mask_ab, get_data(mask_ab_))
    # Test intersect mask images with '>f8'. This function uses
    # largest_connected_component to check if intersect_masks passes with
    # connected=True (which is by default)
    mask_a_img_change_dtype = Nifti1Image(
        get_data(mask_a_img).astype(">f8"), affine=mask_a_img.affine
    )
    mask_b_img_change_dtype = Nifti1Image(
        get_data(mask_b_img).astype(">f8"), affine=mask_b_img.affine
    )
    mask_ab_change_type = intersect_masks(
        [mask_a_img_change_dtype, mask_b_img_change_dtype], threshold=1.0
    )
    assert_array_equal(mask_ab, get_data(mask_ab_change_type))

    mask_abc = mask_a + mask_b + mask_c
    mask_abc_ = intersect_masks(
        [mask_a_img, mask_b_img, mask_c_img], threshold=0.0, connected=False
    )
    assert_array_equal(mask_abc, get_data(mask_abc_))

    mask_abc[0, 0] = 0
    mask_abc_ = intersect_masks(
        [mask_a_img, mask_b_img, mask_c_img], threshold=0.0
    )
    assert_array_equal(mask_abc, get_data(mask_abc_))

    mask_abc = mask_ab
    mask_abc_ = intersect_masks(
        [mask_a_img, mask_b_img, mask_c_img], threshold=1.0
    )
    assert_array_equal(mask_abc, get_data(mask_abc_))

    mask_abc[1, 2] = 1
    mask_abc[3, 2] = 1
    mask_abc_ = intersect_masks([mask_a_img, mask_b_img, mask_c_img])
    assert_array_equal(mask_abc, get_data(mask_abc_))


def test_compute_multi_epi_mask(affine_eye):
    # Check that an empty list of images creates a meaningful error
    with pytest.raises(TypeError):
        compute_multi_epi_mask([])
    # As it calls intersect_masks, we only test resampling here.
    # Same masks as test_intersect_masks
    mask_a = np.zeros((4, 4, 1), dtype=bool)
    mask_a[2:4, 2:4] = 1
    mask_a_img = Nifti1Image(mask_a.astype("uint8"), affine_eye)

    mask_b = np.zeros((8, 8, 1), dtype=bool)
    mask_b[2:6, 2:6] = 1
    mask_b_img = Nifti1Image(mask_b.astype("uint8"), affine_eye / 2.0)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", _MaskWarning)
        with pytest.raises(ValueError):
            compute_multi_epi_mask([mask_a_img, mask_b_img])
    mask_ab = np.zeros((4, 4, 1), dtype=bool)
    mask_ab[2, 2] = 1
    mask_ab_ = compute_multi_epi_mask(
        [mask_a_img, mask_b_img],
        threshold=1.0,
        opening=0,
        target_affine=affine_eye,
        target_shape=(4, 4, 1),
        verbose=1,
    )
    assert_array_equal(mask_ab, get_data(mask_ab_))


def test_compute_multi_brain_mask():
    with pytest.raises(TypeError):
        compute_multi_brain_mask([])

    # Check error raised if images with different shapes are given as input
    imgs = [
        data_gen.generate_mni_space_img(res=8, random_state=0)[0],
        data_gen.generate_mni_space_img(res=12, random_state=0)[0],
    ]
    with pytest.raises(ValueError):
        compute_multi_brain_mask(imgs)

    # Check results are the same if affine is the same
    imgs1 = [
        data_gen.generate_mni_space_img(res=9, random_state=0)[0],
        data_gen.generate_mni_space_img(res=9, random_state=1)[0],
    ]
    imgs2 = [
        data_gen.generate_mni_space_img(res=9, random_state=2)[0],
        data_gen.generate_mni_space_img(res=9, random_state=3)[0],
    ]
    mask1 = compute_multi_brain_mask(imgs1, threshold=0.2, verbose=1)
    mask2 = compute_multi_brain_mask(imgs2, threshold=0.2)
    assert_array_equal(get_data(mask1), get_data(mask2))


def test_error_shape(rng, affine_eye, shape=(3, 5, 7, 11)):
    # open-ended `if .. elif` in masking.unmask

    # setup
    X = rng.standard_normal()
    mask_img = np.zeros(shape, dtype=np.uint8)
    mask_img[rng.standard_normal(size=shape) > 0.4] = 1
    n_features = (mask_img > 0).sum()
    mask_img = Nifti1Image(mask_img, affine_eye)
    n_samples = shape[0]

    X = rng.standard_normal(size=(n_samples, n_features, 2))
    # 3D X (unmask should raise a TypeError)
    with pytest.raises(TypeError):
        unmask(X, mask_img)

    X = rng.standard_normal(size=(n_samples, n_features))
    # Raises an error because the mask is 4D
    with pytest.raises(TypeError):
        unmask(X, mask_img)


def test_nifti_masker_empty_mask_warning(affine_eye):
    X = Nifti1Image(np.ones((2, 2, 2, 5)), affine_eye)
    with pytest.raises(
        ValueError,
        match="The mask is invalid as it is empty: it masks all data",
    ):
        NiftiMasker(mask_strategy="epi").fit_transform(X)


def test_unmask_list(rng, affine_eye):
    shape = (3, 4, 5)
    mask_data = rng.uniform(size=shape) < 0.5
    mask_img = Nifti1Image(mask_data.astype(np.uint8), affine_eye)
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
    target_data = np.array(
        [
            [
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
            ],
            [
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.5, 0.0, 0.0],
                [0.0, 2.0, 1.0, 3.5, 0.0],
                [0.0, 0.0, 3.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
            ],
            [
                [0.0, 0.0, 2.0, 0.0, 0.0],
                [0.0, 2.5, 2.0, 4.0, 0.0],
                [3.0, 3.0, 3.5, 6.0, 6.0],
                [0.0, 4.0, 5.0, 5.5, 0.0],
                [0.0, 0.0, 5.0, 0.0, 0.0],
            ],
            [
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 3.0, 0.0, 0.0],
                [0.0, 3.5, 4.0, 5.0, 0.0],
                [0.0, 0.0, 4.5, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
            ],
            [
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 4.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
            ],
        ]
    )
    target_mask = np.array(
        [
            [
                [False, False, False, False, False],
                [False, False, False, False, False],
                [False, False, True, False, False],
                [False, False, False, False, False],
                [False, False, False, False, False],
            ],
            [
                [False, False, False, False, False],
                [False, False, True, False, False],
                [False, True, True, True, False],
                [False, False, True, False, False],
                [False, False, False, False, False],
            ],
            [
                [False, False, True, False, False],
                [False, True, True, True, False],
                [True, True, True, True, True],
                [False, True, True, True, False],
                [False, False, True, False, False],
            ],
            [
                [False, False, False, False, False],
                [False, False, True, False, False],
                [False, True, True, True, False],
                [False, False, True, False, False],
                [False, False, False, False, False],
            ],
            [
                [False, False, False, False, False],
                [False, False, False, False, False],
                [False, False, True, False, False],
                [False, False, False, False, False],
                [False, False, False, False, False],
            ],
        ]
    )

    # Test:
    extrapolated_data, extrapolated_mask = extrapolate_out_mask(
        initial_data, initial_mask, iterations=1
    )
    assert_array_equal(extrapolated_data, target_data)
    assert_array_equal(extrapolated_mask, target_mask)


def test_unmask_from_to_3d_array(rng, size=5):
    for ndim in range(1, 4):
        shape = [size] * ndim
        mask = np.zeros(shape).astype(bool)
        mask[rng.uniform(size=shape) > 0.8] = 1
        support = rng.standard_normal(size=mask.sum())
        full = unmask_from_to_3d_array(support, mask)
        np.testing.assert_array_equal(full.shape, shape)
        np.testing.assert_array_equal(full[mask], support)
