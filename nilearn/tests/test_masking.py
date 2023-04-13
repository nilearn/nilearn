"""Test the mask-extracting utilities."""

# Authors: Ana Luisa Pinho, Jerome Dockes, NicolasGensollen
# License: simplified BSD

import warnings

import numpy as np
import pytest
from nibabel import Nifti1Image
from nilearn._utils.data_gen import generate_mni_space_img
from nilearn._utils.exceptions import DimensionError
from nilearn._utils.testing import write_tmp_imgs
from nilearn.image import get_data, high_variance_confounds
from nilearn.maskers import NiftiMasker
from nilearn.masking import (
    MaskWarning,
    _extrapolate_out_mask,
    _unmask_3d,
    _unmask_4d,
    _unmask_from_to_3d_array,
    apply_mask,
    compute_background_mask,
    compute_brain_mask,
    compute_epi_mask,
    compute_multi_brain_mask,
    compute_multi_epi_mask,
    intersect_masks,
    unmask,
)
from numpy.testing import assert_array_equal, assert_equal
from sklearn.preprocessing import StandardScaler

_TEST_DIM_ERROR_MSG = (
    "Input data has incompatible dimensionality: "
    "Expected dimension is 3D and you provided "
    "a %s image"
)
AFFINE_EYE = np.eye(4)
SHAPE_2D = (9, 9)
SHAPE_3D = (9, 9, 9)
SHAPE_4D = (9, 9, 9, 9)


def _confounds_regression(standardize_signal=True, standardize_confounds=True):
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


def _simu_img():
    # Random confounds
    conf = 2 + np.random.randn(100, 6)
    # Random 4D volume
    vol = 100 + 10 * np.random.randn(5, 5, 2, 100)
    img = Nifti1Image(vol, AFFINE_EYE)
    # Create an nifti image with the data, and corresponding mask
    mask = Nifti1Image(np.ones([5, 5, 2]), AFFINE_EYE)
    return img, mask, conf


def _cov_conf(tseries, conf):
    conf_n = StandardScaler().fit_transform(conf)
    StandardScaler().fit_transform(tseries)
    cov_mat = np.dot(tseries.T, conf_n)
    return cov_mat


@pytest.mark.parametrize(
    "standardize_signal, expected",
    [(False, 10.0), ("zscore_sample", 1.0), (True, 1.0), ("psc", 10.0)],
)
def test_confounds_standardization_confounds_are_standarized(
    standardize_signal, expected
):
    """See Issue #2584
    Code from @pbellec

    Confounds are explicitly standardized
    """
    eps = 10e-10

    # Signal is not standardized
    # Explicit standardization of confounds
    assert (
        _confounds_regression(
            standardize_signal=standardize_signal, standardize_confounds=True
        )
        < expected * eps
    )


@pytest.mark.parametrize("standardize_signal", [False, "zscore_sample", "psc"])
def test_confounds_standardization_confounds_are_not_standarized(
    standardize_signal,
):
    """See Issue #2584
    Code from @pbellec

    Confounds are not standardized
    In this case, the regression should fail...
    """
    assert (
        _confounds_regression(
            standardize_signal=standardize_signal, standardize_confounds=False
        )
        > 100
    )


def test_high_variance_confounds():
    img, mask, conf = _simu_img()

    hv_confounds = high_variance_confounds(img)

    masker1 = NiftiMasker(
        standardize=True,
        detrend=False,
        high_variance_confounds=False,
        mask_img=mask,
    ).fit()
    tseries1 = masker1.transform(img, confounds=[hv_confounds, conf])

    masker2 = NiftiMasker(
        standardize=True,
        detrend=False,
        high_variance_confounds=True,
        mask_img=mask,
    ).fit()
    tseries2 = masker2.transform(img, confounds=conf)

    assert_array_equal(tseries1, tseries2)


def test_compute_epi_mask():
    mean_image = np.ones((9, 9, 3))
    mean_image[3:-2, 3:-2, :] = 10
    mean_image[5, 5, :] = 11
    mean_image = Nifti1Image(mean_image, AFFINE_EYE)

    mask1 = compute_epi_mask(mean_image, opening=False)

    mask2 = compute_epi_mask(mean_image, exclude_zeros=True, opening=False)

    # With an array with no zeros, exclude_zeros should not make
    # any difference
    assert_array_equal(get_data(mask1), get_data(mask2))

    # Check that padding with zeros does not change the extracted mask
    mean_image2 = np.zeros((30, 30, 3))
    mean_image2[3:12, 3:12, :] = get_data(mean_image)
    mean_image2 = Nifti1Image(mean_image2, AFFINE_EYE)

    mask3 = compute_epi_mask(mean_image2, exclude_zeros=True, opening=False)

    assert_array_equal(get_data(mask1), get_data(mask3)[3:12, 3:12])

    # However, without exclude_zeros, it does
    mask3 = compute_epi_mask(mean_image2, opening=False)

    assert not np.allclose(get_data(mask1), get_data(mask3)[3:12, 3:12])


def test_compute_epi_mask_errors_warnings():
    # Check that we get a ValueError for incorrect shape
    mean_image = np.ones(SHAPE_2D)
    mean_image[3:-3, 3:-3] = 10
    mean_image[5, 5] = 100
    mean_image = Nifti1Image(mean_image, AFFINE_EYE)

    with pytest.raises(
        ValueError, match="Computation expects 3D or 4D images"
    ):
        compute_epi_mask(mean_image)

    # Check that we get a useful warning for empty masks
    mean_image = np.zeros(SHAPE_3D)
    mean_image[0, 0, 1] = -1
    mean_image[0, 0, 0] = 1.2
    mean_image[0, 0, 2] = 1.1
    mean_image = Nifti1Image(mean_image, AFFINE_EYE)

    with pytest.warns(MaskWarning, match="Computed an empty mask"):
        compute_epi_mask(mean_image, exclude_zeros=True)


@pytest.mark.parametrize("value", [0, np.nan])
def test_compute_background_mask(value):
    mean_image = value * np.ones(SHAPE_3D)
    mean_image[3:-3, 3:-3, 3:-3] = 1
    mask = mean_image == 1
    mean_image = Nifti1Image(mean_image, AFFINE_EYE)

    mask1 = compute_background_mask(mean_image, opening=False)

    assert_array_equal(get_data(mask1), mask.astype(np.int8))


def test_compute_background_mask_errors_warnings():
    # Check that we get a ValueError for incorrect shape
    mean_image = np.ones(SHAPE_2D)
    mean_image[3:-3, 3:-3] = 10
    mean_image[5, 5] = 100
    mean_image = Nifti1Image(mean_image, AFFINE_EYE)

    with pytest.raises(
        ValueError, match="Computation expects 3D or 4D images"
    ):
        compute_background_mask(mean_image)

    # Check that we get a useful warning for empty masks
    mean_image = np.zeros(SHAPE_3D)
    mean_image = Nifti1Image(mean_image, AFFINE_EYE)

    with pytest.warns(MaskWarning, match="Computed an empty mask"):
        compute_background_mask(mean_image)


def test_compute_brain_mask():
    img, _ = generate_mni_space_img(res=8, random_state=0)

    brain_mask = compute_brain_mask(img, threshold=0.2)

    gm_mask = compute_brain_mask(img, threshold=0.2, mask_type="gm")

    wm_mask = compute_brain_mask(img, threshold=0.2, mask_type="wm")

    brain_data, gm_data, wm_data = map(
        get_data, (brain_mask, gm_mask, wm_mask)
    )

    # Check that whole-brain mask is non-empty
    assert (brain_data != 0).any()
    # Test that gm and wm masks have empty intersection
    assert (np.logical_and(gm_data, wm_data) == 0).all()
    for subset in gm_data, wm_data:
        # Test that gm and wm masks are included in the whole-brain mask
        assert (
            np.logical_and(brain_data, subset) == subset.astype(bool)
        ).all()
        # Test that gm and wm masks are non-empty
        assert (subset != 0).any()

    # Check that masks obtained from same FOV are the same
    img1, _ = generate_mni_space_img(res=8, random_state=1)

    mask_img1 = compute_brain_mask(img1, verbose=1, threshold=0.2)

    assert (get_data(mask_img1) == brain_data).all()


def test_compute_brain_mask_errors_warnings():
    img, _ = generate_mni_space_img(res=8, random_state=0)

    # Check that we get a useful warning for empty masks
    with pytest.warns(MaskWarning):
        compute_brain_mask(img, threshold=1)

    # Check that error is raised if mask type is unknown
    with pytest.raises(ValueError, match="Unknown mask type foo."):
        compute_brain_mask(img, verbose=1, mask_type="foo")


@pytest.mark.parametrize("create_files", [False, True])
@pytest.mark.parametrize(
    "affine",
    [
        AFFINE_EYE,
        np.diag((1, 1, -1, 1)),
        np.diag((0.5, 1, 0.5, 1)),
    ],
)
def test_apply_mask(create_files, affine):
    """Test smoothing of timeseries extraction"""
    shape = (40, 40, 40)

    data = np.zeros((40, 40, 40, 2))
    data[20, 20, 20] = 1
    data_img = Nifti1Image(data, affine)

    mask = np.ones(shape)
    mask_img = Nifti1Image(mask, affine)

    with write_tmp_imgs(
        data_img, mask_img, create_files=create_files
    ) as filenames:
        series = apply_mask(filenames[0], filenames[1], smoothing_fwhm=9)

    series = np.reshape(series[0, :], shape)
    vmax = series.max()
    # We are expecting a full-width at half maximum of 9mm/voxel_size:
    above_half_max = series > 0.5 * vmax
    for axis in (0, 1, 2):
        proj = np.any(
            np.any(np.rollaxis(above_half_max, axis=axis), axis=-1),
            axis=-1,
        )
        assert_equal(proj.sum(), 9 / np.abs(affine[axis, axis]))


def test_apply_mask_nans_do_not_propagate():
    data = np.zeros(SHAPE_4D)
    data[5, 5, 5] = 1
    data[6, 6, 6] = np.NaN
    data_img = Nifti1Image(data, AFFINE_EYE)

    mask = np.ones(SHAPE_3D)
    mask_img = Nifti1Image(mask, AFFINE_EYE)

    series = apply_mask(data_img, mask_img, smoothing_fwhm=9)

    assert np.all(np.isfinite(series))


def test_apply_mask_3d_data_is_accepted():
    shape = (3, 3, 3)
    data_3d = Nifti1Image(
        np.arange(27, dtype="int32").reshape(shape), AFFINE_EYE
    )

    mask_data_3d = np.zeros(shape)
    mask_data_3d[1, 1, 0] = True
    mask_data_3d[0, 1, 0] = True
    mask_data_3d[0, 1, 1] = True

    data_3d = apply_mask(data_3d, Nifti1Image(mask_data_3d, AFFINE_EYE))

    assert sorted(data_3d.tolist()) == [3.0, 4.0, 12.0]


def test_apply_mask_errors():
    data = np.zeros(SHAPE_4D)
    data_img = Nifti1Image(data, AFFINE_EYE)

    # veriy that 4D masks are rejected
    mask_img_4d = Nifti1Image(np.ones(SHAPE_4D), AFFINE_EYE)
    with pytest.raises(DimensionError, match=_TEST_DIM_ERROR_MSG % "4D"):
        apply_mask(data_img, mask_img_4d)

    # Check data shape and affine
    mask = np.ones(SHAPE_3D)
    with pytest.raises(DimensionError, match=_TEST_DIM_ERROR_MSG % "2D"):
        apply_mask(data_img, Nifti1Image(mask[5, ...], AFFINE_EYE))

    with pytest.raises(ValueError, match="Mask affine"):
        apply_mask(
            data_img,
            Nifti1Image(mask, AFFINE_EYE / 2.0),
        )

    # Check that full masking raises error
    full_mask = np.zeros(SHAPE_3D)
    full_mask_img = Nifti1Image(full_mask, AFFINE_EYE)
    with pytest.raises(ValueError, match="The mask is invalid as it is empty"):
        apply_mask(data_img, full_mask_img)

    # Check weird values in data
    mask[5, 5, 5] = 2
    with pytest.raises(
        ValueError, match="Background of the mask must be represented with 0"
    ):
        apply_mask(data_img, Nifti1Image(mask, AFFINE_EYE))

    mask[6, 6, 6] = 3
    mask_img = Nifti1Image(mask, AFFINE_EYE)
    with pytest.raises(ValueError, match="Given mask is not made of 2 values"):
        apply_mask(Nifti1Image(data, AFFINE_EYE), mask_img)


def test_unmask_4d():
    rng = np.random.RandomState(42)

    data4D = rng.uniform(size=SHAPE_4D)

    mask = rng.randint(2, size=SHAPE_3D, dtype="int32")
    mask_img = Nifti1Image(mask, AFFINE_EYE)
    mask = mask.astype(bool)

    masked4D = data4D[mask, :].T

    unmasked4D = data4D.copy()
    unmasked4D[np.logical_not(mask), :] = 0

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


@pytest.mark.parametrize("create_files", [False, True])
def test_unmask_3d(create_files):
    """Check both with Nifti1Image and file."""
    rng = np.random.RandomState(42)

    data3D = rng.uniform(size=SHAPE_3D)

    mask = rng.randint(2, size=SHAPE_3D, dtype="int32")
    mask_img = Nifti1Image(mask, AFFINE_EYE)
    mask = mask.astype(bool)

    masked3D = data3D[mask]

    unmasked3D = data3D.copy()
    unmasked3D[np.logical_not(mask)] = 0

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


def test_unmask_errors():
    rng = np.random.RandomState(42)

    mask = rng.randint(2, size=SHAPE_3D, dtype="int32")
    mask_img = Nifti1Image(mask, AFFINE_EYE)
    mask = mask.astype(bool)

    # Error test: shape
    vec_1D = np.empty((500,), dtype=int)
    with pytest.raises(TypeError, match="X must be of shape "):
        unmask(vec_1D, mask_img)
    with pytest.raises(TypeError, match="X must be of shape "):
        unmask([vec_1D], mask_img)

    vec_2D = np.empty((500, 500), dtype=np.float64)
    with pytest.raises(TypeError, match="X must be of shape "):
        unmask(vec_2D, mask_img)
    with pytest.raises(TypeError, match="X must be of shape "):
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


SHAPE_MASK = (4, 4, 1)


def make_mask_a():
    # +---+---+---+---+
    # |   |   |   |   |
    # +---+---+---+---+
    # |   |   |   |   |
    # +---+---+---+---+
    # |   |   | X | X |
    # +---+---+---+---+
    # |   |   | X | X |
    # +---+---+---+---+
    mask_a = np.zeros(SHAPE_MASK, dtype=bool)
    mask_a[2:4, 2:4] = 1
    return mask_a


@pytest.fixture
def mask_a():
    return make_mask_a()


@pytest.fixture
def mask_a_img():
    return Nifti1Image(make_mask_a().astype("int32"), AFFINE_EYE)


def make_mask_b():
    # +---+---+---+---+
    # |   |   |   |   |
    # +---+---+---+---+
    # |   | X | X |   |
    # +---+---+---+---+
    # |   | X | X |   |
    # +---+---+---+---+
    # |   |   |   |   |
    # +---+---+---+---+
    mask_b = np.zeros(SHAPE_MASK, dtype=bool)
    mask_b[1:3, 1:3] = 1
    return mask_b


@pytest.fixture
def mask_b():
    return make_mask_b()


@pytest.fixture
def mask_b_img():
    return Nifti1Image(make_mask_b().astype("int32"), AFFINE_EYE)


@pytest.fixture
def expected_mask_ab():
    expected_mask_ab = np.zeros(SHAPE_MASK, dtype=bool)
    expected_mask_ab[2, 2] = 1
    return expected_mask_ab


def make_mask_c():
    # +---+---+---+---+
    # | X |   | X |   |
    # +---+---+---+---+
    # |   |   | X |   |
    # +---+---+---+---+
    # |   |   | X |   |
    # +---+---+---+---+
    # |   |   | X |   |
    # +---+---+---+---+
    mask_c = np.zeros(SHAPE_MASK, dtype=bool)
    mask_c[:, 2] = 1
    mask_c[0, 0] = 1
    return mask_c


@pytest.fixture
def mask_c():
    return make_mask_c()


@pytest.fixture
def mask_c_img():
    return Nifti1Image(make_mask_c().astype("int32"), AFFINE_EYE)


def make_mask_d():
    mask_d = np.zeros((8, 8, 1), dtype=bool)
    mask_d[2:6, 2:6] = 1
    return mask_d


@pytest.fixture
def mask_d():
    return make_mask_c()


@pytest.fixture
def mask_d_img():
    return Nifti1Image(make_mask_d().astype("uint8"), AFFINE_EYE / 2.0)


def test_intersect_masks_filename(mask_a_img, mask_b_img, expected_mask_ab):
    with write_tmp_imgs(
        mask_a_img, mask_b_img, create_files=True
    ) as filenames:
        mask_ab = intersect_masks(filenames, threshold=1.0)

        assert_array_equal(get_data(mask_ab), expected_mask_ab)


def test_intersect_masks_2_images(
    mask_a_img,
    mask_b_img,
    expected_mask_ab,
):
    """Test the intersect_masks function"""
    mask_ab = intersect_masks([mask_a_img, mask_b_img], threshold=1.0)

    assert_array_equal(get_data(mask_ab), expected_mask_ab)


@pytest.mark.parametrize("threshold", [0.5, 1.0])
def test_intersect_masks_3_images(
    threshold,
    mask_a_img,
    mask_b_img,
    mask_c_img,
    expected_mask_ab,
):
    """Test the intersect_masks function

    default threshold is 0.5
    """
    mask_abc = intersect_masks(
        [mask_a_img, mask_b_img, mask_c_img], threshold=threshold
    )

    expected_mask_abc = expected_mask_ab
    if threshold == 0.5:
        expected_mask_abc[1, 2] = 1
        expected_mask_abc[3, 2] = 1
    assert_array_equal(get_data(mask_abc), expected_mask_abc)


@pytest.mark.parametrize("connected", [True, False])
def test_intersect_masks_3_images_connected(
    mask_a, mask_a_img, mask_b, mask_b_img, mask_c, mask_c_img, connected
):
    """Test the intersect_masks function"""
    mask_abc = intersect_masks(
        [mask_a_img, mask_b_img, mask_c_img],
        threshold=0.0,
        connected=connected,
    )

    expected_mask_abc = mask_a + mask_b + mask_c
    if connected:
        expected_mask_abc[0, 0] = 0
    assert_array_equal(get_data(mask_abc), expected_mask_abc)


def test_intersect_masks_with_f8(mask_a_img, mask_b_img, expected_mask_ab):
    """Test intersect mask images with '>f8'.

    This function uses
    largest_connected_component to check if intersect_masks passes with
    connected=True (which is by default)
    """
    mask_a_img_change_dtype = Nifti1Image(
        get_data(mask_a_img).astype(">f8"), affine=mask_a_img.affine
    )
    mask_b_img_change_dtype = Nifti1Image(
        get_data(mask_b_img).astype(">f8"), affine=mask_b_img.affine
    )
    mask_ab_change_type = intersect_masks(
        [mask_a_img_change_dtype, mask_b_img_change_dtype], threshold=1.0
    )

    assert_array_equal(get_data(mask_ab_change_type), expected_mask_ab)


def test_compute_multi_epi_mask_errors(mask_a_img, mask_d_img):
    # Check that an empty list of images creates a meaningful error
    pytest.raises(TypeError, compute_multi_epi_mask, [])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", MaskWarning)
        with pytest.raises(
            ValueError, match="cannot convert float NaN to integer"
        ):
            compute_multi_epi_mask([mask_a_img, mask_d_img])


def test_compute_multi_epi_mask(mask_a_img, mask_d_img, expected_mask_ab):
    mask_ab = compute_multi_epi_mask(
        [mask_a_img, mask_d_img],
        threshold=1.0,
        opening=0,
        target_affine=AFFINE_EYE,
        target_shape=SHAPE_MASK,
    )

    assert_array_equal(get_data(mask_ab), expected_mask_ab)


def test_compute_multi_brain_mask():
    # Check results are the same if affine is the same
    imgs1 = [
        generate_mni_space_img(res=9, random_state=0)[0],
        generate_mni_space_img(res=9, random_state=1)[0],
    ]
    imgs2 = [
        generate_mni_space_img(res=9, random_state=2)[0],
        generate_mni_space_img(res=9, random_state=3)[0],
    ]

    mask1 = compute_multi_brain_mask(imgs1, threshold=0.2)
    mask2 = compute_multi_brain_mask(imgs2, threshold=0.2)

    assert_array_equal(get_data(mask1), get_data(mask2))


def test_compute_multi_brain_mask_errors():
    with pytest.raises(
        TypeError, match="unsupported format string passed to list"
    ):
        compute_multi_brain_mask([])

    # Check error raised if images with different shapes are given as input
    imgs = [
        generate_mni_space_img(res=8, random_state=0)[0],
        generate_mni_space_img(res=r, random_state=0)[0] for r in (8, 12)
    ]
    with pytest.raises(
        ValueError,
        match="Field of view of image #1 is different from reference FOV",
    ):
        compute_multi_brain_mask(imgs)


def test_unmask_error_shape(random_state=42, shape=SHAPE_4D):
    # open-ended `if .. elif` in masking.unmask
    rng = np.random.RandomState(random_state)

    # setup
    X = rng.standard_normal()
    mask_img = np.zeros(shape, dtype=np.uint8)
    mask_img[rng.standard_normal(size=shape) > 0.4] = 1
    n_features = (mask_img > 0).sum()
    mask_img = Nifti1Image(mask_img, AFFINE_EYE)

    n_samples = shape[0]

    # 3D X (unmask should raise a TypeError)
    X = rng.standard_normal(size=(n_samples, n_features, 2))
    with pytest.raises(TypeError, match=_TEST_DIM_ERROR_MSG % "4D"):
        unmask(X, mask_img)

    # Raises an error because the mask is 4D
    X = rng.standard_normal(size=(n_samples, n_features))
    with pytest.raises(TypeError, match=_TEST_DIM_ERROR_MSG % "4D"):
        unmask(X, mask_img)


def test_nifti_masker_empty_mask_warning():
    X = Nifti1Image(np.ones((2, 2, 2, 5)), AFFINE_EYE)

    with pytest.raises(
        ValueError,
        match="The mask is invalid as it is empty: it masks all data",
    ):
        NiftiMasker(mask_strategy="epi").fit_transform(X)


def test_unmask_list(random_state=42):
    rng = np.random.RandomState(random_state)
    mask_data = rng.uniform(size=SHAPE_3D) < 0.5
    mask_img = Nifti1Image(mask_data.astype(np.uint8), AFFINE_EYE)

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
    extrapolated_data, extrapolated_mask = _extrapolate_out_mask(
        initial_data, initial_mask, iterations=1
    )

    assert_array_equal(extrapolated_data, target_data)
    assert_array_equal(extrapolated_mask, target_mask)


@pytest.mark.parametrize("ndim", range(1, 4))
def test_unmask_from_to_3d_array(ndim, size=5):
    rng = np.random.RandomState(42)

    shape = [size] * ndim
    mask = np.zeros(shape).astype(bool)
    mask[rng.uniform(size=shape) > 0.8] = 1

    support = rng.standard_normal(size=mask.sum())

    full = _unmask_from_to_3d_array(support, mask)

    assert_array_equal(full.shape, shape)
    assert_array_equal(full[mask], support)
