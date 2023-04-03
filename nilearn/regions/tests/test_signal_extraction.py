"""Test for "region" module."""
# Author: Ph. Gervais
# License: simplified BSD

import warnings

import numpy as np
import pytest
from nibabel import Nifti1Image
from nilearn._utils.data_gen import (
    generate_fake_fmri,
    generate_labeled_regions,
    generate_maps,
    generate_timeseries,
)
from nilearn._utils.exceptions import DimensionError
from nilearn._utils.testing import write_tmp_imgs
from nilearn.image import get_data, new_img_like
from nilearn.maskers import NiftiLabelsMasker
from nilearn.regions.signal_extraction import (
    _check_shape_and_affine_compatibility,
    _trim_maps,
    img_to_signals_labels,
    img_to_signals_maps,
    signals_to_img_labels,
    signals_to_img_maps,
)
from numpy.testing import assert_almost_equal, assert_equal

_3D_EXPECTED_ERROR_MSG = (
    "Input data has incompatible dimensionality: "
    "Expected dimension is 3D and you provided "
    "a 4D image"
)

_4D_EXPECTED_ERROR_MSG = (
    "Input data has incompatible dimensionality: "
    "Expected dimension is 4D and you provided "
    "a 3D image"
)

SHAPE_ERROR_MSG = "Images have incompatible shapes."

AFFINE_ERROR_MSG = "Images have different affine matrices."

AFFINE_EYE = np.eye(4)

EPS = np.finfo(np.float64).eps

INF = 1000 * np.finfo(np.float32).eps

N_REGIONS = 8

N_TIMEPOINTS = 17

SHAPE = (8, 9, 10)


def _make_label_data(shape=SHAPE):
    labels_data = np.zeros(shape, dtype="int32")
    h0, h1, h2 = (s // 2 for s in shape)
    labels_data[:h0, :h1, :h2] = 1
    labels_data[:h0, :h1, h2:] = 2
    labels_data[:h0, h1:, :h2] = 3
    labels_data[:h0, h1:, h2:] = 4
    labels_data[h0:, :h1, :h2] = 5
    labels_data[h0:, :h1, h2:] = 6
    labels_data[h0:, h1:, :h2] = 7
    labels_data[h0:, h1:, h2:] = 8
    return labels_data


def _create_mask_with_3_regions_from_labels_data(labels_data, affine):
    """Create a mask containing only 3 regions."""
    mask_data = (labels_data == 1) + (labels_data == 2) + (labels_data == 5)
    return Nifti1Image(mask_data.astype(np.int8), affine)


@pytest.fixture
def labels_data():
    return _make_label_data()


@pytest.fixture
def labels_img():
    return Nifti1Image(_make_label_data(SHAPE), AFFINE_EYE)


@pytest.fixture
def mask_img():
    mask_data = np.zeros(SHAPE)
    mask_data[1:-1, 1:-1, 1:-1] = 1
    return Nifti1Image(mask_data, AFFINE_EYE)


@pytest.fixture
def signals():
    return generate_timeseries(n_timepoints=N_TIMEPOINTS, n_features=N_REGIONS)


@pytest.fixture
def img_3D():
    return Nifti1Image(np.zeros(SHAPE), AFFINE_EYE)


@pytest.fixture
def img_4D():
    shape = SHAPE + (7,)
    return Nifti1Image(np.zeros(shape), AFFINE_EYE)


@pytest.fixture
def fmri_img():
    return generate_fake_fmri(shape=SHAPE, affine=AFFINE_EYE)[0]


@pytest.fixture
def labeled_regions():
    labels = list(range(N_REGIONS + 1))  # 0 is background
    return generate_labeled_regions(
        shape=SHAPE, n_regions=N_REGIONS, labels=labels
    )


def _all_voxel_of_each_region_have_same_values(
    data, labels_data, n_regions, signals
):
    for n in range(1, n_regions + 1):
        sigs = data[labels_data == n, :]
        assert_almost_equal(sigs[0, :], signals[:, n - 1])
        assert abs(sigs - sigs[0, :]).max() < EPS


def test_check_shape_and_affine_compatibility_without_dim(img_3D):
    """Ensure correct behaviour for valid data without dim"""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _check_shape_and_affine_compatibility(img1=img_3D, img2=img_3D)


def test_check_shape_and_affine_compatibility_with_dim(img_3D, img_4D):
    """Ensure correct behaviour for valid data without dim"""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _check_shape_and_affine_compatibility(img1=img_4D, img2=img_3D, dim=3)


@pytest.mark.parametrize(
    "test_shape, test_affine, msg",
    [
        ((8, 9, 11), AFFINE_EYE, SHAPE_ERROR_MSG),
        (SHAPE, 2 * AFFINE_EYE, AFFINE_ERROR_MSG),
    ],
)
def test_check_shape_and_affine_compatibility_error(
    img_3D, test_shape, test_affine, msg
):
    img2 = Nifti1Image(np.zeros(test_shape), test_affine)

    with pytest.raises(ValueError, match=msg):
        _check_shape_and_affine_compatibility(img1=img_3D, img2=img2)


def test_errors_3D(img_3D):
    """Verify that 3D images are refused."""
    wrong_dim_image = img_3D

    with pytest.raises(DimensionError, match=_4D_EXPECTED_ERROR_MSG):
        img_to_signals_labels(imgs=wrong_dim_image, labels_img=img_3D)

    with pytest.raises(DimensionError, match=_4D_EXPECTED_ERROR_MSG):
        img_to_signals_maps(imgs=img_4D, maps_img=wrong_dim_image)


def test_errors_4D_labels(img_4D):
    """Verify that 4D images are refused."""
    wrong_dim_label_img = img_4D

    with pytest.raises(DimensionError, match=_3D_EXPECTED_ERROR_MSG):
        img_to_signals_labels(imgs=img_4D, labels_img=wrong_dim_label_img)

    with pytest.raises(DimensionError, match=_3D_EXPECTED_ERROR_MSG):
        signals_to_img_labels(signals=img_4D, labels_img=wrong_dim_label_img)


def test_errors_4D_masks(img_3D, img_4D):
    """Verify that 4D images are refused."""
    wrong_dim_mask_img = img_4D

    with pytest.raises(DimensionError, match=_3D_EXPECTED_ERROR_MSG):
        img_to_signals_labels(
            imgs=img_4D, labels_img=img_3D, mask_img=wrong_dim_mask_img
        )

    with pytest.raises(DimensionError, match=_3D_EXPECTED_ERROR_MSG):
        signals_to_img_labels(
            signals=img_4D, labels_img=img_3D, mask_img=wrong_dim_mask_img
        )

    with pytest.raises(DimensionError, match=_3D_EXPECTED_ERROR_MSG):
        img_to_signals_maps(
            imgs=img_4D, maps_img=img_4D, mask_img=wrong_dim_mask_img
        )


@pytest.mark.parametrize(
    "shape, affine, error_msg",
    [
        ((8, 9, 11), AFFINE_EYE, SHAPE_ERROR_MSG),
        (SHAPE, 2 * AFFINE_EYE, AFFINE_ERROR_MSG),
    ],
)
def test_img_to_signals_labels_bad_labels_input(
    img_4D, shape, affine, error_msg
):
    bad_img = Nifti1Image(np.zeros(shape), affine)

    with pytest.raises(ValueError, match=error_msg):
        img_to_signals_labels(imgs=img_4D, labels_img=bad_img)


@pytest.mark.parametrize(
    "shape, affine, error_msg",
    [
        ((8, 9, 11), AFFINE_EYE, SHAPE_ERROR_MSG),
        (SHAPE, 2 * AFFINE_EYE, AFFINE_ERROR_MSG),
    ],
)
def test_img_to_signals_labels_bad_mask_input(
    img_4D, img_3D, shape, affine, error_msg
):
    bad_img = Nifti1Image(np.zeros(shape), affine)

    with pytest.raises(ValueError, match=error_msg):
        img_to_signals_labels(imgs=img_4D, labels_img=img_3D, mask_img=bad_img)


def test_img_to_signals_labels_error_strategy(img_4D, img_3D):
    with pytest.raises(ValueError, match="Invalid strategy"):
        img_to_signals_labels(imgs=img_4D, labels_img=img_3D, strategy="foo")


@pytest.mark.parametrize(
    "shape, affine, error_msg",
    [
        ((8, 9, 11), AFFINE_EYE, SHAPE_ERROR_MSG),
        (SHAPE, 2 * AFFINE_EYE, AFFINE_ERROR_MSG),
    ],
)
def test_signals_to_img_labels_bad_label_input(
    img_4D, img_3D, shape, affine, error_msg
):
    bad_img = Nifti1Image(np.zeros(shape), affine)

    with pytest.raises(ValueError, match=error_msg):
        signals_to_img_labels(
            signals=img_4D, labels_img=bad_img, mask_img=img_3D
        )


@pytest.mark.parametrize(
    "shape, affine, error_msg",
    [
        ((8, 9, 11), AFFINE_EYE, SHAPE_ERROR_MSG),
        (SHAPE, 2 * AFFINE_EYE, AFFINE_ERROR_MSG),
    ],
)
def test_signals_to_img_labels_bad_mask_input(
    img_4D, img_3D, shape, affine, error_msg
):
    bad_img = Nifti1Image(np.zeros(shape), affine)

    with pytest.raises(ValueError, match=error_msg):
        signals_to_img_labels(
            signals=img_4D, labels_img=img_3D, mask_img=bad_img
        )


@pytest.mark.parametrize(
    "shape, affine, error_msg",
    [
        ((8, 9, 11, 7), AFFINE_EYE, SHAPE_ERROR_MSG),
        (SHAPE + (7,), 2 * AFFINE_EYE, AFFINE_ERROR_MSG),
    ],
)
def test_img_to_signals_maps_bad_maps(img_4D, shape, affine, error_msg):
    bad_img = Nifti1Image(np.zeros(shape), affine)

    with pytest.raises(ValueError, match=error_msg):
        img_to_signals_maps(
            imgs=img_4D,
            maps_img=bad_img,
        )


@pytest.mark.parametrize(
    "shape, affine, error_msg",
    [
        ((8, 9, 11), AFFINE_EYE, SHAPE_ERROR_MSG),
        (SHAPE, 2 * AFFINE_EYE, AFFINE_ERROR_MSG),
    ],
)
def test_img_to_signals_maps_bad_masks(img_4D, shape, affine, error_msg):
    bad_img = Nifti1Image(np.zeros(shape), affine)

    with pytest.raises(ValueError, match=error_msg):
        img_to_signals_maps(imgs=img_4D, maps_img=img_4D, mask_img=bad_img)


def test_signals_extraction_with_labels_without_mask(
    signals, labels_data, labels_img
):
    """Test conversion between signals and images \
    using regions defined by labels."""
    data_img = signals_to_img_labels(signals=signals, labels_img=labels_img)

    assert data_img.shape == (SHAPE + (N_TIMEPOINTS,))
    data = get_data(data_img)
    assert np.all(data.std(axis=-1) > 0)
    # There must be non-zero data (safety net)
    assert abs(data).max() > 1e-9

    _all_voxel_of_each_region_have_same_values(
        data, labels_data, N_REGIONS, signals
    )

    # and back
    signals_r, labels_r = img_to_signals_labels(
        imgs=data_img, labels_img=labels_img
    )

    assert_almost_equal(signals_r, signals)
    assert labels_r == list(range(1, 9))

    with write_tmp_imgs(data_img) as filenames:
        signals_r, labels_r = img_to_signals_labels(
            imgs=filenames, labels_img=labels_img
        )

        assert_almost_equal(signals_r, signals)
        assert labels_r == list(range(1, 9))


def test_signals_extraction_with_labels_with_mask(
    signals, labels_img, labels_data, mask_img
):
    """Test conversion between signals and images \
    using regions defined by labels with a mask."""
    data_img = signals_to_img_labels(
        signals=signals, labels_img=labels_img, mask_img=mask_img
    )

    assert data_img.shape == (SHAPE + (N_TIMEPOINTS,))
    # There must be non-zero data (safety net)
    data = get_data(data_img)
    assert abs(data).max() > 1e-9

    # Zero outside of the mask
    assert np.all(data[np.logical_not(get_data(mask_img))].std(axis=-1) < EPS)

    with write_tmp_imgs(labels_img, mask_img) as filenames:
        data_img = signals_to_img_labels(
            signals=signals, labels_img=filenames[0], mask_img=filenames[1]
        )

        assert data_img.shape == (SHAPE + (N_TIMEPOINTS,))
        data = get_data(data_img)
        assert abs(data).max() > 1e-9
        # Zero outside of the mask
        assert np.all(
            data[np.logical_not(get_data(mask_img))].std(axis=-1) < EPS
        )

    # mask labels before checking
    masked_labels_data = labels_data.copy()
    masked_labels_data[np.logical_not(get_data(mask_img))] = 0
    _all_voxel_of_each_region_have_same_values(
        data, masked_labels_data, N_REGIONS, signals
    )

    # and back
    signals_r, labels_r = img_to_signals_labels(
        imgs=data_img, labels_img=labels_img, mask_img=mask_img
    )

    assert_almost_equal(signals_r, signals)
    assert labels_r == list(range(1, 9))


def test_signal_extraction_with_maps():
    # Generate signal imgs
    rng = np.random.RandomState(42)
    maps_img, mask_img = generate_maps(SHAPE, N_REGIONS)
    maps_data = get_data(maps_img)
    data = np.zeros(SHAPE + (N_TIMEPOINTS,))
    signals = np.zeros((N_TIMEPOINTS, maps_data.shape[-1]))
    for n in range(maps_data.shape[-1]):
        signals[:, n] = rng.standard_normal(size=N_TIMEPOINTS)
        data[maps_data[..., n] > 0, :] = signals[:, n]
    imgs = Nifti1Image(data, AFFINE_EYE)

    # Get signals
    signals_r, _ = img_to_signals_maps(
        imgs=imgs, maps_img=maps_img, mask_img=mask_img
    )
    assert_almost_equal(signals, signals_r)

    # Recover image
    img_r = signals_to_img_maps(signals, maps_img, mask_img=mask_img)
    assert_almost_equal(get_data(img_r), get_data(imgs))

    # Same thing without mask
    signals_r, _ = img_to_signals_maps(imgs, maps_img)
    assert_almost_equal(signals, signals_r)
    img_r = signals_to_img_maps(signals, maps_img)
    assert_almost_equal(get_data(img_r), get_data(imgs))


def test_signal_extraction_with_maps_and_labels(labeled_regions, fmri_img):
    labels = list(range(N_REGIONS + 1))
    labels_data = get_data(labeled_regions)
    # Convert to maps
    maps_data = np.zeros(SHAPE + (N_REGIONS,))
    for n, l in enumerate(labels):
        if n == 0:
            continue
        maps_data[labels_data == l, n - 1] = 1

    maps_img = Nifti1Image(maps_data, labeled_regions.affine)

    # Extract signals from maps and labels: results must be identical.
    maps_signals, maps_labels = img_to_signals_maps(fmri_img, maps_img)
    labels_signals, labels_labels = img_to_signals_labels(
        imgs=fmri_img, labels_img=labeled_regions
    )
    assert_almost_equal(maps_signals, labels_signals)

    # Same thing with a mask, containing only 3 regions.
    mask_img = _create_mask_with_3_regions_from_labels_data(
        labels_data, labeled_regions.affine
    )
    labels_signals, labels_labels = img_to_signals_labels(
        imgs=fmri_img, labels_img=labeled_regions, mask_img=mask_img
    )
    maps_signals, maps_labels = img_to_signals_maps(
        fmri_img, maps_img, mask_img=mask_img
    )

    assert_almost_equal(maps_signals, labels_signals)
    assert maps_signals.shape[1] == N_REGIONS
    assert maps_labels == list(range(len(maps_labels)))
    assert labels_signals.shape == (N_TIMEPOINTS, N_REGIONS)
    assert labels_labels == labels[1:]

    # Inverse operation (mostly smoke test)
    labels_img_r = signals_to_img_labels(
        labels_signals, labeled_regions, mask_img=mask_img
    )
    assert labels_img_r.shape == SHAPE + (N_TIMEPOINTS,)

    maps_img_r = signals_to_img_maps(maps_signals, maps_img, mask_img=mask_img)
    assert maps_img_r.shape == SHAPE + (N_TIMEPOINTS,)


def test_signal_extraction_nans_in_regions_are_replaced_with_zeros():
    shape = (4, 5, 6)
    labels = list(range(N_REGIONS + 1))  # 0 is background
    labels_img = generate_labeled_regions(shape, N_REGIONS, labels=labels)
    labels_data = get_data(labels_img)
    fmri_img, _ = generate_fake_fmri(
        shape=shape, affine=labels_img.affine, length=N_TIMEPOINTS
    )

    mask_img = _create_mask_with_3_regions_from_labels_data(
        labels_data, labels_img.affine
    )

    region1 = labels_data == 2
    indices = tuple(ind[:1] for ind in np.where(region1))
    get_data(fmri_img)[indices] = np.nan

    labels_signals, labels_labels = img_to_signals_labels(
        imgs=fmri_img, labels_img=labels_img, mask_img=mask_img
    )

    assert np.all(labels_signals[:, labels_labels.index(2)] == 0.0)


def test_trim_maps():
    # maps
    maps_data = np.zeros(SHAPE + (N_REGIONS,), dtype=np.float32)
    h0, h1, h2 = (s // 2 for s in SHAPE)
    maps_data[:h0, :h1, :h2, 0] = 1
    maps_data[:h0, :h1, h2:, 1] = 1.1
    maps_data[:h0, h1:, :h2, 2] = 1
    maps_data[:h0, h1:, h2:, 3] = 0.5
    maps_data[h0:, :h1, :h2, 4] = 1
    maps_data[h0:, :h1, h2:, 5] = 1.4
    maps_data[h0:, h1:, :h2, 6] = 1
    maps_data[h0:, h1:, h2:, 7] = 1

    # mask intersecting all regions
    mask_data = np.zeros(SHAPE, dtype=np.int8)
    mask_data[1:-1, 1:-1, 1:-1] = 1

    maps_i, maps_i_mask, maps_i_indices = _trim_maps(maps_data, mask_data)

    assert maps_i.flags["F_CONTIGUOUS"]
    assert len(maps_i_indices) == maps_i.shape[-1]
    assert maps_i.shape == maps_data.shape
    maps_i_correct = maps_data.copy()
    maps_i_correct[np.logical_not(mask_data), :] = 0
    assert_almost_equal(maps_i_correct, maps_i)
    assert_equal(mask_data, maps_i_mask)
    assert_equal(np.asarray(list(range(8))), maps_i_indices)

    # mask intersecting half of the regions
    mask_data = np.zeros(SHAPE, dtype=np.int8)
    mask_data[1:2, 1:-1, 1:-1] = 1
    maps_data[1, 1, 1, 0] = 0  # remove one point inside mask

    maps_i, maps_i_mask, maps_i_indices = _trim_maps(maps_data, mask_data)

    assert maps_i.flags["F_CONTIGUOUS"]
    assert len(maps_i_indices) == maps_i.shape[-1]
    assert maps_i.shape == (maps_data.shape[:3] + (4,))
    maps_i_correct = maps_data[..., :4].copy()
    maps_i_correct[np.logical_not(mask_data), :] = 0
    assert_almost_equal(maps_i_correct, maps_i)
    mask_data[1, 1, 1] = 0  # for test to succeed
    assert_equal(mask_data, maps_i_mask)
    mask_data[1, 1, 1] = 1  # reset, just in case.
    assert_equal(np.asarray(list(range(4))), maps_i_indices)


@pytest.mark.parametrize(
    "target_dtype",
    (float, np.float32, np.float64, int, np.uint),
)
def test_img_to_signals_labels_non_float_type(target_dtype):
    rng = np.random.RandomState(42)

    fake_fmri_data = rng.uniform(size=(10, 10, 10, N_TIMEPOINTS)) > 0.5
    fake_affine = np.eye(4, 4).astype(np.float64)
    fake_fmri_img_orig = Nifti1Image(
        fake_fmri_data.astype(np.float64), fake_affine
    )
    fake_fmri_img_target_dtype = new_img_like(
        fake_fmri_img_orig, fake_fmri_data.astype(target_dtype)
    )
    fake_mask_data = np.ones((10, 10, 10), dtype=np.uint8)
    fake_mask = Nifti1Image(fake_mask_data, fake_affine)

    masker = NiftiLabelsMasker(fake_mask)
    masker.fit()
    timeseries_int = masker.transform(fake_fmri_img_target_dtype)
    timeseries_float = masker.transform(fake_fmri_img_orig)
    assert np.sum(timeseries_int) != 0
    assert np.allclose(timeseries_int, timeseries_float)
