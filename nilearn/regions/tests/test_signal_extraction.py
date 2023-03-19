"""Test for "region" module."""
# Author: Ph. Gervais
# License: simplified BSD

import nibabel
import numpy as np
import pytest
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

_TEST_DIM_ERROR_MSG = (
    "Input data has incompatible dimensionality: "
    "Expected dimension is 3D and you provided "
    "a 4D image"
)
_TEST_SHAPE_ERROR_MSG = "Images have incompatible shapes."
_TEST_AFFINE_ERROR_MSG = "Images have different affine matrices."

INF = 1000 * np.finfo(np.float32).eps
EPS = np.finfo(np.float64).eps


AFFINE = np.eye(4)

SHAPE = (8, 9, 10)

N_TIMEPOINTS = 17

N_REGIONS = 8


def _make_label_data(shape=SHAPE):
    labels_data = np.zeros(shape, dtype="int32")
    h0 = shape[0] // 2
    h1 = shape[1] // 2
    h2 = shape[2] // 2
    labels_data[:h0, :h1, :h2] = 1
    labels_data[:h0, :h1, h2:] = 2
    labels_data[:h0, h1:, :h2] = 3
    labels_data[:h0, h1:, h2:] = 4
    labels_data[h0:, :h1, :h2] = 5
    labels_data[h0:, :h1, h2:] = 6
    labels_data[h0:, h1:, :h2] = 7
    labels_data[h0:, h1:, h2:] = 8
    return labels_data


@pytest.fixture
def labels_data():
    return _make_label_data()


@pytest.fixture
def labels_img():
    labels_data = _make_label_data(SHAPE)
    return nibabel.Nifti1Image(labels_data, AFFINE)


@pytest.fixture
def mask_img():
    mask_data = np.zeros(SHAPE)
    mask_data[1:-1, 1:-1, 1:-1] = 1
    return nibabel.Nifti1Image(mask_data, AFFINE)


@pytest.fixture
def signals():
    return generate_timeseries(n_timepoints=N_TIMEPOINTS, n_features=N_REGIONS)


@pytest.fixture
def empty_img_3D():
    shape = (2, 3, 4)
    return nibabel.Nifti1Image(np.zeros(shape), AFFINE)


@pytest.fixture
def empty_img_4D():
    shape = (2, 3, 4, 7)
    return nibabel.Nifti1Image(np.zeros(shape), AFFINE)


@pytest.fixture
def fmri_img():
    fmri_img, _ = generate_fake_fmri(shape=SHAPE, affine=AFFINE)
    return fmri_img


@pytest.fixture
def labeled_regions():
    labels = list(range(N_REGIONS + 1))  # 0 is background
    return generate_labeled_regions(
        shape=SHAPE, n_regions=N_REGIONS, labels=labels
    )


def _make_signal_extraction_test_data(shape, n_regions, n_timepoints):
    # Generate labels
    labels = list(range(n_regions + 1))  # 0 is background
    labels_img = generate_labeled_regions(shape, n_regions, labels=labels)
    labels_data = get_data(labels_img)
    # Convert to maps
    maps_data = np.zeros(shape + (n_regions,))
    for n, l in enumerate(labels):
        if n == 0:
            continue

        maps_data[labels_data == l, n - 1] = 1

    maps_img = nibabel.Nifti1Image(maps_data, labels_img.affine)

    # Generate fake data
    fmri_img, _ = generate_fake_fmri(
        shape=shape, affine=labels_img.affine, length=n_timepoints
    )

    return fmri_img, labels, labels_data, labels_img, maps_img


def _all_voxel_of_each_region_have_same_values(
    data, labels_data, n_regions, signals
):
    for n in range(1, n_regions + 1):
        sigs = data[labels_data == n, :]
        np.testing.assert_almost_equal(sigs[0, :], signals[:, n - 1])
        assert abs(sigs - sigs[0, :]).max() < EPS


def test_check_shape_and_affine_compatibility_without_dim(empty_img_3D):
    """Ensure correct behaviour for valid data without dim"""
    with pytest.warns(None):
        _check_shape_and_affine_compatibility(
            img1=empty_img_3D, img2=empty_img_3D
        )


def test_check_shape_and_affine_compatibility_with_dim(
    empty_img_3D, empty_img_4D
):
    """Ensure correct behaviour for valid data without dim"""
    with pytest.warns(None):
        _check_shape_and_affine_compatibility(
            img1=empty_img_4D, img2=empty_img_3D, dim=3
        )


@pytest.mark.parametrize(
    "test_shape, test_affine, msg",
    [
        ((2, 3, 5), np.eye(4), _TEST_SHAPE_ERROR_MSG),
        ((2, 3, 4), 2 * np.eye(4), _TEST_AFFINE_ERROR_MSG),
    ],
)
def test_check_shape_and_affine_compatibility_error(
    empty_img_3D, test_shape, test_affine, msg
):
    img2 = nibabel.Nifti1Image(np.zeros(test_shape), test_affine)

    with pytest.raises(ValueError, match=msg):
        _check_shape_and_affine_compatibility(img1=empty_img_3D, img2=img2)


def test_signals_extraction_with_labels_without_mask(
    signals, labels_data, labels_img
):
    """Test conversion between signals and images \
    using regions defined by labels."""
    shape = SHAPE
    n_timepoints = N_TIMEPOINTS
    n_regions = N_REGIONS  # must be 8

    # from labels
    data_img = signals_to_img_labels(signals=signals, labels_img=labels_img)

    assert data_img.shape == (shape + (n_timepoints,))
    data = get_data(data_img)
    assert np.all(data.std(axis=-1) > 0)
    # There must be non-zero data (safety net)
    assert abs(data).max() > 1e-9

    _all_voxel_of_each_region_have_same_values(
        data, labels_data, n_regions, signals
    )

    # and back
    signals_r, labels_r = img_to_signals_labels(
        imgs=data_img, labels_img=labels_img
    )

    np.testing.assert_almost_equal(signals_r, signals)
    assert labels_r == list(range(1, 9))

    with write_tmp_imgs(data_img) as fname_img:
        signals_r, labels_r = img_to_signals_labels(
            imgs=fname_img, labels_img=labels_img
        )

        np.testing.assert_almost_equal(signals_r, signals)
        assert labels_r == list(range(1, 9))


def test_signals_extraction_with_labels_with_mask(
    signals, labels_img, labels_data, mask_img
):
    """Test conversion between signals and images \
    using regions defined by labels with a mask."""
    shape = SHAPE
    n_timepoints = N_TIMEPOINTS
    n_regions = N_REGIONS  # must be 8

    data_img = signals_to_img_labels(
        signals=signals, labels_img=labels_img, mask_img=mask_img
    )

    assert data_img.shape == (shape + (n_timepoints,))
    # There must be non-zero data (safety net)
    data = get_data(data_img)
    assert abs(data).max() > 1e-9

    # Zero outside of the mask
    assert np.all(data[np.logical_not(get_data(mask_img))].std(axis=-1) < EPS)

    with write_tmp_imgs(labels_img, mask_img) as filenames:
        data_img = signals_to_img_labels(
            signals=signals, labels_img=filenames[0], mask_img=filenames[1]
        )

        assert data_img.shape == (shape + (n_timepoints,))
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
        data, masked_labels_data, n_regions, signals
    )

    # and back
    signals_r, labels_r = img_to_signals_labels(
        imgs=data_img, labels_img=labels_img, mask_img=mask_img
    )

    np.testing.assert_almost_equal(signals_r, signals)
    assert labels_r == list(range(1, 9))


def test_signals_extraction_with_labels_errors(
    signals, labels_img, labels_data
):
    data_img = signals_to_img_labels(signals=signals, labels_img=labels_img)

    # should raise no error
    img_to_signals_labels(imgs=data_img, labels_img=labels_img)

    # verify that 4D label images are refused
    labels_4d_data = np.zeros((SHAPE) + (2,))
    labels_4d_data[..., 0] = labels_data
    labels_4d_data[..., 1] = labels_data
    labels_4d_img = nibabel.Nifti1Image(labels_4d_data, np.eye(4))
    with pytest.raises(DimensionError, match=_TEST_DIM_ERROR_MSG):
        img_to_signals_labels(imgs=data_img, labels_img=labels_4d_img)

    mask_4d_img = nibabel.Nifti1Image(np.ones(SHAPE + (2,)), AFFINE)
    with pytest.raises(DimensionError, match=_TEST_DIM_ERROR_MSG):
        img_to_signals_labels(
            imgs=data_img, labels_img=labels_img, mask_img=mask_4d_img
        )


def test_signals_extraction_with_labels_with_mask_errors(
    signals, labels_img, labels_data, mask_img
):
    data_img = signals_to_img_labels(
        signals=signals, labels_img=labels_img, mask_img=mask_img
    )

    # should raise no error
    img_to_signals_labels(imgs=data_img, labels_img=labels_img)

    # verify that 4D label images are refused
    labels_4d_data = np.zeros((SHAPE) + (2,))
    labels_4d_data[..., 0] = labels_data
    labels_4d_data[..., 1] = labels_data
    labels_4d_img = nibabel.Nifti1Image(labels_4d_data, np.eye(4))

    mask_4d_img = nibabel.Nifti1Image(np.ones(SHAPE + (2,)), AFFINE)

    with pytest.raises(DimensionError, match=_TEST_DIM_ERROR_MSG):
        signals_to_img_labels(
            signals=data_img, labels_img=labels_img, mask_img=mask_4d_img
        )

    with pytest.raises(TypeError):
        signals_to_img_labels(
            signals=data_img, labels_img=labels_4d_img, mask_img=mask_img
        )


def test_signal_extraction_with_maps():
    shape = SHAPE
    n_regions = N_REGIONS
    n_timepoints = N_TIMEPOINTS

    # Generate signals
    rng = np.random.RandomState(42)

    maps_img, mask_img = generate_maps(shape, n_regions, border=1)
    maps_data = get_data(maps_img)
    data = np.zeros(shape + (n_timepoints,), dtype=np.float32)

    mask_4d_img = nibabel.Nifti1Image(np.ones(shape + (2,)), np.eye(4))

    signals = np.zeros((n_timepoints, maps_data.shape[-1]))
    for n in range(maps_data.shape[-1]):
        signals[:, n] = rng.standard_normal(size=n_timepoints)
        data[maps_data[..., n] > 0, :] = signals[:, n]
    img = nibabel.Nifti1Image(data, np.eye(4))

    # verify that 4d masks are refused
    with pytest.raises(TypeError, match=_TEST_DIM_ERROR_MSG):
        img_to_signals_maps(img, maps_img, mask_img=mask_4d_img)

    # Get signals
    signals_r, _ = img_to_signals_maps(img, maps_img, mask_img=mask_img)

    # The output must be identical to the input signals, because every region
    # is homogeneous:
    # there is the same signal in all voxels of one given region
    # (and all maps are uniform).
    np.testing.assert_almost_equal(signals, signals_r)

    # Same thing without mask (in that case)
    signals_r, _ = img_to_signals_maps(img, maps_img)
    np.testing.assert_almost_equal(signals, signals_r)

    # Recover image
    img_r = signals_to_img_maps(signals, maps_img, mask_img=mask_img)
    np.testing.assert_almost_equal(get_data(img_r), get_data(img))
    img_r = signals_to_img_maps(signals, maps_img)
    np.testing.assert_almost_equal(get_data(img_r), get_data(img))


@pytest.mark.parametrize("z_dim, affine_diag", [(5, 1), (4, 2)])
def test_input_validation_bad_masks(empty_img_4D, z_dim, affine_diag):
    data_img = empty_img_4D

    bad_mask_img = nibabel.Nifti1Image(
        np.zeros((2, 3, z_dim)), affine_diag * np.eye(4)
    )

    labels_img = nibabel.Nifti1Image(np.zeros((2, 3, 4)), np.eye(4))
    pytest.raises(
        ValueError,
        img_to_signals_labels,
        empty_img_4D,
        labels_img,
        mask_img=bad_mask_img,
    )

    maps_img = empty_img_4D
    pytest.raises(
        ValueError,
        img_to_signals_maps,
        data_img,
        maps_img,
        mask_img=bad_mask_img,
    )


@pytest.mark.parametrize("z_dim, affine_diag", [(5, 1), (4, 2)])
def test_input_validation_bad_labels(
    empty_img_4D, empty_img_3D, z_dim, affine_diag
):
    data_img = empty_img_4D

    bad_labels_img = nibabel.Nifti1Image(
        np.zeros((2, 3, z_dim)), affine_diag * np.eye(4)
    )

    pytest.raises(
        ValueError,
        img_to_signals_labels,
        data_img,
        bad_labels_img,
    )

    good_mask_img = empty_img_3D
    pytest.raises(
        ValueError,
        img_to_signals_labels,
        data_img,
        bad_labels_img,
        mask_img=good_mask_img,
    )


@pytest.mark.parametrize("z_dim, affine_diag", [(5, 1), (4, 2)])
def test_input_validation_bad_maps(
    empty_img_4D, empty_img_3D, z_dim, affine_diag
):
    data_img = empty_img_4D

    bad_maps_img = nibabel.Nifti1Image(
        np.zeros((2, 3, z_dim, 7)), affine_diag * np.eye(4)
    )

    pytest.raises(
        ValueError,
        img_to_signals_maps,
        data_img,
        bad_maps_img,
    )

    good_mask_img = empty_img_3D
    pytest.raises(
        ValueError,
        img_to_signals_maps,
        data_img,
        bad_maps_img,
        mask_img=good_mask_img,
    )


def test_signal_extraction_with_labels_error_strategy():
    shape = SHAPE
    n_regions = N_REGIONS
    labels = list(range(n_regions + 1))  # 0 is background
    labels_img = generate_labeled_regions(shape, n_regions, labels=labels)

    fmri_img, _ = generate_fake_fmri(shape=shape, affine=labels_img.affine)

    with pytest.raises(ValueError, match="Invalid strategy"):
        img_to_signals_labels(
            imgs=fmri_img, labels_img=labels_img, strategy="foo"
        )


def test_signal_extraction_with_maps_and_labels(labeled_regions, fmri_img):
    shape = SHAPE
    n_regions = N_REGIONS
    n_timepoints = N_TIMEPOINTS

    (
        _,
        labels,
        labels_data,
        _,
        maps_img,
    ) = _make_signal_extraction_test_data(shape, n_regions, n_timepoints)

    # Extract signals from maps and labels: results must be identical.
    maps_signals, maps_labels = img_to_signals_maps(fmri_img, maps_img)
    labels_signals, labels_labels = img_to_signals_labels(
        imgs=fmri_img, labels_img=labeled_regions
    )
    np.testing.assert_almost_equal(maps_signals, labels_signals)

    # Same thing with a mask, containing only 3 regions.
    mask_data = (labels_data == 1) + (labels_data == 2) + (labels_data == 5)
    mask_img = nibabel.Nifti1Image(
        mask_data.astype(np.int8), labeled_regions.affine
    )
    labels_signals, labels_labels = img_to_signals_labels(
        imgs=fmri_img, labels_img=labeled_regions, mask_img=mask_img
    )

    maps_signals, maps_labels = img_to_signals_maps(
        fmri_img, maps_img, mask_img=mask_img
    )

    np.testing.assert_almost_equal(maps_signals, labels_signals)
    assert maps_signals.shape[1] == n_regions
    assert maps_labels == list(range(len(maps_labels)))
    assert labels_signals.shape == (n_timepoints, n_regions)
    assert labels_labels == labels[1:]

    # Inverse operation (mostly smoke test)
    labels_img_r = signals_to_img_labels(
        labels_signals, labeled_regions, mask_img=mask_img
    )
    assert labels_img_r.shape == shape + (n_timepoints,)

    maps_img_r = signals_to_img_maps(maps_signals, maps_img, mask_img=mask_img)
    assert maps_img_r.shape == shape + (n_timepoints,)


def test_signal_extraction_nans_in_regions_are_replaced_with_zeros():
    shape = (4, 5, 6)
    n_regions = N_REGIONS
    n_timepoints = N_TIMEPOINTS

    (
        fmri_img,
        _,
        labels_data,
        labels_img,
        _,
    ) = _make_signal_extraction_test_data(shape, n_regions, n_timepoints)

    mask_data = (labels_data == 1) + (labels_data == 2) + (labels_data == 5)
    mask_img = nibabel.Nifti1Image(
        mask_data.astype(np.int8), labels_img.affine
    )

    region1 = labels_data == 2
    indices = tuple(ind[:1] for ind in np.where(region1))
    get_data(fmri_img)[indices] = np.nan

    labels_signals, labels_labels = img_to_signals_labels(
        imgs=fmri_img, labels_img=labels_img, mask_img=mask_img
    )

    assert np.all(labels_signals[:, labels_labels.index(2)] == 0.0)


def test_trim_maps():
    shape = SHAPE
    n_regions = N_REGIONS

    # maps
    maps_data = np.zeros(shape + (n_regions,), dtype=np.float32)
    h0 = shape[0] // 2
    h1 = shape[1] // 2
    h2 = shape[2] // 2
    maps_data[:h0, :h1, :h2, 0] = 1
    maps_data[:h0, :h1, h2:, 1] = 1.1
    maps_data[:h0, h1:, :h2, 2] = 1
    maps_data[:h0, h1:, h2:, 3] = 0.5
    maps_data[h0:, :h1, :h2, 4] = 1
    maps_data[h0:, :h1, h2:, 5] = 1.4
    maps_data[h0:, h1:, :h2, 6] = 1
    maps_data[h0:, h1:, h2:, 7] = 1

    # mask intersecting all regions
    mask_data = np.zeros(shape, dtype=np.int8)
    mask_data[1:-1, 1:-1, 1:-1] = 1

    maps_i, maps_i_mask, maps_i_indices = _trim_maps(maps_data, mask_data)
    assert maps_i.flags["F_CONTIGUOUS"]
    assert len(maps_i_indices) == maps_i.shape[-1]
    assert maps_i.shape == maps_data.shape
    maps_i_correct = maps_data.copy()
    maps_i_correct[np.logical_not(mask_data), :] = 0
    np.testing.assert_almost_equal(maps_i_correct, maps_i)
    np.testing.assert_equal(mask_data, maps_i_mask)
    np.testing.assert_equal(np.asarray(list(range(8))), maps_i_indices)

    # mask intersecting half of the regions
    mask_data = np.zeros(shape, dtype=np.int8)
    mask_data[1:2, 1:-1, 1:-1] = 1
    maps_data[1, 1, 1, 0] = 0  # remove one point inside mask

    maps_i, maps_i_mask, maps_i_indices = _trim_maps(maps_data, mask_data)
    assert maps_i.flags["F_CONTIGUOUS"]
    assert len(maps_i_indices) == maps_i.shape[-1]
    assert maps_i.shape == (maps_data.shape[:3] + (4,))
    maps_i_correct = maps_data[..., :4].copy()
    maps_i_correct[np.logical_not(mask_data), :] = 0
    np.testing.assert_almost_equal(maps_i_correct, maps_i)
    mask_data[1, 1, 1] = 0  # for test to succeed
    np.testing.assert_equal(mask_data, maps_i_mask)
    mask_data[1, 1, 1] = 1  # reset, just in case.
    np.testing.assert_equal(np.asarray(list(range(4))), maps_i_indices)


@pytest.mark.parametrize(
    "target_dtype",
    (float, np.float32, np.float64, int, np.uint),
)
def test_img_to_signals_labels_non_float_type(target_dtype):
    rng = np.random.RandomState(42)

    fake_fmri_data = rng.uniform(size=(10, 10, 10, N_TIMEPOINTS)) > 0.5
    fake_affine = np.eye(4, 4).astype(np.float64)
    fake_fmri_img_orig = nibabel.Nifti1Image(
        fake_fmri_data.astype(np.float64), fake_affine
    )
    fake_fmri_img_target_dtype = new_img_like(
        fake_fmri_img_orig, fake_fmri_data.astype(target_dtype)
    )
    fake_mask_data = np.ones((10, 10, 10), dtype=np.uint8)
    fake_mask = nibabel.Nifti1Image(fake_mask_data, fake_affine)

    masker = NiftiLabelsMasker(fake_mask)
    masker.fit()
    timeseries_int = masker.transform(fake_fmri_img_target_dtype)
    timeseries_float = masker.transform(fake_fmri_img_orig)
    assert np.sum(timeseries_int) != 0
    assert np.allclose(timeseries_int, timeseries_float)
