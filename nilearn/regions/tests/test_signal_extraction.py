"""
Test for "region" module.
"""
# Author: Ph. Gervais
# License: simplified BSD

import numpy as np
import nibabel
import pytest

from nilearn.maskers import NiftiLabelsMasker
from nilearn.regions import signal_extraction
from nilearn._utils.testing import write_tmp_imgs
from nilearn._utils.data_gen import generate_timeseries, generate_regions_ts
from nilearn._utils.data_gen import generate_labeled_regions, generate_maps
from nilearn._utils.data_gen import generate_fake_fmri
from nilearn._utils.exceptions import DimensionError
from nilearn.image import get_data, new_img_like


_TEST_DIM_ERROR_MSG = ("Input data has incompatible dimensionality: "
                       "Expected dimension is 3D and you provided "
                       "a 4D image")
_TEST_SHAPE_ERROR_MSG = "Images have incompatible shapes."
_TEST_AFFINE_ERROR_MSG = "Images have different affine matrices."

INF = 1000 * np.finfo(np.float32).eps

        
def test__check_shape_and_affine_compatibility_without_dim():
    """Ensure correct behaviour for valid data without dim"""
    shape = (2, 3, 4)
    affine = np.eye(4)
    img1 = nibabel.Nifti1Image(np.zeros(shape), affine)

    img2 = nibabel.Nifti1Image(np.zeros(shape), affine)

    with pytest.warns(None):
        signal_extraction._check_shape_and_affine_compatibility(
            img1,
            img2)
        
        
def test__check_shape_and_affine_compatibility_with_dim():
    """Ensure correct behaviour for valid data without dim"""
    shape = (2, 3, 4, 7)
    affine = np.eye(4)
    img1 = nibabel.Nifti1Image(np.zeros(shape[:3]), affine)

    mask_shape = (2, 3, 4)
    img2 = nibabel.Nifti1Image(np.zeros(mask_shape), affine)

    with pytest.warns(None):
        signal_extraction._check_shape_and_affine_compatibility(
            img1,
            img2,
            dim=3)

@pytest.mark.parametrize(
    "test_shape,test_affine,msg", 
    [((2, 3, 5), np.eye(4), _TEST_SHAPE_ERROR_MSG),
     ((2, 3, 4), 2*np.eye(4), _TEST_AFFINE_ERROR_MSG)
     ])
def test__check_shape_and_affine_compatibility_error(test_shape,test_affine,msg):
    shape = (2, 3, 4)
    affine = np.eye(4)
    img1 = nibabel.Nifti1Image(np.zeros(shape), affine)

    img2 = nibabel.Nifti1Image(np.zeros(test_shape), test_affine)

    with pytest.raises(ValueError, match=msg):
        signal_extraction._check_shape_and_affine_compatibility(
            img1,
            img2)


def test_generate_regions_ts():
    """Minimal testing of generate_regions_ts()."""
    # Check that no regions overlap
    n_voxels = 50
    n_regions = 10
    regions = generate_regions_ts(n_voxels, n_regions, overlap=0)
    assert regions.shape == (n_regions, n_voxels)
    # check: no overlap
    np.testing.assert_array_less((regions > 0).sum(axis=0) - 0.1,
                                 np.ones(regions.shape[1]))
    # check: a region everywhere
    np.testing.assert_array_less(np.zeros(regions.shape[1]),
                                 (regions > 0).sum(axis=0))

    regions = generate_regions_ts(n_voxels, n_regions, overlap=0,
                                  window="hamming")
    assert regions.shape == (n_regions, n_voxels)
    # check: no overlap
    np.testing.assert_array_less((regions > 0).sum(axis=0) - 0.1,
                                 np.ones(regions.shape[1]))
    # check: a region everywhere
    np.testing.assert_array_less(np.zeros(regions.shape[1]),
                                 (regions > 0).sum(axis=0))

    # Check that some regions overlap
    regions = generate_regions_ts(n_voxels, n_regions, overlap=1)
    assert regions.shape == (n_regions, n_voxels)
    assert (np.any((regions > 0).sum(axis=-1) > 1.9))

    regions = generate_regions_ts(n_voxels, n_regions, overlap=1,
                                  window="hamming")
    assert (np.any((regions > 0).sum(axis=-1) > 1.9))


def test_generate_labeled_regions():
    """Minimal testing of generate_labeled_regions."""
    shape = (3, 4, 5)
    n_regions = 10
    regions = generate_labeled_regions(shape, n_regions)
    assert regions.shape == shape
    assert (len(np.unique(get_data(regions))) == n_regions + 1)


def test_signals_extraction_with_labels():
    """Test conversion between signals and images \
    using regions defined by labels."""
    shape = (8, 9, 10)
    n_instants = 11
    n_regions = 8  # must be 8

    eps = np.finfo(np.float64).eps
    # data
    affine = np.eye(4)
    signals = generate_timeseries(n_instants, n_regions)

    # mask
    mask_data = np.zeros(shape)
    mask_data[1:-1, 1:-1, 1:-1] = 1
    mask_img = nibabel.Nifti1Image(mask_data, affine)

    mask_4d_img = nibabel.Nifti1Image(np.ones(shape + (2, )), affine)

    # labels
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

    labels_img = nibabel.Nifti1Image(labels_data, affine)

    labels_4d_data = np.zeros((shape) + (2, ))
    labels_4d_data[..., 0] = labels_data
    labels_4d_data[..., 1] = labels_data
    labels_4d_img = nibabel.Nifti1Image(labels_4d_data, np.eye(4))

    # Without mask
    # from labels
    data_img = signal_extraction.signals_to_img_labels(signals, labels_img)
    data = get_data(data_img)
    assert data_img.shape == (shape + (n_instants,))
    assert np.all(data.std(axis=-1) > 0)

    # verify that 4D label images are refused
    with pytest.raises(DimensionError, match=_TEST_DIM_ERROR_MSG):
        signal_extraction.img_to_signals_labels(data_img, labels_4d_img)

    # There must be non-zero data (safety net)
    assert abs(data).max() > 1e-9

    # Check that signals in each region are identical in each voxel
    for n in range(1, n_regions + 1):
        sigs = data[labels_data == n, :]
        np.testing.assert_almost_equal(sigs[0, :], signals[:, n - 1])
        assert abs(sigs - sigs[0, :]).max() < eps

    # and back
    signals_r, labels_r = signal_extraction.img_to_signals_labels(data_img,
                                                                  labels_img)
    np.testing.assert_almost_equal(signals_r, signals)
    assert labels_r == list(range(1, 9))

    with write_tmp_imgs(data_img) as fname_img:
        signals_r, labels_r = signal_extraction.img_to_signals_labels(
            fname_img, labels_img)
        np.testing.assert_almost_equal(signals_r, signals)
        assert labels_r == list(range(1, 9))

    # Same thing, with mask.
    with pytest.raises(DimensionError, match=_TEST_DIM_ERROR_MSG):
        signal_extraction.img_to_signals_labels(data_img, labels_img,
                                                mask_img=mask_4d_img
                                                )
    with pytest.raises(DimensionError, match=_TEST_DIM_ERROR_MSG):
        signal_extraction.signals_to_img_labels(data_img, labels_img,
                                                mask_img=mask_4d_img
                                                )
    data_img = signal_extraction.signals_to_img_labels(signals, labels_img,
                                                       mask_img=mask_img)
    with pytest.raises(TypeError):
        signal_extraction.signals_to_img_labels(data_img, labels_4d_img,
                                                mask_img=mask_img
                                                )
    assert data_img.shape == (shape + (n_instants,))

    data = get_data(data_img)
    assert abs(data).max() > 1e-9
    # Zero outside of the mask
    assert np.all(data[np.logical_not(get_data(mask_img))].std(axis=-1) < eps)

    with write_tmp_imgs(labels_img, mask_img) as filenames:
        data_img = signal_extraction.signals_to_img_labels(
            signals, filenames[0], mask_img=filenames[1])
        assert data_img.shape == (shape + (n_instants,))

        data = get_data(data_img)
        assert abs(data).max() > 1e-9
        # Zero outside of the mask
        assert np.all(data[np.logical_not(get_data(mask_img))].std(axis=-1)
                      < eps)

    # mask labels before checking
    masked_labels_data = labels_data.copy()
    masked_labels_data[np.logical_not(get_data(mask_img))] = 0
    for n in range(1, n_regions + 1):
        sigs = data[masked_labels_data == n, :]
        np.testing.assert_almost_equal(sigs[0, :], signals[:, n - 1])
        assert abs(sigs - sigs[0, :]).max() < eps

    # and back
    signals_r, labels_r = signal_extraction.img_to_signals_labels(
        data_img, labels_img, mask_img=mask_img)
    np.testing.assert_almost_equal(signals_r, signals)
    assert labels_r == list(range(1, 9))


def test_signal_extraction_with_maps():
    shape = (10, 11, 12)
    n_regions = 9
    n_instants = 13

    # Generate signals
    rng = np.random.RandomState(42)

    maps_img, mask_img = generate_maps(shape, n_regions, border=1)
    maps_data = get_data(maps_img)
    data = np.zeros(shape + (n_instants,), dtype=np.float32)

    mask_4d_img = nibabel.Nifti1Image(np.ones((shape + (2, ))), np.eye(4))

    signals = np.zeros((n_instants, maps_data.shape[-1]))
    for n in range(maps_data.shape[-1]):
        signals[:, n] = rng.standard_normal(size=n_instants)
        data[maps_data[..., n] > 0, :] = signals[:, n]
    img = nibabel.Nifti1Image(data, np.eye(4))

    # verify that 4d masks are refused
    with pytest.raises(TypeError, match=_TEST_DIM_ERROR_MSG):
        signal_extraction.img_to_signals_maps(img, maps_img,
                                              mask_img=mask_4d_img)

    # Get signals
    signals_r, labels = signal_extraction.img_to_signals_maps(
        img, maps_img, mask_img=mask_img)

    # The output must be identical to the input signals, because every region
    # is homogeneous:
    # there is the same signal in all voxels of one given region
    # (and all maps are uniform).
    np.testing.assert_almost_equal(signals, signals_r)

    # Same thing without mask (in that case)
    signals_r, labels = signal_extraction.img_to_signals_maps(img, maps_img)
    np.testing.assert_almost_equal(signals, signals_r)

    # Recover image
    img_r = signal_extraction.signals_to_img_maps(signals, maps_img,
                                                  mask_img=mask_img)
    np.testing.assert_almost_equal(get_data(img_r), get_data(img))
    img_r = signal_extraction.signals_to_img_maps(signals, maps_img)
    np.testing.assert_almost_equal(get_data(img_r), get_data(img))


@pytest.mark.parametrize("z_dim,affine_diag", [(5, 1), (4, 2)])
def test_input_validation_bad_masks(z_dim, affine_diag):
    data_img = nibabel.Nifti1Image(np.zeros((2, 3, 4, 5)), np.eye(4))

    bad_mask_img = nibabel.Nifti1Image(np.zeros((2, 3, z_dim)),
                                       affine_diag * np.eye(4))

    labels_img = nibabel.Nifti1Image(np.zeros((2, 3, 4)), np.eye(4))
    pytest.raises(ValueError, signal_extraction.img_to_signals_labels,
                  data_img,
                  labels_img,
                  mask_img=bad_mask_img)

    maps_img = nibabel.Nifti1Image(np.zeros((2, 3, 4, 7)), np.eye(4))
    pytest.raises(ValueError, signal_extraction.img_to_signals_maps,
                  data_img,
                  maps_img,
                  mask_img=bad_mask_img)


@pytest.mark.parametrize("z_dim,affine_diag", [(5, 1), (4, 2)])
def test_input_validation_bad_labels(z_dim, affine_diag):
    data_img = nibabel.Nifti1Image(np.zeros((2, 3, 4, 5)), np.eye(4))

    bad_labels_img = nibabel.Nifti1Image(np.zeros((2, 3, z_dim)),
                                         affine_diag * np.eye(4))

    pytest.raises(ValueError, signal_extraction.img_to_signals_labels,
                  data_img,
                  bad_labels_img)

    good_mask_img = nibabel.Nifti1Image(np.zeros((2, 3, 4)), np.eye(4))
    pytest.raises(ValueError, signal_extraction.img_to_signals_labels,
                  data_img,
                  bad_labels_img,
                  mask_img=good_mask_img)


@pytest.mark.parametrize("z_dim,affine_diag", [(5, 1), (4, 2)])
def test_input_validation_bad_maps(z_dim, affine_diag):
    data_img = nibabel.Nifti1Image(np.zeros((2, 3, 4, 5)), np.eye(4))

    bad_maps_img = nibabel.Nifti1Image(np.zeros((2, 3, z_dim, 7)),
                                       affine_diag * np.eye(4))

    pytest.raises(ValueError, signal_extraction.img_to_signals_maps,
                  data_img,
                  bad_maps_img)

    good_mask_img = nibabel.Nifti1Image(np.zeros((2, 3, 4)), np.eye(4))
    pytest.raises(ValueError, signal_extraction.img_to_signals_maps,
                  data_img,
                  bad_maps_img,
                  mask_img=good_mask_img)


def test_signal_extraction_with_labels_error_strategy():
    shape = (4, 5, 6)
    n_regions = 7
    labels = list(range(n_regions + 1))  # 0 is background
    labels_img = generate_labeled_regions(shape, n_regions, labels=labels)

    length = 8
    fmri_img, _ = generate_fake_fmri(shape=shape,
                                     length=length,
                                     affine=labels_img.affine)
    
    with pytest.raises(ValueError, match="Invalid strategy"):
        signal_extraction.img_to_signals_labels(
            fmri_img,
            labels_img,
            strategy="foo"
        )


def test_signal_extraction_with_maps_and_labels():
    shape = (4, 5, 6)
    n_regions = 7
    length = 8

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
    fmri_img, _ = generate_fake_fmri(shape=shape, length=length,
                                     affine=labels_img.affine)

    # Extract signals from maps and labels: results must be identical.
    maps_signals, maps_labels = signal_extraction.img_to_signals_maps(
        fmri_img, maps_img)
    labels_signals, labels_labels = signal_extraction.img_to_signals_labels(
        fmri_img, labels_img)

    np.testing.assert_almost_equal(maps_signals, labels_signals)

    # Same thing with a mask, containing only 3 regions.
    mask_data = (labels_data == 1) + (labels_data == 2) + (labels_data == 5)
    mask_img = nibabel.Nifti1Image(mask_data.astype(np.int8),
                                   labels_img.affine)
    labels_signals, labels_labels = signal_extraction.img_to_signals_labels(
        fmri_img, labels_img, mask_img=mask_img)

    maps_signals, maps_labels = signal_extraction.img_to_signals_maps(
        fmri_img, maps_img, mask_img=mask_img)

    np.testing.assert_almost_equal(maps_signals, labels_signals)
    assert maps_signals.shape[1] == n_regions
    assert maps_labels == list(range(len(maps_labels)))
    assert labels_signals.shape == (length, n_regions)
    assert labels_labels == labels[1:]

    # Inverse operation (mostly smoke test)
    labels_img_r = signal_extraction.signals_to_img_labels(
        labels_signals, labels_img, mask_img=mask_img)
    assert labels_img_r.shape == shape + (length,)

    maps_img_r = signal_extraction.signals_to_img_maps(
        maps_signals, maps_img, mask_img=mask_img)
    assert maps_img_r.shape == shape + (length,)

    # Check that NaNs in regions inside mask are replaced with zeros
    region1 = labels_data == 2
    indices = tuple(ind[:1] for ind in np.where(region1))
    get_data(fmri_img)[indices] = np.nan
    labels_signals, labels_labels = signal_extraction.img_to_signals_labels(
        fmri_img, labels_img, mask_img=mask_img)
    assert np.all(labels_signals[:, labels_labels.index(2)] == 0.)


def test_generate_maps():
    # Basic testing of generate_maps()
    shape = (10, 11, 12)
    n_regions = 9
    maps_img, _ = generate_maps(shape, n_regions, border=1)
    maps = get_data(maps_img)
    assert maps.shape == shape + (n_regions,)
    # no empty map
    assert np.all(abs(maps).sum(axis=0).sum(axis=0).sum(axis=0) > 0)
    # check border
    assert np.all(maps[0, ...] == 0)
    assert np.all(maps[:, 0, ...] == 0)
    assert np.all(maps[:, :, 0, :] == 0)


def test__trim_maps():
    shape = (7, 9, 10)
    n_regions = 8

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

    maps_i, maps_i_mask, maps_i_indices = signal_extraction._trim_maps(
        maps_data, mask_data)
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

    maps_i, maps_i_mask, maps_i_indices = signal_extraction._trim_maps(
        maps_data, mask_data)
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


@pytest.mark.parametrize('target_dtype',
                         (float, np.float32, np.float64, int, np.uint),
                         )
def test_img_to_signals_labels_non_float_type(target_dtype):
    fake_fmri_data = (
        np.random.RandomState(42).uniform(size=(10, 10, 10, 10)) > 0.5
    )
    fake_affine = np.eye(4, 4).astype(np.float64)
    fake_fmri_img_orig = nibabel.Nifti1Image(fake_fmri_data.astype(np.float64),
                                             fake_affine)
    fake_fmri_img_target_dtype = new_img_like(
        fake_fmri_img_orig,
        fake_fmri_data.astype(target_dtype)
    )
    fake_mask_data = np.ones((10, 10, 10), dtype=np.uint8)
    fake_mask = nibabel.Nifti1Image(fake_mask_data, fake_affine)

    masker = NiftiLabelsMasker(fake_mask)
    masker.fit()
    timeseries_int = masker.transform(fake_fmri_img_target_dtype)
    timeseries_float = masker.transform(fake_fmri_img_orig)
    assert np.sum(timeseries_int) != 0
    assert np.allclose(timeseries_int, timeseries_float)
