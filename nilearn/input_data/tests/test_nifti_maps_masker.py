"""Test the nifti_region module

Functions in this file only test features added by the NiftiLabelsMasker class,
non_overlappingt the underlying functions (clean(), img_to_signals_labels(), etc.). See
test_masking.py and test_signal.py for details.
"""

import numpy as np

import nibabel
import pytest

from nilearn.input_data.nifti_maps_masker import NiftiMapsMasker
from nilearn._utils import testing, as_ndarray, data_gen
from nilearn._utils.exceptions import DimensionError
from nilearn.image import get_data


def generate_random_img(shape, length=1, affine=np.eye(4),
                        rand_gen=np.random.RandomState(0)):
    data = rand_gen.standard_normal(size=(shape + (length,)))
    return nibabel.Nifti1Image(data, affine), nibabel.Nifti1Image(
        as_ndarray(data[..., 0] > 0.2, dtype=np.int8), affine)


def test_nifti_maps_masker():
    # Check working of shape/affine checks
    shape1 = (13, 11, 12)
    affine1 = np.eye(4)

    shape2 = (12, 10, 14)
    affine2 = np.diag((1, 2, 3, 1))

    n_regions = 9
    length = 3

    fmri11_img, mask11_img = generate_random_img(shape1, affine=affine1,
                                                 length=length)
    fmri12_img, mask12_img = generate_random_img(shape1, affine=affine2,
                                                 length=length)
    fmri21_img, mask21_img = generate_random_img(shape2, affine=affine1,
                                                 length=length)

    labels11_img, labels_mask_img = \
        data_gen.generate_maps(shape1, n_regions, affine=affine1)

    # No exception raised here
    for create_files in (True, False):
        with testing.write_tmp_imgs(labels11_img, create_files=create_files) \
                as labels11:
            masker11 = NiftiMapsMasker(labels11, resampling_target=None)
            signals11 = masker11.fit().transform(fmri11_img)
            assert signals11.shape == (length, n_regions)
            # enables to delete "labels11" on windows
            del masker11

    masker11 = NiftiMapsMasker(labels11_img, mask_img=mask11_img,
                               resampling_target=None)

    with pytest.raises(ValueError, match='has not been fitted. '):
        masker11.transform(fmri11_img)
    signals11 = masker11.fit().transform(fmri11_img)
    assert signals11.shape == (length, n_regions)

    NiftiMapsMasker(labels11_img).fit_transform(fmri11_img)

    # Test all kinds of mismatches between shapes and between affines
    for create_files in (True, False):
        with testing.write_tmp_imgs(labels11_img, mask12_img,
                                    create_files=create_files) as images:
            labels11, mask12 = images
            masker11 = NiftiMapsMasker(labels11, resampling_target=None)
            masker11.fit()
            pytest.raises(ValueError, masker11.transform, fmri12_img)
            pytest.raises(ValueError, masker11.transform, fmri21_img)

            masker11 = NiftiMapsMasker(labels11, mask_img=mask12,
                                       resampling_target=None)
            pytest.raises(ValueError, masker11.fit)
            del masker11

    masker11 = NiftiMapsMasker(labels11_img, mask_img=mask21_img,
                               resampling_target=None)
    pytest.raises(ValueError, masker11.fit)

    # Transform, with smoothing (smoke test)
    masker11 = NiftiMapsMasker(labels11_img, smoothing_fwhm=3,
                               resampling_target=None)
    signals11 = masker11.fit().transform(fmri11_img)
    assert signals11.shape == (length, n_regions)

    masker11 = NiftiMapsMasker(labels11_img, smoothing_fwhm=3,
                               resampling_target=None)
    signals11 = masker11.fit_transform(fmri11_img)
    assert signals11.shape == (length, n_regions)

    with pytest.raises(ValueError, match='has not been fitted. '):
        NiftiMapsMasker(labels11_img).inverse_transform(signals11)

    # Call inverse transform (smoke test)
    fmri11_img_r = masker11.inverse_transform(signals11)
    assert fmri11_img_r.shape == fmri11_img.shape
    np.testing.assert_almost_equal(fmri11_img_r.affine, fmri11_img.affine)

    # Now try on a masker that has never seen the call to "transform"
    masker2 = NiftiMapsMasker(labels11_img, resampling_target=None)
    masker2.fit()
    masker2.inverse_transform(signals11)

    # Test with data and atlas of different shape: the atlas should be
    # resampled to the data
    shape22 = (5, 5, 6)
    affine2 = 2 * np.eye(4)
    affine2[-1, -1] = 1

    fmri22_img, _ = generate_random_img(shape22, affine=affine2,
                                        length=length)
    masker = NiftiMapsMasker(labels11_img, mask_img=mask21_img)

    masker.fit_transform(fmri22_img)
    np.testing.assert_array_equal(masker._resampled_maps_img_.affine,
                                  affine2)


def test_nifti_maps_masker_with_nans_and_infs():
    """Apply a NiftiMapsMasker containing NaNs and infs.

    The masker should replace those NaNs and infs with zeros,
    without raising a warning.
    """
    length = 3
    n_regions = 8
    fmri_img, mask_img = generate_random_img((13, 11, 12),
                                             affine=np.eye(4), length=length)
    maps_img, maps_mask_img = data_gen.generate_maps((13, 11, 12), n_regions,
                                                     affine=np.eye(4))

    # Add NaNs and infs to atlas
    maps_data = get_data(maps_img).astype(np.float32)
    mask_data = get_data(mask_img).astype(np.float32)
    maps_data = maps_data * mask_data[..., None]

    # Choose a good voxel from the first label
    vox_idx = np.where(maps_data[..., 0] > 0)
    i1, j1, k1 = vox_idx[0][0], vox_idx[1][0], vox_idx[2][0]
    i2, j2, k2 = vox_idx[0][1], vox_idx[1][1], vox_idx[2][1]

    maps_data[:, :, :, 0] = np.nan
    maps_data[i2, j2, k2, 0] = np.inf
    maps_data[i1, j1, k1, 0] = 1

    maps_img = nibabel.Nifti1Image(maps_data, np.eye(4))

    # No warning, because maps_img is run through clean_img
    # *before* _safe_get_data.
    masker = NiftiMapsMasker(maps_img, mask_img=mask_img)

    sig = masker.fit_transform(fmri_img)

    assert sig.shape == (length, n_regions)
    assert np.all(np.isfinite(sig))


def test_nifti_maps_masker_with_nans_and_infs_in_mask():
    """Apply a NiftiMapsMasker with a mask containing NaNs and infs.

    The masker should replace those NaNs and infs with zeros,
    while raising a warning.
    """
    length = 3
    n_regions = 8
    fmri_img, mask_img = generate_random_img((13, 11, 12),
                                             affine=np.eye(4), length=length)
    maps_img, maps_mask_img = data_gen.generate_maps((13, 11, 12), n_regions,
                                                     affine=np.eye(4))

    # Add NaNs and infs to mask
    mask_data = np.array(get_data(mask_img), dtype=np.float64)

    mask_data[:, :, 7] = np.nan
    mask_data[:, :, 5] = np.inf

    mask_img = nibabel.Nifti1Image(mask_data, np.eye(4))

    masker = NiftiMapsMasker(maps_img, mask_img=mask_img)

    with pytest.warns(UserWarning, match="Non-finite values detected."):
        sig = masker.fit_transform(fmri_img)

    assert sig.shape == (length, n_regions)
    assert np.all(np.isfinite(sig))


def test_nifti_maps_masker_with_nans_and_infs_in_data():
    """Apply a NiftiMapsMasker to 4D data containing NaNs and infs.

    The masker should replace those NaNs and infs with zeros,
    while raising a warning.
    """
    length = 3
    n_regions = 8
    fmri_img, mask_img = generate_random_img((13, 11, 12),
                                             affine=np.eye(4), length=length)
    maps_img, maps_mask_img = data_gen.generate_maps((13, 11, 12), n_regions,
                                                     affine=np.eye(4))

    # Add NaNs and infs to data
    fmri_data = get_data(fmri_img)

    fmri_data[:, 9, 9, :] = np.nan
    fmri_data[:, 5, 5, :] = np.inf

    fmri_img = nibabel.Nifti1Image(fmri_data, np.eye(4))

    masker = NiftiMapsMasker(maps_img, mask_img=mask_img)

    with pytest.warns(UserWarning, match="Non-finite values detected."):
        sig = masker.fit_transform(fmri_img)

    assert sig.shape == (length, n_regions)
    assert np.all(np.isfinite(sig))


def test_nifti_maps_masker_2():
    # Test resampling in NiftiMapsMasker
    affine = np.eye(4)

    shape1 = (10, 11, 12)  # fmri
    shape2 = (13, 14, 15)  # mask
    shape3 = (16, 17, 18)  # maps

    n_regions = 9
    length = 3

    fmri11_img, _ = generate_random_img(shape1, affine=affine,
                                        length=length)
    _, mask22_img = generate_random_img(shape2, affine=affine,
                                        length=length)

    maps33_img, _ = \
        data_gen.generate_maps(shape3, n_regions, affine=affine)

    mask_img_4d = nibabel.Nifti1Image(np.ones((2, 2, 2, 2), dtype=np.int8),
                                      affine=np.diag((4, 4, 4, 1)))

    # verify that 4D mask arguments are refused
    masker = NiftiMapsMasker(maps33_img, mask_img=mask_img_4d)
    with pytest.raises(DimensionError,
                       match="Input data has incompatible dimensionality: "
                             "Expected dimension is 3D and you provided "
                             "a 4D image."):
        masker.fit()

    # Test error checking
    pytest.raises(ValueError, NiftiMapsMasker, maps33_img,
                  resampling_target="mask")
    pytest.raises(ValueError, NiftiMapsMasker, maps33_img,
                  resampling_target="invalid")

    # Target: mask
    masker = NiftiMapsMasker(maps33_img, mask_img=mask22_img,
                             resampling_target="mask")

    masker.fit()
    np.testing.assert_almost_equal(masker.mask_img_.affine,
                                   mask22_img.affine)
    assert masker.mask_img_.shape == mask22_img.shape

    np.testing.assert_almost_equal(masker.mask_img_.affine,
                                   masker.maps_img_.affine)
    assert masker.mask_img_.shape == masker.maps_img_.shape[:3]

    transformed = masker.transform(fmri11_img)
    assert transformed.shape == (length, n_regions)

    fmri11_img_r = masker.inverse_transform(transformed)
    np.testing.assert_almost_equal(fmri11_img_r.affine,
                                   masker.maps_img_.affine)
    assert fmri11_img_r.shape == (masker.maps_img_.shape[:3] + (length,))

    # Target: maps
    masker = NiftiMapsMasker(maps33_img, mask_img=mask22_img,
                             resampling_target="maps")

    masker.fit()
    np.testing.assert_almost_equal(masker.maps_img_.affine,
                                   maps33_img.affine)
    assert masker.maps_img_.shape == maps33_img.shape

    np.testing.assert_almost_equal(masker.mask_img_.affine,
                                   masker.maps_img_.affine)
    assert masker.mask_img_.shape == masker.maps_img_.shape[:3]

    transformed = masker.transform(fmri11_img)
    assert transformed.shape == (length, n_regions)

    fmri11_img_r = masker.inverse_transform(transformed)
    np.testing.assert_almost_equal(fmri11_img_r.affine,
                                   masker.maps_img_.affine)
    assert fmri11_img_r.shape == (masker.maps_img_.shape[:3] + (length,))

    # Test with clipped maps: mask does not contain all maps.
    # Shapes do matter in that case
    affine1 = np.eye(4)
    shape1 = (10, 11, 12)
    shape2 = (8, 9, 10)  # mask
    affine2 = np.diag((2, 2, 2, 1))  # just for mask
    shape3 = (16, 18, 20)  # maps

    n_regions = 9
    length = 21

    fmri11_img, _ = generate_random_img(shape1, affine=affine1, length=length)
    _, mask22_img = data_gen.generate_fake_fmri(shape2, length=1,
                                                affine=affine2)
    # Target: maps
    maps33_img, _ = \
        data_gen.generate_maps(shape3, n_regions, affine=affine1)

    masker = NiftiMapsMasker(maps33_img, mask_img=mask22_img,
                             resampling_target="maps")

    masker.fit()
    np.testing.assert_almost_equal(masker.maps_img_.affine,
                                   maps33_img.affine)
    assert masker.maps_img_.shape == maps33_img.shape

    np.testing.assert_almost_equal(masker.mask_img_.affine,
                                   masker.maps_img_.affine)
    assert masker.mask_img_.shape == masker.maps_img_.shape[:3]

    transformed = masker.transform(fmri11_img)
    assert transformed.shape == (length, n_regions)
    # Some regions have been clipped. Resulting signal must be zero
    assert (transformed.var(axis=0) == 0).sum() < n_regions

    fmri11_img_r = masker.inverse_transform(transformed)
    np.testing.assert_almost_equal(fmri11_img_r.affine,
                                   masker.maps_img_.affine)
    assert (fmri11_img_r.shape ==
                 (masker.maps_img_.shape[:3] + (length,)))


def test_nifti_maps_masker_overlap():
    # Test resampling in NiftiMapsMasker
    affine = np.eye(4)
    shape = (5, 5, 5)
    length = 10

    fmri_img, _ = generate_random_img(shape, affine=affine,
                                      length=length)
    non_overlapping_maps = np.zeros(shape + (2,))
    non_overlapping_maps[:2, :, :, 0] = 1.
    non_overlapping_maps[2:, :, :, 1] = 1.
    non_overlapping_maps_img = nibabel.Nifti1Image(non_overlapping_maps,
                                                   affine)

    overlapping_maps = np.zeros(shape + (2,))
    overlapping_maps[:3, :, :, 0] = 1.
    overlapping_maps[2:, :, :, 1] = 1.
    overlapping_maps_img = nibabel.Nifti1Image(overlapping_maps, affine)

    overlapping_masker = NiftiMapsMasker(non_overlapping_maps_img,
                                         allow_overlap=True)
    overlapping_masker.fit_transform(fmri_img)
    overlapping_masker = NiftiMapsMasker(overlapping_maps_img,
                                         allow_overlap=True)
    overlapping_masker.fit_transform(fmri_img)

    non_overlapping_masker = NiftiMapsMasker(non_overlapping_maps_img,
                                             allow_overlap=False)
    non_overlapping_masker.fit_transform(fmri_img)
    non_overlapping_masker = NiftiMapsMasker(overlapping_maps_img,
                                             allow_overlap=False)
    with pytest.raises(ValueError, match='Overlap detected'):
        non_overlapping_masker.fit_transform(fmri_img)


def test_standardization():
    rng = np.random.RandomState(42)
    data_shape = (9, 9, 5)
    n_samples = 500

    signals = rng.standard_normal(size=(np.prod(data_shape), n_samples))
    means = rng.standard_normal(size=(np.prod(data_shape), 1)) * 50 + 1000
    signals += means
    img = nibabel.Nifti1Image(signals.reshape(data_shape + (n_samples,)),
                              np.eye(4))

    maps, _ = data_gen.generate_maps((9, 9, 5), 10)

    # Unstandarized
    masker = NiftiMapsMasker(maps, standardize=False)
    unstandarized_label_signals = masker.fit_transform(img)

    # z-score
    masker = NiftiMapsMasker(maps,
                             standardize='zscore')
    trans_signals = masker.fit_transform(img)

    np.testing.assert_almost_equal(trans_signals.mean(0), 0)
    np.testing.assert_almost_equal(trans_signals.std(0), 1)

    # psc
    masker = NiftiMapsMasker(maps, standardize='psc')
    trans_signals = masker.fit_transform(img)

    np.testing.assert_almost_equal(trans_signals.mean(0), 0)
    np.testing.assert_almost_equal(
            trans_signals,
            unstandarized_label_signals /
            unstandarized_label_signals.mean(0) * 100 - 100,
            )


def test_3d_images():
    # Test that the NiftiMapsMasker works with 3D images
    affine = np.eye(4)
    n_regions = 3
    shape3 = (16, 17, 18)

    maps33_img, _ = data_gen.generate_maps(shape3, n_regions)
    mask_img = nibabel.Nifti1Image(np.ones(shape3, dtype=np.int8),
                           affine=affine)
    epi_img1 = nibabel.Nifti1Image(np.ones(shape3),
                           affine=affine)
    epi_img2 = nibabel.Nifti1Image(np.ones(shape3),
                           affine=affine)
    masker = NiftiMapsMasker(maps33_img, mask_img=mask_img)

    epis = masker.fit_transform(epi_img1)
    assert(epis.shape == (1, 3))
    epis = masker.fit_transform([epi_img1, epi_img2])
    assert(epis.shape == (2, 3))
