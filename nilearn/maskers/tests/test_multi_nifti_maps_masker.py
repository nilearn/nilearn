"""Test the multi_nifti_maps_masker module."""

import numpy as np
import pytest
from nibabel import Nifti1Image
from numpy.testing import assert_almost_equal, assert_array_equal

from nilearn._utils import data_gen, testing
from nilearn._utils.exceptions import DimensionError
from nilearn.maskers import MultiNiftiMapsMasker, NiftiMapsMasker


@pytest.fixture()
def n_regions():
    return 9


@pytest.fixture()
def length():
    return 3


@pytest.mark.parametrize("create_files", (True, False))
def test_multi_nifti_maps_masker_with_files(
    create_files, affine_eye, shape_3d_default, n_regions, length
):
    fmri_img, _ = data_gen.generate_fake_fmri(
        shape=shape_3d_default, affine=affine_eye, length=length
    )

    maps_img, _ = data_gen.generate_maps(
        shape=shape_3d_default, n_regions=n_regions, affine=affine_eye
    )

    with testing.write_tmp_imgs(maps_img, create_files=create_files) as labels:
        masker = MultiNiftiMapsMasker(labels, resampling_target=None)
        signals = masker.fit().transform(fmri_img)
        assert signals.shape == (length, n_regions)


@pytest.mark.parametrize("create_files", (True, False))
def test_multi_nifti_maps_masker_errors_with_files(
    create_files, affine_eye, shape_3d_default, n_regions, length
):
    shape2 = (12, 10, 14)
    affine2 = np.diag((1, 2, 3, 1))

    maps_img, _ = data_gen.generate_maps(
        shape=shape_3d_default, n_regions=n_regions, affine=affine_eye
    )

    fmri12_img, mask12_img = data_gen.generate_fake_fmri(
        shape=shape_3d_default, affine=affine2, length=length
    )
    fmri21_img, _ = data_gen.generate_fake_fmri(
        shape2, affine=affine_eye, length=length
    )

    with testing.write_tmp_imgs(
        maps_img, mask12_img, create_files=create_files
    ) as images:
        labels, mask12 = images
        masker = MultiNiftiMapsMasker(labels, resampling_target=None)
        masker.fit()
        with pytest.raises(ValueError):
            masker.transform(fmri12_img)
        with pytest.raises(ValueError):
            masker.transform(fmri21_img)

        masker = MultiNiftiMapsMasker(
            labels, mask_img=mask12, resampling_target=None
        )
        with pytest.raises(ValueError):
            masker.fit()
        del masker


def test_multi_nifti_maps_masker(
    affine_eye, shape_3d_default, n_regions, length
):
    # Check working of shape/affine checks
    affine2 = np.diag((1, 2, 3, 1))

    fmri_img, mask_img = data_gen.generate_fake_fmri(
        shape=shape_3d_default, affine=affine_eye, length=length
    )

    maps_img, _ = data_gen.generate_maps(
        shape=shape_3d_default, n_regions=n_regions, affine=affine_eye
    )

    # No exception raised here
    masker = MultiNiftiMapsMasker(
        maps_img, mask_img=mask_img, resampling_target=None
    )

    with pytest.raises(ValueError, match="has not been fitted. "):
        masker.transform(fmri_img)
    signals = masker.fit().transform(fmri_img)
    assert signals.shape == (length, n_regions)

    MultiNiftiMapsMasker(maps_img).fit_transform(fmri_img)

    # Should work with 4D + 1D input too (also test fit_transform)
    signals_input = [fmri_img, fmri_img]
    signals_list = masker.fit_transform(signals_input)
    assert len(signals_list) == len(signals_input)
    for signals in signals_list:
        assert signals.shape == (length, n_regions)

    # NiftiMapsMasker should not work with 4D + 1D input
    signals_input = [fmri_img, fmri_img]
    masker = NiftiMapsMasker(maps_img, resampling_target=None)
    with pytest.raises(DimensionError, match="incompatible dimensionality"):
        masker.fit_transform(signals_input)

    shape2 = (12, 10, 14)
    _, mask21_img = data_gen.generate_fake_fmri(
        shape2, affine=affine_eye, length=length
    )

    masker = MultiNiftiMapsMasker(
        maps_img, mask_img=mask21_img, resampling_target=None
    )
    with pytest.raises(ValueError):
        masker.fit()

    # Transform, with smoothing (smoke test)
    masker = MultiNiftiMapsMasker(
        maps_img, smoothing_fwhm=3, resampling_target=None
    )
    signals_list = masker.fit().transform(signals_input)
    for signals in signals_list:
        assert signals.shape == (length, n_regions)

        with pytest.raises(ValueError, match="has not been fitted. "):
            MultiNiftiMapsMasker(maps_img).inverse_transform(signals)

    # Call inverse transform (smoke test)
    for signals in signals_list:
        fmri_img_r = masker.inverse_transform(signals)
        assert fmri_img_r.shape == fmri_img.shape
        assert_almost_equal(fmri_img_r.affine, fmri_img.affine)

    # Now try on a masker that has never seen the call to "transform"
    masker2 = MultiNiftiMapsMasker(maps_img, resampling_target=None)
    masker2.fit()
    masker2.inverse_transform(signals)

    # Test with data and atlas of different shape: the atlas should be
    # resampled to the data
    shape22 = (5, 5, 6)
    affine2 = 2 * np.eye(4)
    affine2[-1, -1] = 1

    fmri22_img, _ = data_gen.generate_fake_fmri(
        shape22, affine=affine2, length=length
    )
    masker = MultiNiftiMapsMasker(maps_img, mask_img=mask21_img)

    masker.fit_transform(fmri22_img)
    assert_array_equal(masker._resampled_maps_img_.affine, affine2)


def test_multi_nifti_maps_masker_resampling(
    shape_3d_default, affine_eye, n_regions, length
):
    # Test resampling in MultiNiftiMapsMasker
    shape2 = (13, 14, 15)  # mask
    shape3 = (16, 17, 18)  # maps

    _, mask22_img = data_gen.generate_fake_fmri(
        shape2, affine=affine_eye, length=length
    )

    maps33_img, _ = data_gen.generate_maps(
        shape3, n_regions, affine=affine_eye
    )

    mask_img_4d = Nifti1Image(
        np.ones((2, 2, 2, 2), dtype=np.int8), affine=np.diag((4, 4, 4, 1))
    )

    # verify that 4D mask arguments are refused
    masker = MultiNiftiMapsMasker(maps33_img, mask_img=mask_img_4d)
    with pytest.raises(
        DimensionError,
        match="Input data has incompatible dimensionality: "
        "Expected dimension is 3D and you provided "
        "a 4D image.",
    ):
        masker.fit()

    # Test error checking
    with pytest.raises(ValueError):
        MultiNiftiMapsMasker(maps33_img, resampling_target="mask")
    with pytest.raises(ValueError):
        MultiNiftiMapsMasker(
            maps33_img,
            resampling_target="invalid",
        )

    # Target: mask
    masker = MultiNiftiMapsMasker(
        maps33_img, mask_img=mask22_img, resampling_target="mask"
    )

    masker.fit()
    assert_almost_equal(masker.mask_img_.affine, mask22_img.affine)
    assert masker.mask_img_.shape == mask22_img.shape

    assert_almost_equal(masker.mask_img_.affine, masker.maps_img_.affine)
    assert masker.mask_img_.shape == masker.maps_img_.shape[:3]

    # Multi-subject example
    fmri_img, _ = data_gen.generate_fake_fmri(
        shape=shape_3d_default, affine=affine_eye, length=length
    )
    fmri_img = [fmri_img, fmri_img]

    transformed = masker.transform(fmri_img)
    for t in transformed:
        assert t.shape == (length, n_regions)

        fmri_img_r = masker.inverse_transform(t)
        assert_almost_equal(fmri_img_r.affine, masker.maps_img_.affine)
        assert fmri_img_r.shape == (masker.maps_img_.shape[:3] + (length,))


def test_multi_nifti_maps_masker_resampling_maps(
    shape_3d_default, affine_eye, n_regions, length
):
    shape2 = (13, 14, 15)  # mask
    shape3 = (16, 17, 18)  # maps

    _, mask22_img = data_gen.generate_fake_fmri(
        shape2, affine=affine_eye, length=length
    )

    maps33_img, _ = data_gen.generate_maps(
        shape3, n_regions, affine=affine_eye
    )

    # Target: maps
    masker = MultiNiftiMapsMasker(
        maps33_img, mask_img=mask22_img, resampling_target="maps"
    )

    masker.fit()
    assert_almost_equal(masker.maps_img_.affine, maps33_img.affine)
    assert masker.maps_img_.shape == maps33_img.shape

    assert_almost_equal(masker.mask_img_.affine, masker.maps_img_.affine)
    assert masker.mask_img_.shape == masker.maps_img_.shape[:3]

    # Multi-subject example
    fmri_img, _ = data_gen.generate_fake_fmri(
        shape=shape_3d_default, affine=affine_eye, length=length
    )
    fmri_img = [fmri_img, fmri_img]

    transformed = masker.transform(fmri_img)
    for t in transformed:
        assert t.shape == (length, n_regions)

        fmri_img_r = masker.inverse_transform(t)
        assert_almost_equal(fmri_img_r.affine, masker.maps_img_.affine)
        assert fmri_img_r.shape == (masker.maps_img_.shape[:3] + (length,))


def test_multi_nifti_maps_masker_resampling_clipped_maps(
    shape_3d_default, affine_eye, n_regions, length
):
    # Test with clipped maps: mask does not contain all maps.
    # Shapes do matter in that case
    shape2 = (8, 9, 10)  # mask
    affine2 = np.diag((2, 2, 2, 1))  # just for mask
    shape3 = (16, 18, 20)  # maps

    n_regions = 9
    length = 21

    _, mask22_img = data_gen.generate_fake_fmri(
        shape2, length=1, affine=affine2
    )
    # Target: maps
    maps33_img, _ = data_gen.generate_maps(
        shape3, n_regions, affine=affine_eye
    )

    masker = MultiNiftiMapsMasker(
        maps33_img, mask_img=mask22_img, resampling_target="maps"
    )

    masker.fit()
    assert_almost_equal(masker.maps_img_.affine, maps33_img.affine)
    assert masker.maps_img_.shape == maps33_img.shape

    assert_almost_equal(masker.mask_img_.affine, masker.maps_img_.affine)
    assert masker.mask_img_.shape == masker.maps_img_.shape[:3]

    # Multi-subject example
    fmri_img, _ = data_gen.generate_fake_fmri(
        shape=shape_3d_default, affine=affine_eye, length=length
    )
    fmri_img = [fmri_img, fmri_img]

    transformed = masker.transform(fmri_img)
    for t in transformed:
        assert t.shape == (length, n_regions)
        # Some regions have been clipped. Resulting signal must be zero
        assert (t.var(axis=0) == 0).sum() < n_regions

        fmri_img_r = masker.inverse_transform(t)
        assert_almost_equal(fmri_img_r.affine, masker.maps_img_.affine)
        assert fmri_img_r.shape == (masker.maps_img_.shape[:3] + (length,))
