"""Test the multi_nifti_maps_masker module."""

import numpy as np
import pytest
from nibabel import Nifti1Image

from nilearn._utils import data_gen, testing
from nilearn._utils.class_inspect import check_estimator
from nilearn._utils.exceptions import DimensionError
from nilearn.conftest import _shape_3d_default
from nilearn.maskers import MultiNiftiMapsMasker, NiftiMapsMasker

extra_valid_checks = [
    "check_get_params_invariance",
    "check_estimators_unfitted",
    "check_transformer_n_iter",
    "check_transformers_unfitted",
]


@pytest.mark.parametrize(
    "estimator, check, name",
    check_estimator(
        estimator=[
            MultiNiftiMapsMasker(
                data_gen.generate_maps(_shape_3d_default(), n_regions=9)[0]
            ),
            NiftiMapsMasker(
                data_gen.generate_maps(_shape_3d_default(), n_regions=9)[0]
            ),
        ],
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
        estimator=[
            MultiNiftiMapsMasker(
                data_gen.generate_maps(_shape_3d_default(), n_regions=9)[0]
            ),
            NiftiMapsMasker(
                data_gen.generate_maps(_shape_3d_default(), n_regions=9)[0]
            ),
        ],
        extra_valid_checks=extra_valid_checks,
        valid=False,
    ),
)
def test_check_estimator_invalid(estimator, check, name):  # noqa: ARG001
    """Check compliance with sklearn estimators."""
    check(estimator)


def test_multi_nifti_maps_masker(tmp_path):
    # Check working of shape/affine checks
    shape1 = (13, 11, 12)
    affine1 = np.eye(4)

    shape2 = (12, 10, 14)
    affine2 = np.diag((1, 2, 3, 1))

    n_regions = 9
    length = 3

    fmri11_img, mask11_img = data_gen.generate_fake_fmri(
        shape1, affine=affine1, length=length
    )
    fmri12_img, mask12_img = data_gen.generate_fake_fmri(
        shape1, affine=affine2, length=length
    )
    fmri21_img, mask21_img = data_gen.generate_fake_fmri(
        shape2, affine=affine1, length=length
    )

    maps11_img, _ = data_gen.generate_maps(shape1, n_regions, affine=affine1)

    # No exception raised here
    for create_files in (True, False):
        labels11 = testing.write_imgs_to_path(
            maps11_img, file_path=tmp_path, create_files=create_files
        )
        masker11 = MultiNiftiMapsMasker(labels11, resampling_target=None)
        signals11 = masker11.fit().transform(fmri11_img)
        assert signals11.shape == (length, n_regions)
        # enables to delete "labels11" on windows
        del masker11

    masker11 = MultiNiftiMapsMasker(
        maps11_img, mask_img=mask11_img, resampling_target=None
    )

    with pytest.raises(ValueError, match="has not been fitted. "):
        masker11.transform(fmri11_img)
    signals11 = masker11.fit().transform(fmri11_img)
    assert signals11.shape == (length, n_regions)

    MultiNiftiMapsMasker(maps11_img).fit_transform(fmri11_img)

    # Should work with 4D + 1D input too (also test fit_transform)
    signals_input = [fmri11_img, fmri11_img]
    signals11_list = masker11.fit_transform(signals_input)
    assert len(signals11_list) == len(signals_input)
    for signals in signals11_list:
        assert signals.shape == (length, n_regions)

    # NiftiMapsMasker should not work with 4D + 1D input
    signals_input = [fmri11_img, fmri11_img]
    masker11 = NiftiMapsMasker(maps11_img, resampling_target=None)
    with pytest.raises(DimensionError, match="incompatible dimensionality"):
        masker11.fit_transform(signals_input)

    # Test all kinds of mismatches between shapes and between affines
    for create_files in (True, False):
        images = testing.write_imgs_to_path(
            maps11_img,
            mask12_img,
            file_path=tmp_path,
            create_files=create_files,
        )
        labels11, mask12 = images
        masker11 = MultiNiftiMapsMasker(labels11, resampling_target=None)
        masker11.fit()
        with pytest.raises(ValueError):
            masker11.transform(fmri12_img)
        with pytest.raises(ValueError):
            masker11.transform(fmri21_img)

        masker11 = MultiNiftiMapsMasker(
            labels11, mask_img=mask12, resampling_target=None
        )
        with pytest.raises(ValueError):
            masker11.fit()
        del masker11

    masker11 = MultiNiftiMapsMasker(
        maps11_img, mask_img=mask21_img, resampling_target=None
    )
    with pytest.raises(ValueError):
        masker11.fit()

    # Transform, with smoothing (smoke test)
    masker11 = MultiNiftiMapsMasker(
        maps11_img, smoothing_fwhm=3, resampling_target=None
    )
    signals11_list = masker11.fit().transform(signals_input)
    for signals in signals11_list:
        assert signals.shape == (length, n_regions)

        with pytest.raises(ValueError, match="has not been fitted. "):
            MultiNiftiMapsMasker(maps11_img).inverse_transform(signals)

    # Call inverse transform (smoke test)
    for signals in signals11_list:
        fmri11_img_r = masker11.inverse_transform(signals)
        assert fmri11_img_r.shape == fmri11_img.shape
        np.testing.assert_almost_equal(fmri11_img_r.affine, fmri11_img.affine)

    # Now try on a masker that has never seen the call to "transform"
    masker2 = MultiNiftiMapsMasker(maps11_img, resampling_target=None)
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
    masker = MultiNiftiMapsMasker(maps11_img, mask_img=mask21_img)

    masker.fit_transform(fmri22_img)
    np.testing.assert_array_equal(masker._resampled_maps_img_.affine, affine2)


def test_multi_nifti_maps_masker_resampling():
    # Test resampling in MultiNiftiMapsMasker
    affine = np.eye(4)

    shape1 = (10, 11, 12)  # fmri
    shape2 = (13, 14, 15)  # mask
    shape3 = (16, 17, 18)  # maps

    n_regions = 9
    length = 3

    fmri11_img, _ = data_gen.generate_fake_fmri(
        shape1, affine=affine, length=length
    )
    _, mask22_img = data_gen.generate_fake_fmri(
        shape2, affine=affine, length=length
    )

    maps33_img, _ = data_gen.generate_maps(shape3, n_regions, affine=affine)

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

    # Multi-subject example
    fmri11_img = [fmri11_img, fmri11_img]

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
    np.testing.assert_almost_equal(masker.mask_img_.affine, mask22_img.affine)
    assert masker.mask_img_.shape == mask22_img.shape

    np.testing.assert_almost_equal(
        masker.mask_img_.affine, masker.maps_img_.affine
    )
    assert masker.mask_img_.shape == masker.maps_img_.shape[:3]

    transformed = masker.transform(fmri11_img)
    for t in transformed:
        assert t.shape == (length, n_regions)

        fmri11_img_r = masker.inverse_transform(t)
        np.testing.assert_almost_equal(
            fmri11_img_r.affine, masker.maps_img_.affine
        )
        assert fmri11_img_r.shape == (masker.maps_img_.shape[:3] + (length,))

    # Target: maps
    masker = MultiNiftiMapsMasker(
        maps33_img, mask_img=mask22_img, resampling_target="maps"
    )

    masker.fit()
    np.testing.assert_almost_equal(masker.maps_img_.affine, maps33_img.affine)
    assert masker.maps_img_.shape == maps33_img.shape

    np.testing.assert_almost_equal(
        masker.mask_img_.affine, masker.maps_img_.affine
    )
    assert masker.mask_img_.shape == masker.maps_img_.shape[:3]

    transformed = masker.transform(fmri11_img)
    for t in transformed:
        assert t.shape == (length, n_regions)

        fmri11_img_r = masker.inverse_transform(t)
        np.testing.assert_almost_equal(
            fmri11_img_r.affine, masker.maps_img_.affine
        )
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

    fmri11_img, _ = data_gen.generate_fake_fmri(
        shape1, affine=affine1, length=length
    )
    _, mask22_img = data_gen.generate_fake_fmri(
        shape2, length=1, affine=affine2
    )
    # Target: maps
    maps33_img, _ = data_gen.generate_maps(shape3, n_regions, affine=affine1)

    # Multi-subject example
    fmri11_img = [fmri11_img, fmri11_img]

    masker = MultiNiftiMapsMasker(
        maps33_img, mask_img=mask22_img, resampling_target="maps"
    )

    masker.fit()
    np.testing.assert_almost_equal(masker.maps_img_.affine, maps33_img.affine)
    assert masker.maps_img_.shape == maps33_img.shape

    np.testing.assert_almost_equal(
        masker.mask_img_.affine, masker.maps_img_.affine
    )
    assert masker.mask_img_.shape == masker.maps_img_.shape[:3]

    transformed = masker.transform(fmri11_img)
    for t in transformed:
        assert t.shape == (length, n_regions)
        # Some regions have been clipped. Resulting signal must be zero
        assert (t.var(axis=0) == 0).sum() < n_regions

        fmri11_img_r = masker.inverse_transform(t)
        np.testing.assert_almost_equal(
            fmri11_img_r.affine, masker.maps_img_.affine
        )
        assert fmri11_img_r.shape == (masker.maps_img_.shape[:3] + (length,))


def test_multi_nifti_maps_masker_list_of_sample_mask():
    """Tests MultiNiftiMapsMasker.fit_transform with a list of "sample_mask".

    "sample_mask" was directly sent as input to the parallel calls of
    "transform_single_imgs" instead of sending iterations.
    See https://github.com/nilearn/nilearn/issues/3967 for more details.
    """
    shape1 = (13, 11, 12)
    affine1 = np.eye(4)

    n_regions = 9
    length = 6
    n_scrub1 = 3
    n_scrub2 = 2

    fmri11_img, mask11_img = data_gen.generate_fake_fmri(
        shape1, affine=affine1, length=length
    )

    maps11_img, _ = data_gen.generate_maps(shape1, n_regions, affine=affine1)
    sample_mask1 = np.arange(length - n_scrub1)
    sample_mask2 = np.arange(length - n_scrub2)

    masker = MultiNiftiMapsMasker(maps11_img)
    ts_list = masker.fit_transform(
        [fmri11_img, fmri11_img], sample_mask=[sample_mask1, sample_mask2]
    )

    assert len(ts_list) == 2
    for ts, n_scrub in zip(ts_list, [n_scrub1, n_scrub2]):
        assert ts.shape == (length - n_scrub, n_regions)
