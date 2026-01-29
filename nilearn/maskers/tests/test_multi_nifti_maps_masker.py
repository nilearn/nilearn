"""Test the multi_nifti_maps_masker module."""

import numpy as np
import pytest
from sklearn.utils.estimator_checks import parametrize_with_checks

from nilearn._utils.data_gen import generate_fake_fmri, generate_maps
from nilearn._utils.estimator_checks import (
    check_estimator,
    nilearn_check_estimator,
    return_expected_failed_checks,
)
from nilearn._utils.versions import SKLEARN_LT_1_6
from nilearn.conftest import _img_maps
from nilearn.maskers import MultiNiftiMapsMasker
from nilearn.maskers.tests.conftest import (
    check_nifti_maps_masker_post_fit,
    check_nifti_maps_masker_post_transform,
)

ESTIMATORS_TO_CHECK = [MultiNiftiMapsMasker(standardize=None)]

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
        check_estimator(
            estimators=ESTIMATORS_TO_CHECK,
            valid=False,
        ),
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


@pytest.mark.slow
@pytest.mark.parametrize(
    "estimator, check, name",
    nilearn_check_estimator(
        estimators=[
            # pass less than the default number of regions
            # to speed up the tests
            MultiNiftiMapsMasker(_img_maps(n_regions=2), standardize=None),
        ]
    ),
)
def test_check_estimator_nilearn(estimator, check, name):  # noqa: ARG001
    """Check compliance with nilearn estimators rules."""
    check(estimator)


@pytest.mark.slow
@pytest.mark.parametrize("keep_masked_maps", [True, False])
def test_multi_nifti_maps_masker(
    affine_eye, length, n_regions, shape_3d_default, img_maps, keep_masked_maps
):
    """Check basic functions of MultiNiftiMapsMasker.

    - fit, transform, fit_transform, inverse_transform.
    - 4D and list[4D] inputs
    """
    fmri11_img, mask11_img = generate_fake_fmri(
        shape_3d_default, affine=affine_eye, length=length
    )

    masker = MultiNiftiMapsMasker(
        img_maps,
        mask_img=mask11_img,
        resampling_target=None,
        keep_masked_maps=keep_masked_maps,
        standardize=None,
    )

    signals11 = masker.fit_transform(fmri11_img)

    expected_n_regions = n_regions
    if not keep_masked_maps:
        expected_n_regions = n_regions - 3

    check_nifti_maps_masker_post_transform(
        masker,
        expected_n_regions,
        signals11,
        length,
        ref_affine=affine_eye,
        ref_shape=shape_3d_default,
    )

    # Should work with 4D + 1D input too (also test fit_transform)
    signals_input = [fmri11_img, fmri11_img]

    if keep_masked_maps:
        # TODO (nilearn >=0.15)
        # only keep else block
        with pytest.warns(
            FutureWarning, match='"keep_masked_maps" parameter will be removed'
        ):
            signals11_list = masker.fit_transform(signals_input)
    else:
        signals11_list = masker.fit_transform(signals_input)

    check_nifti_maps_masker_post_transform(
        masker,
        expected_n_regions,
        signals11_list,
        length,
        ref_affine=affine_eye,
        ref_shape=shape_3d_default,
    )


@pytest.mark.slow
@pytest.mark.parametrize("keep_masked_maps", [True, False])
def test_transfer_on_another_masker(
    affine_eye, length, shape_3d_default, img_maps, n_regions, keep_masked_maps
):
    """Inverse transform signal obtained from another masker."""
    fmri11_img, _ = generate_fake_fmri(
        shape_3d_default, affine=affine_eye, length=length
    )

    masker = MultiNiftiMapsMasker(
        img_maps, standardize=None, keep_masked_maps=keep_masked_maps
    )

    signals_input = [fmri11_img, fmri11_img]

    signals11_list = masker.fit_transform(signals_input)

    check_nifti_maps_masker_post_transform(
        masker,
        n_regions,
        signals11_list,
        length,
        ref_affine=affine_eye,
        ref_shape=shape_3d_default,
    )

    # Now try on a masker that has never seen the call to "transform"
    masker = MultiNiftiMapsMasker(
        img_maps,
        resampling_target=None,
        standardize=None,
        keep_masked_maps=keep_masked_maps,
    )
    masker.fit()
    masker.inverse_transform(signals11_list[0])


@pytest.mark.parametrize("keep_masked_maps", [True, False])
def test_data_atlas_different_shape(
    affine_eye, length, img_maps, keep_masked_maps, n_regions
):
    """Test with data and atlas of different shape.

    The atlas should be resampled to the data.
    """
    # Check working of shape/affine checks
    shape2 = (12, 10, 14)
    shape22 = (5, 5, 6)
    affine2 = np.diag((1, 2, 3, 1))
    affine2 = 2 * np.eye(4)
    affine2[-1, -1] = 1

    _, mask21_img = generate_fake_fmri(
        shape2, affine=affine_eye, length=length
    )
    fmri22_img, _ = generate_fake_fmri(shape22, affine=affine2, length=length)

    masker = MultiNiftiMapsMasker(
        img_maps,
        mask_img=mask21_img,
        standardize=None,
        keep_masked_maps=keep_masked_maps,
    )

    masker.fit(fmri22_img)

    expected_n_regions = n_regions

    check_nifti_maps_masker_post_fit(
        masker, expected_n_regions, ref_affine=affine2, ref_shape=shape22
    )

    signals = masker.transform(fmri22_img)

    if not keep_masked_maps:
        expected_n_regions = n_regions - 7

    check_nifti_maps_masker_post_transform(
        masker,
        expected_n_regions,
        signals,
        length,
        ref_affine=affine2,
        ref_shape=shape22,
    )


@pytest.mark.slow
@pytest.mark.parametrize("keep_masked_maps", [True, False])
def test_resampling_to_mask(
    shape_mask,
    affine_eye,
    length,
    n_regions,
    shape_3d_large,
    img_fmri,
    keep_masked_maps,
):
    """Test resampling to mask in MultiNiftiMapsMasker."""
    _, mask22_img = generate_fake_fmri(
        shape_mask, affine=affine_eye, length=length
    )
    maps33_img, _ = generate_maps(shape_3d_large, n_regions, affine=affine_eye)

    masker = MultiNiftiMapsMasker(
        maps33_img,
        mask_img=mask22_img,
        resampling_target="mask",
        keep_masked_maps=keep_masked_maps,
        standardize=None,
    )

    if keep_masked_maps:
        # TODO (nilearn >=0.15)
        # only keep else block
        with pytest.warns(
            FutureWarning, match='"keep_masked_maps" parameter will be removed'
        ):
            signals = masker.fit_transform([img_fmri, img_fmri])
    else:
        signals = masker.fit_transform([img_fmri, img_fmri])

    expected_n_regions = n_regions
    if not keep_masked_maps:
        expected_n_regions = n_regions - 7

    check_nifti_maps_masker_post_transform(
        masker,
        expected_n_regions,
        signals,
        length,
        ref_affine=mask22_img.affine,
        ref_shape=mask22_img.shape,
    )


@pytest.mark.slow
@pytest.mark.parametrize("keep_masked_maps", [True, False])
def test_resampling_to_maps(
    shape_mask,
    affine_eye,
    length,
    n_regions,
    shape_3d_large,
    keep_masked_maps,
    img_fmri,
):
    """Test resampling to maps in MultiNiftiMapsMasker."""
    _, mask22_img = generate_fake_fmri(
        shape_mask, affine=affine_eye, length=length
    )
    maps33_img, _ = generate_maps(shape_3d_large, n_regions, affine=affine_eye)

    masker = MultiNiftiMapsMasker(
        maps33_img,
        mask_img=mask22_img,
        resampling_target="maps",
        keep_masked_maps=keep_masked_maps,
        standardize=None,
    )

    if keep_masked_maps:
        # TODO (nilearn >=0.15)
        # only keep else block
        with pytest.warns(
            FutureWarning, match='"keep_masked_maps" parameter will be removed'
        ):
            signals = masker.fit_transform([img_fmri, img_fmri])
    else:
        signals = masker.fit_transform([img_fmri, img_fmri])

    expected_n_regions = n_regions
    if not keep_masked_maps:
        expected_n_regions = n_regions - 7

    check_nifti_maps_masker_post_transform(
        masker,
        expected_n_regions,
        signals,
        length,
        ref_affine=maps33_img.affine,
        ref_shape=maps33_img.shape[:3],
    )


@pytest.mark.slow
@pytest.mark.parametrize("keep_masked_maps", [True, False])
def test_resampling_clipped_mask(
    affine_eye, length, n_regions, img_fmri, keep_masked_maps
):
    """Test with clipped maps: mask does not contain all maps."""
    # Shapes do matter in that case
    shape2 = (8, 9, 10)  # mask
    shape3 = (16, 18, 20)  # maps
    affine2 = np.diag((2, 2, 2, 1))  # just for mask

    _, mask22_img = generate_fake_fmri(shape2, length=1, affine=affine2)
    maps33_img, _ = generate_maps(shape3, n_regions, affine=affine_eye)

    masker = MultiNiftiMapsMasker(
        maps33_img,
        mask_img=mask22_img,
        resampling_target="maps",
        keep_masked_maps=keep_masked_maps,
        standardize=None,
    )

    if keep_masked_maps:
        # TODO (nilearn >=0.15)
        # only keep else block
        with pytest.warns(
            FutureWarning, match='"keep_masked_maps" parameter will be removed'
        ):
            signals = masker.fit_transform([img_fmri, img_fmri])
    else:
        signals = masker.fit_transform([img_fmri, img_fmri])

    expected_n_regions = n_regions
    if not keep_masked_maps:
        expected_n_regions = n_regions - 5

    check_nifti_maps_masker_post_transform(
        masker,
        expected_n_regions,
        signals,
        length,
        ref_affine=maps33_img.affine,
        ref_shape=maps33_img.shape[:3],
    )

    for t in signals:
        # Some regions have been clipped. Resulting signal must be zero
        assert (t.var(axis=0) == 0).sum() < n_regions
