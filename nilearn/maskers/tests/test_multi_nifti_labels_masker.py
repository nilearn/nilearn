"""Test the multi_nifti_labels_masker module."""

import pytest
from numpy.testing import assert_almost_equal

from nilearn._utils.data_gen import generate_fake_fmri
from nilearn.maskers import MultiNiftiLabelsMasker


def test_multi_nifti_labels_masker(
    affine_eye, n_regions, shape_3d_default, length, img_labels
):
    """Check working of shape/affine checks."""
    fmri11_img, mask11_img = generate_fake_fmri(
        shape_3d_default, affine=affine_eye, length=length
    )

    masker11 = MultiNiftiLabelsMasker(img_labels, resampling_target=None)

    # No exception raised here
    signals11 = masker11.fit_transform(fmri11_img)

    assert signals11.shape == (length, n_regions)

    # No exception should be raised either
    masker11 = MultiNiftiLabelsMasker(img_labels, resampling_target=None)
    masker11.fit()
    masker11.inverse_transform(signals11)

    # the mask removes some regions
    masker11 = MultiNiftiLabelsMasker(
        img_labels, mask_img=mask11_img, resampling_target=None
    )
    with pytest.warns(
        UserWarning,
        match=r"After applying mask to the labels .*only contains 5 labels",
    ):
        signals11 = masker11.fit_transform(fmri11_img)

    n_regions_left = 4
    assert signals11.shape == (length, n_regions_left)

    # Should work with 4D + 1D input too (also test fit_transform)
    signals_input = [fmri11_img, fmri11_img]
    signals11_list = masker11.fit_transform(signals_input)

    for signals in signals11_list:
        assert signals.shape == (length, n_regions_left)

    masker11 = MultiNiftiLabelsMasker(img_labels, resampling_target=None)
    signals11_list = masker11.fit_transform(signals_input)

    # Call inverse transform (smoke test)
    for signals in signals11_list:
        assert signals.shape == (length, n_regions)
        fmri11_img_r = masker11.inverse_transform(signals)

        assert fmri11_img_r.shape == fmri11_img.shape
        assert_almost_equal(fmri11_img_r.affine, fmri11_img.affine)
