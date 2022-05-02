"""Test the multi_nifti_labels_masker module

Functions in this file only test features added by the MultiNiftiLabelsMasker
"""

import numpy as np

import nibabel
import pytest
from nilearn.maskers import NiftiLabelsMasker, MultiNiftiLabelsMasker
from nilearn._utils import as_ndarray, data_gen
from nilearn._utils.exceptions import DimensionError


def generate_random_img(shape, length=1, affine=np.eye(4),
                        rand_gen=np.random.RandomState(0)):
    data = rand_gen.standard_normal(size=(shape + (length,)))
    return nibabel.Nifti1Image(data, affine), nibabel.Nifti1Image(
        as_ndarray(data[..., 0] > 0.2, dtype=np.int8), affine)


def test_multi_nifti_labels_masker():
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

    labels11_img = data_gen.generate_labeled_regions(shape1, affine=affine1,
                                                     n_regions=n_regions)

    mask_img_4d = nibabel.Nifti1Image(np.ones((2, 2, 2, 2), dtype=np.int8),
                                      affine=np.diag((4, 4, 4, 1)))

    # verify that 4D mask arguments are refused
    masker = MultiNiftiLabelsMasker(labels11_img, mask_img=mask_img_4d)
    with pytest.raises(DimensionError,
                       match="Input data has incompatible dimensionality: "
                             "Expected dimension is 3D and you provided "
                             "a 4D image."):
        masker.fit()

    # check exception when transform() called without prior fit()
    masker11 = MultiNiftiLabelsMasker(labels11_img, resampling_target=None)
    with pytest.raises(ValueError, match='has not been fitted. '):
        masker11.transform(fmri11_img)

    # No exception raised here
    signals11 = masker11.fit().transform(fmri11_img)
    assert signals11.shape == (length, n_regions)

    # Should work with 4D + 1D input too (also test fit_transform)
    signals_input = [fmri11_img, fmri11_img]
    signals11_list = masker11.fit_transform(signals_input)
    assert len(signals11_list) == len(signals_input)
    for signals in signals11_list:
        assert signals.shape == (length, n_regions)

    # NiftiLabelsMasker should not work with 4D + 1D input
    signals_input = [fmri11_img, fmri11_img]
    masker11 = NiftiLabelsMasker(labels11_img, resampling_target=None)
    with pytest.raises(DimensionError, match="incompatible dimensionality"):
        masker11.fit_transform(signals_input)

    # No exception should be raised either
    masker11 = MultiNiftiLabelsMasker(labels11_img, resampling_target=None)
    masker11.fit()
    assert signals11.shape == (length, n_regions)

    masker11 = MultiNiftiLabelsMasker(labels11_img, mask_img=mask11_img,
                                      resampling_target=None)
    signals11_list = masker11.fit().transform(signals_input)
    for signals in signals11_list:
        assert signals.shape == (length, n_regions)

    # Test all kinds of mismatch between shapes and between affines
    masker11 = MultiNiftiLabelsMasker(labels11_img, resampling_target=None)
    masker11.fit()
    pytest.raises(ValueError, masker11.transform, fmri12_img)
    pytest.raises(ValueError, masker11.transform, fmri21_img)

    masker11 = MultiNiftiLabelsMasker(labels11_img, mask_img=mask12_img,
                                      resampling_target=None)
    pytest.raises(ValueError, masker11.fit)

    masker11 = MultiNiftiLabelsMasker(labels11_img, mask_img=mask21_img,
                                      resampling_target=None)
    pytest.raises(ValueError, masker11.fit)

    # Transform, with smoothing (smoke test)
    masker11 = MultiNiftiLabelsMasker(labels11_img, smoothing_fwhm=3,
                                      resampling_target=None)
    signals11_list = masker11.fit().transform(signals_input)
    for signals in signals11_list:
        assert signals.shape == (length, n_regions)

    masker11 = MultiNiftiLabelsMasker(labels11_img, smoothing_fwhm=3,
                                      resampling_target=None)
    signals11_list = masker11.fit_transform(signals_input)
    for signals in signals11_list:
        assert signals.shape == (length, n_regions)

        with pytest.raises(ValueError, match='has not been fitted. '):
            MultiNiftiLabelsMasker(labels11_img).inverse_transform(signals)

    # Call inverse transform (smoke test)
    for signals in signals11_list:
        fmri11_img_r = masker11.inverse_transform(signals)
        assert fmri11_img_r.shape == fmri11_img.shape
        np.testing.assert_almost_equal(fmri11_img_r.affine, fmri11_img.affine)
