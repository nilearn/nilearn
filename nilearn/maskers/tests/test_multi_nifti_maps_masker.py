"""
Test the multi_nifti_maps_masker module
"""

import numpy as np

import pytest

from nilearn._utils import testing, data_gen
from nilearn._utils.exceptions import DimensionError
from nilearn.maskers import NiftiMapsMasker, MultiNiftiMapsMasker


def test_multi_nifti_maps_masker():
    # Check working of shape/affine checks
    shape1 = (13, 11, 12)
    affine1 = np.eye(4)

    shape2 = (12, 10, 14)
    affine2 = np.diag((1, 2, 3, 1))

    n_regions = 9
    length = 3

    fmri11_img, mask11_img = data_gen.generate_fake_fmri(shape1,
                                                         affine=affine1,
                                                         length=length)
    fmri12_img, mask12_img = data_gen.generate_fake_fmri(shape1,
                                                         affine=affine2,
                                                         length=length)
    fmri21_img, mask21_img = data_gen.generate_fake_fmri(shape2,
                                                         affine=affine1,
                                                         length=length)

    labels11_img, labels_mask_img = data_gen.generate_maps(shape1, n_regions,
                                                           affine=affine1)

    # No exception raised here
    for create_files in (True, False):
        with testing.write_tmp_imgs(labels11_img,
                                    create_files=create_files) as labels11:
            masker11 = MultiNiftiMapsMasker(labels11, resampling_target=None)
            signals11 = masker11.fit().transform(fmri11_img)
            assert signals11.shape == (length, n_regions)
            # enables to delete "labels11" on windows
            del masker11

    masker11 = MultiNiftiMapsMasker(labels11_img, mask_img=mask11_img,
                                    resampling_target=None)

    with pytest.raises(ValueError, match='has not been fitted. '):
        masker11.transform(fmri11_img)
    signals11 = masker11.fit().transform(fmri11_img)
    assert signals11.shape == (length, n_regions)

    MultiNiftiMapsMasker(labels11_img).fit_transform(fmri11_img)

    # Should work with 4D + 1D input too (also test fit_transform)
    signals_input = [fmri11_img, fmri11_img]
    signals11_list = masker11.fit_transform(signals_input)
    assert len(signals11_list) == len(signals_input)
    for signals in signals11_list:
        assert signals.shape == (length, n_regions)

    # NiftiMapsMasker should not work with 4D + 1D input
    signals_input = [fmri11_img, fmri11_img]
    masker11 = NiftiMapsMasker(labels11_img, resampling_target=None)
    with pytest.raises(DimensionError, match="incompatible dimensionality"):
        masker11.fit_transform(signals_input)

    # Test all kinds of mismatches between shapes and between affines
    for create_files in (True, False):
        with testing.write_tmp_imgs(labels11_img, mask12_img,
                                    create_files=create_files) as images:
            labels11, mask12 = images
            masker11 = MultiNiftiMapsMasker(labels11, resampling_target=None)
            masker11.fit()
            pytest.raises(ValueError, masker11.transform, fmri12_img)
            pytest.raises(ValueError, masker11.transform, fmri21_img)

            masker11 = MultiNiftiMapsMasker(labels11, mask_img=mask12,
                                            resampling_target=None)
            pytest.raises(ValueError, masker11.fit)
            del masker11

    masker11 = MultiNiftiMapsMasker(labels11_img, mask_img=mask21_img,
                                    resampling_target=None)
    pytest.raises(ValueError, masker11.fit)

    # Transform, with smoothing (smoke test)
    masker11 = MultiNiftiMapsMasker(labels11_img, smoothing_fwhm=3,
                                    resampling_target=None)
    signals11_list = masker11.fit().transform(signals_input)
    for signals in signals11_list:
        assert signals.shape == (length, n_regions)

        with pytest.raises(ValueError, match='has not been fitted. '):
            MultiNiftiMapsMasker(labels11_img).inverse_transform(signals)

    # Call inverse transform (smoke test)
    for signals in signals11_list:
        fmri11_img_r = masker11.inverse_transform(signals)
        assert fmri11_img_r.shape == fmri11_img.shape
        np.testing.assert_almost_equal(fmri11_img_r.affine, fmri11_img.affine)

    # Now try on a masker that has never seen the call to "transform"
    masker2 = MultiNiftiMapsMasker(labels11_img, resampling_target=None)
    masker2.fit()
    masker2.inverse_transform(signals)

    # Test with data and atlas of different shape: the atlas should be
    # resampled to the data
    shape22 = (5, 5, 6)
    affine2 = 2 * np.eye(4)
    affine2[-1, -1] = 1

    fmri22_img, _ = data_gen.generate_fake_fmri(shape22, affine=affine2,
                                                length=length)
    masker = MultiNiftiMapsMasker(labels11_img, mask_img=mask21_img)

    masker.fit_transform(fmri22_img)
    np.testing.assert_array_equal(masker._resampled_maps_img_.affine,
                                  affine2)
