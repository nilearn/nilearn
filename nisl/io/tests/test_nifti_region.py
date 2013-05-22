"""Test the nifti_region module

Functions in this file only test features added by the NiftiLabelsMasker class,
not the underlying functions (clean(), img_to_signals_labels(), etc.). See
test_masking.py and test_signal.py for details.
"""

from nose.tools import assert_raises, assert_equal
import numpy as np

import nibabel

from ..nifti_region import NiftiLabelsMasker, NiftiMapsMasker
from ... import testing
from ... import utils


def generate_random_img(shape, length=1, affine=np.eye(4),
                        rand_gen=np.random.RandomState(0)):
    data = rand_gen.randn(*(shape + (length,)))
    return nibabel.Nifti1Image(data, affine), nibabel.Nifti1Image(
        utils.as_ndarray(data[..., 0] > 0.2, dtype=np.int8), affine)


def test_nifti_labels_masker():
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

    labels11_img = testing.generate_labeled_regions(shape1, affine=affine1,
                                                    n_regions=n_regions)

    # No exception raised here
    masker11 = NiftiLabelsMasker(labels11_img)
    signals11 = masker11.fit().transform(fmri11_img)
    assert_equal(signals11.shape, (length, n_regions))

    masker11 = NiftiLabelsMasker(labels11_img, mask_img=mask11_img)
    signals11 = masker11.fit().transform(fmri11_img)
    assert_equal(signals11.shape, (length, n_regions))

    # Test all kinds of mismatch between shapes and between affines
    masker11 = NiftiLabelsMasker(labels11_img)
    masker11.fit()
    assert_raises(ValueError, masker11.transform, fmri12_img)
    assert_raises(ValueError, masker11.transform, fmri21_img)

    masker11 = NiftiLabelsMasker(labels11_img, mask_img=mask12_img)
    assert_raises(ValueError, masker11.fit)

    masker11 = NiftiLabelsMasker(labels11_img, mask_img=mask21_img)
    assert_raises(ValueError, masker11.fit)

    # Transform, with smoothing (smoke test)
    masker11 = NiftiLabelsMasker(labels11_img, smoothing_fwhm=3)
    signals11 = masker11.fit().transform(fmri11_img)
    assert_equal(signals11.shape, (length, n_regions))

    masker11 = NiftiLabelsMasker(labels11_img, smoothing_fwhm=3)
    signals11 = masker11.fit_transform(fmri11_img)
    assert_equal(signals11.shape, (length, n_regions))

    # Call inverse transform (smoke test)
    fmri11_img_r = masker11.inverse_transform(signals11)
    assert_equal(fmri11_img_r.shape, fmri11_img.shape)
    np.testing.assert_almost_equal(fmri11_img_r.get_affine(),
                                   fmri11_img.get_affine())


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
                  testing.generate_maps(shape1, n_regions, affine=affine1)

    # No exception raised here
    for create_files in (True, False):
        with testing.write_tmp_imgs(labels11_img, create_files=create_files) \
                as labels11:
            masker11 = NiftiMapsMasker(labels11)
            signals11 = masker11.fit().transform(fmri11_img)
            assert_equal(signals11.shape, (length, n_regions))

    masker11 = NiftiMapsMasker(labels11_img, mask_img=mask11_img)
    signals11 = masker11.fit().transform(fmri11_img)
    assert_equal(signals11.shape, (length, n_regions))

    # Test all kinds of mismatches between shapes and between affines
    for create_files in (True, False):
        with testing.write_tmp_imgs(labels11_img, mask12_img,
                                    create_files=create_files) as images:
            labels11, mask12 = images
            masker11 = NiftiMapsMasker(labels11)
            masker11.fit()
            assert_raises(ValueError, masker11.transform, fmri12_img)
            assert_raises(ValueError, masker11.transform, fmri21_img)

            masker11 = NiftiMapsMasker(labels11, mask_img=mask12)
            assert_raises(ValueError, masker11.fit)

    masker11 = NiftiMapsMasker(labels11_img, mask_img=mask21_img)
    assert_raises(ValueError, masker11.fit)

    # Transform, with smoothing (smoke test)
    masker11 = NiftiMapsMasker(labels11_img, smoothing_fwhm=3)
    signals11 = masker11.fit().transform(fmri11_img)
    assert_equal(signals11.shape, (length, n_regions))

    masker11 = NiftiMapsMasker(labels11_img, smoothing_fwhm=3)
    signals11 = masker11.fit_transform(fmri11_img)
    assert_equal(signals11.shape, (length, n_regions))

    # Call inverse transform (smoke test)
    fmri11_img_r = masker11.inverse_transform(signals11)
    assert_equal(fmri11_img_r.shape, fmri11_img.shape)
    np.testing.assert_almost_equal(fmri11_img_r.get_affine(),
                                   fmri11_img.get_affine())


def test_nifti_maps_masker_2():
    # Test resampling in NiftiMapsMasker
    shape1 = (10, 11, 12)
    affine = np.eye(4)

    # mask
    shape2 = (16, 17, 18)

    # maps
    shape3 = (13, 14, 15)

    n_regions = 9
    length = 3

    fmri11_img, _ = generate_random_img(shape1, affine=affine,
                                                 length=length)
    _, mask22_img = generate_random_img(shape2, affine=affine,
                                                 length=length)

    maps33_img, _ = \
                  testing.generate_maps(shape3, n_regions, affine=affine)

    # Test error checking
    assert_raises(ValueError, NiftiMapsMasker, maps33_img, target="mask")
    assert_raises(ValueError, NiftiMapsMasker, maps33_img, target="invalid")

    # Target: mask
    masker = NiftiMapsMasker(maps33_img, mask_img=mask22_img, target="mask")

    masker.fit()
    np.testing.assert_almost_equal(masker.mask_img_.get_affine(),
                                   mask22_img.get_affine())
    assert_equal(masker.mask_img_.shape, mask22_img.shape)

    np.testing.assert_almost_equal(masker.mask_img_.get_affine(),
                                   masker.maps_img_.get_affine())
    assert_equal(masker.mask_img_.shape, masker.maps_img_.shape[:3])

    transformed = masker.transform(fmri11_img)
    assert_equal(transformed.shape, (length, n_regions))

    fmri11_img_r = masker.inverse_transform(transformed)
    np.testing.assert_almost_equal(fmri11_img_r.get_affine(),
                                   masker.maps_img_.get_affine())
    assert_equal(fmri11_img_r.shape, (masker.maps_img_.shape[:3] + (length,)))

    # Target: maps
    masker = NiftiMapsMasker(maps33_img, mask_img=mask22_img, target="maps")

    masker.fit()
    np.testing.assert_almost_equal(masker.maps_img_.get_affine(),
                                   maps33_img.get_affine())
    assert_equal(masker.maps_img_.shape, maps33_img.shape)

    np.testing.assert_almost_equal(masker.mask_img_.get_affine(),
                                   masker.maps_img_.get_affine())
    assert_equal(masker.mask_img_.shape, masker.maps_img_.shape[:3])

    transformed = masker.transform(fmri11_img)
    assert_equal(transformed.shape, (length, n_regions))

    fmri11_img_r = masker.inverse_transform(transformed)
    np.testing.assert_almost_equal(fmri11_img_r.get_affine(),
                                   masker.maps_img_.get_affine())
    assert_equal(fmri11_img_r.shape, (masker.maps_img_.shape[:3] + (length,)))

    # Test with clipped maps: mask does not contain all maps.
    shape1 = (10, 11, 12)
    shape2 = (13, 14, 15)
    shape3 = (16, 17, 18)

    n_regions = 9
    length = 21

    fmri11_img, _ = generate_random_img(shape1, affine=affine,
                                                 length=length)
    _, mask22_img = generate_random_img(shape2, affine=affine,
                                                 length=length)
    # Target: maps
    maps33_img, _ = \
                  testing.generate_maps(shape3, n_regions, affine=affine)
    masker = NiftiMapsMasker(maps33_img, mask_img=mask22_img, target="maps")

    masker.fit()
    np.testing.assert_almost_equal(masker.maps_img_.get_affine(),
                                   maps33_img.get_affine())
    assert_equal(masker.maps_img_.shape, maps33_img.shape)

    np.testing.assert_almost_equal(masker.mask_img_.get_affine(),
                                   masker.maps_img_.get_affine())
    assert_equal(masker.mask_img_.shape, masker.maps_img_.shape[:3])

    transformed = masker.transform(fmri11_img)
    assert_equal(transformed.shape[0], length)
    assert(transformed.shape[1] < n_regions)

    fmri11_img_r = masker.inverse_transform(transformed)
    np.testing.assert_almost_equal(fmri11_img_r.get_affine(),
                                   masker.maps_img_.get_affine())
    assert_equal(fmri11_img_r.shape,
                 (masker.maps_img_.shape[:3] + (length,)))

