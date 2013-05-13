"""Test the nifti_region module

Functions in this file only test features added by the NiftiLabelsMasker class,
not the underlying functions (clean(), img_to_signals_labels(), etc.). See
test_masking.py and test_signal.py for details.
"""

from nose.tools import assert_true, assert_false, assert_raises
import numpy as np

import nibabel

from ..nifti_region import NiftiLabelsMasker
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
    assert_true(signals11.shape == (length, n_regions))

    masker11 = NiftiLabelsMasker(labels11_img, mask_img=mask11_img)
    signals11 = masker11.fit().transform(fmri11_img)
    assert_true(signals11.shape == (length, n_regions))

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
    masker11 = NiftiLabelsMasker(labels11_img, smooth=3)
    signals11 = masker11.fit().transform(fmri11_img)
    assert_true(signals11.shape == (length, n_regions))

    masker11 = NiftiLabelsMasker(labels11_img, smooth=3)
    signals11 = masker11.fit_transform(fmri11_img)
    assert_true(signals11.shape == (length, n_regions))

    # Call inverse transform (smoke test)
    fmri11_img_r = masker11.inverse_transform(signals11)
    assert_true(fmri11_img_r.shape == fmri11_img.shape)
    np.testing.assert_almost_equal(fmri11_img_r.get_affine(),
                                   fmri11_img.get_affine())
