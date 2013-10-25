"""
Test the nifti_masker module

Functions in this file only test features added by the NiftiMasker class,
not the underlying functions used (e.g. clean()). See test_masking.py and
test_signal.py for this.
"""
# Author: Gael Varoquaux, Philippe Gervais
# License: simplified BSD
from distutils.version import LooseVersion

from nose.tools import assert_true, assert_false, assert_raises
from nose import SkipTest
import numpy as np

from nibabel import Nifti1Image
import nibabel

from ..nifti_masker import NiftiMasker
from ..._utils import testing


def test_auto_mask():
    # This mostly a smoke test
    data = np.zeros((9, 9, 9))
    data[3:-3, 3:-3, 3:-3] = 10
    img = Nifti1Image(data, np.eye(4))
    masker = NiftiMasker()
    # Check that if we have not fit the masker we get a intelligible
    # error
    assert_raises(ValueError, masker.transform, [img, ])
    # Smoke test the fit
    masker.fit(img)
    # Smoke test the transform
    # With a 4D niimg
    masker.transform([img, ])
    # With a 3D niimg
    masker.transform(img)


def test_with_files():
    # Standard masking
    data = np.zeros((40, 40, 40, 2))
    data[20, 20, 20] = 1
    data_img = Nifti1Image(data, np.eye(4))

    with testing.write_tmp_imgs(data_img, create_files=True)\
                as filename:
        masker = NiftiMasker()
        masker.fit(filename)
        masker.transform(filename)


def test_nan():
    data = np.ones((9, 9, 9))
    data[0] = np.nan
    data[:, 0] = np.nan
    data[:, :, 0] = np.nan
    data[-1] = np.nan
    data[:, -1] = np.nan
    data[:, :, -1] = np.nan
    data[3:-3, 3:-3, 3:-3] = 10
    img = Nifti1Image(data, np.eye(4))
    masker = NiftiMasker(mask_opening=0)
    masker.fit(img)
    mask = masker.mask_img_.get_data()
    assert_true(mask[1:-1, 1:-1, 1:-1].all())
    assert_false(mask[0].any())
    assert_false(mask[:, 0].any())
    assert_false(mask[:, :, 0].any())
    assert_false(mask[-1].any())
    assert_false(mask[:, -1].any())
    assert_false(mask[:, :, -1].any())


def test_matrix_orientation():
    """Test if processing is performed along the correct axis."""

    # the "step" kind generate heavyside-like signals for each voxel.
    # all signals being identical, standardizing along the wrong axis
    # would leave a null signal. Along the correct axis, the step remains.
    fmri, mask = testing.generate_fake_fmri(shape=(40, 41, 42), kind="step")
    masker = NiftiMasker(mask=mask, standardize=True, detrend=True)
    timeseries = masker.fit_transform(fmri)
    assert(timeseries.shape[0] == fmri.shape[3])
    assert(timeseries.shape[1] == mask.get_data().sum())
    std = timeseries.std(axis=0)
    assert(std.shape[0] == timeseries.shape[1])  # paranoid
    assert(not np.any(std < 0.1))

    # Test inverse transform
    masker = NiftiMasker(mask=mask, standardize=False, detrend=False)
    masker.fit()
    timeseries = masker.transform(fmri)
    recovered = masker.inverse_transform(timeseries)
    np.testing.assert_array_almost_equal(recovered.get_data(), fmri.get_data())


def test_joblib_cache():
    if not LooseVersion(nibabel.__version__) > LooseVersion('1.1.0'):
        # Old nibabel do not pickle
        raise SkipTest
    from sklearn.externals.joblib import hash
    # Dummy mask
    data = np.zeros((40, 40, 40, 2))
    data[20, 20, 20] = 1
    data_img = Nifti1Image(data, np.eye(4))

    with testing.write_tmp_imgs(data_img, create_files=True)\
                as filename:
        masker = NiftiMasker(mask=filename)
        masker.fit()
        mask_hash = hash(masker.mask_img_)
        masker.mask_img_.get_data()
        assert_true(mask_hash == hash(masker.mask_img_))
