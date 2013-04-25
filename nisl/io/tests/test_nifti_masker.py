"""
Test the nifti_masker module

Functions in this file only test features added by the NiftiMasker class,
not the underlying functions used (e.g. clean()). See test_masking.py and
test_signal.py for this.
"""
# Author: Gael Varoquaux, Philippe Gervais
# License: simplified BSD

from nose.tools import assert_true, assert_false, assert_raises
import numpy as np

from nibabel import Nifti1Image

from ..nifti_masker import NiftiMasker


def generate_fake_fmri(shape=(10, 11, 12), length=17, kind="noise"):
    """Generate a signal which can be used for testing.

    The return value is a 4D array, representing 3D volumes along time.
    Only the voxels in the center are non-zero, to mimic the presence of
    brain voxels in real signals.

    Parameters
    ==========
    shape (tuple, optional)
        Shape of 3D volume
    length (integer, optional)
        Number of time instants
    kind (string, optional)
        Kind of signal used as timeseries.
        "noise": uniformly sampled values in [0..255]
        "step": 0.5 for the first half then 1.

    Returns
    =======
    fmri (nibabel.Nifti1Image)
        fake fmri signal.
        shape: shape + (length,)
    mask (nibabel.Nifti1Image)
        mask giving non-zero voxels
    """
    full_shape = shape + (length, )
    fmri = np.zeros(full_shape)
    # Fill central voxels timeseries with random signals
    rand_gen = np.random.RandomState(0)
    width = [s / 2 for s in shape]
    shift = [s / 4 for s in shape]

    if kind == "noise":
        signals = rand_gen.randint(256, size=(width + [length]))
    elif kind == "step":
        signals = np.ones(width + [length])
        signals[..., :length / 2] = 0.5
    else:
        raise ValueError("Unhandled value for parameter 'kind'")

    fmri[shift[0]:shift[0] + width[0],
         shift[1]:shift[1] + width[1],
         shift[2]:shift[2] + width[2],
         :] = signals

    mask = np.zeros(shape)
    mask[shift[0]:shift[0] + width[0],
         shift[1]:shift[1] + width[1],
         shift[2]:shift[2] + width[2]] = 1
    return Nifti1Image(fmri, np.eye(4)), Nifti1Image(mask, np.eye(4))


def test_auto_mask():
    # This mostly a smoke test
    data = np.ones((9, 9, 9))
    data[3:-3, 3:-3, 3:-3] = 10
    img = Nifti1Image(data, np.eye(4))
    masker = NiftiMasker()
    # Check that if we have not fit the masker we get a intelligible
    # error
    assert_raises(ValueError, masker.transform, [img, ])
    # Smoke test the fit
    masker.fit(img)
    # Smoke test the transform
    masker.transform([img, ])


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
    masker = NiftiMasker()
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
    fmri, mask = generate_fake_fmri(shape=(40, 41, 42), kind="step")
    masker = NiftiMasker(mask=mask, standardize=True, detrend=True)
    masker.fit()
    timeseries = masker.transform(fmri)
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
