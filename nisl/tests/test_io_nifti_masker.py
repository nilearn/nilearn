"""
Test for the NiftiMasker class

Function in this file only test features added by the NiftiMasker class,
not the underlying functions used (e.g. clean()). See test_masking.py and
test_signal.py for this.
"""

import numpy as np

#from nose.tools import assert_raises
from ..io import NiftiMasker
from nibabel import Nifti1Image


def generate_fake_fmri(shape=(10, 11, 12), length=17, kind="noise"):
    """Generate a signal which can be used for testing.

    The return value is a 4D array, representing 3D volumes along time.
    Only the voxels in the center are non-zero, to mimic the presence of
    brain voxels in real signals.

    Parameters
    ==========
    shape (tuple)
        Shape of 3D volume
    length (integer)
        Number of time instants

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
    randgen = np.random.RandomState(0)
    width = [s / 2 for s in shape]
    shift = [s / 4 for s in shape]

    if kind == "noise":
        signals = randgen.randint(256, size=(width + [length]))
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


def test_matrix_orientation():
    """Test if processing is performed along the correct axis."""

    # the "step" kind generate heavyside-like signals for each voxel.
    # all signals being identical, standardizing along the wrong axis
    # would leave a null signal. Along the correct axis, the step remains.
    fmri, mask = generate_fake_fmri(shape=(40, 41, 42), kind="step")
    masker = NiftiMasker(mask=mask, standardize=True)
    masker.fit()
    timeseries = masker.transform(fmri)
    assert(timeseries.shape[0] == fmri.shape[3])
    assert(timeseries.shape[1] == mask.get_data().sum())
    std = timeseries.std(axis=0)
    assert(std.shape[0] == timeseries.shape[1])  # paranoid
    assert(not np.any(std < 0.1))

    # Test inverse transform
    masker = NiftiMasker(mask=mask, standardize=False)
    masker.fit()
    timeseries = masker.transform(fmri)
    recovered = masker.inverse_transform(timeseries)
    np.testing.assert_array_almost_equal(recovered.get_data(), fmri.get_data())
