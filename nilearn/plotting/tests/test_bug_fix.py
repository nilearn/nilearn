import nibabel as nib
import numpy as np
import pytest

from nilearn import plotting


def test_out_of_bounds_error():
    """Ensure plotting out-of-bounds slice raises
    ValueError."""
    # Create a small dummy image (10x10x10 voxels)
    data = np.zeros((10, 10, 10))
    img = nib.Nifti1Image(data, np.eye(4))

    # This should crash with ValueError because 50 is > 10
    # We use pytest.raises to "catch" the crash
    #  and mark the test as PASS if it crashes correctly
    with pytest.raises(ValueError, match="out of bounds"):
        plotting.plot_img(img, display_mode="x", cut_coords=[50])
