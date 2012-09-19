"""
Test the nifti_masker module
"""
import numpy as np

from nibabel import Nifti1Image

from ..nifti_masker import NiftiMasker

def test_auto_mask():
    data = np.ones((9, 9, 9))
    data[3:-3, 3:-3, 3:-3] = 10
    img = Nifti1Image(data, np.eye(4))
    masker = NiftiMasker()
    masker.fit(img)

