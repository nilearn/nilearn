
from nose.tools import assert_equal, assert_true, assert_raises
import warnings
import os
import numpy as np
import nibabel
from sklearn.base import BaseEstimator

from nilearn.input_data import NiftiMasker
from nilearn._utils.testing import assert_warns
from nilearn._utils.param_validation import (MNI152_BRAIN_VOLUME,
                                             _get_mask_volume)

mni152_brain_mask = (
    "/usr/share/fsl/data/standard/MNI152_T1_1mm_brain_mask.nii.gz")


def test_get_mask_volume():
    # Test that hard-coded standard mask volume can be corrected computed
    if os.path.isfile(mni152_brain_mask):
        assert_equal(MNI152_BRAIN_VOLUME, _get_mask_volume(nibabel.load(
            mni152_brain_mask)))
    else:
        warnings.warn("Couldn't find %s (for testing)" % (mni152_brain_mask))

