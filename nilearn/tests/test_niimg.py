import os
import numpy as np
import nibabel as nib

from nilearn._utils.testing import assert_raises_regex
from nilearn._utils import niimg

currdir = os.path.dirname(os.path.abspath(__file__))
datadir = os.path.join(currdir, 'data')


def test_copy_img():
    assert_raises_regex(ValueError, "Input value is not an image",
        niimg.copy_img, 3)


def test_new_img_like_mgz():
    ref_img = nib.load(os.path.join(datadir, 'test.mgz'))
    data = np.ones(ref_img.get_data().shape, dtype=np.bool)
    affine = ref_img.get_affine()
    new_img = niimg.new_img_like(ref_img, data, affine, copy_header=False)
