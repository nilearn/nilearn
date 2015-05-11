import os
from distutils.version import LooseVersion
import numpy as np
import nibabel
from nose import SkipTest

from nilearn._utils.testing import assert_raises_regex
from nilearn._utils import niimg

currdir = os.path.dirname(os.path.abspath(__file__))
datadir = os.path.join(currdir, 'data')


def test_copy_img():
    assert_raises_regex(ValueError, "Input value is not an image",
                        niimg.copy_img, 3)


def test_new_img_like_mgz():
    """Check that new images can be generated with bool MGZ type
    This is usually when computing masks using MGZ inputs, e.g.
    when using plot_stap_map
    """

    if not LooseVersion(nibabel.__version__) >= LooseVersion('1.2.0'):
        # Old nibabel do not support MGZ files
        raise SkipTest

    ref_img = nibabel.load(os.path.join(datadir, 'test.mgz'))
    data = np.ones(ref_img.get_data().shape, dtype=np.bool)
    affine = ref_img.get_affine()
    niimg.new_img_like(ref_img, data, affine, copy_header=False)
