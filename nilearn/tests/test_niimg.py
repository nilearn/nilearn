import os

from nilearn._utils.testing import assert_raises_regex
from nilearn._utils import niimg

currdir = os.path.dirname(os.path.abspath(__file__))
datadir = os.path.join(currdir, 'data')


def test_copy_img():
    assert_raises_regex(ValueError, "Input value is not an image",
                        niimg.copy_img, 3)
