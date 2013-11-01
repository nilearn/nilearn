"""
Test image pre-processing functions
"""
from nose.tools import assert_true

from .. import image
from ..._utils import testing
import nibabel
import numpy as np


def test_high_variance_confounds():
    # See also test_signals.test_high_variance_confounds()
    # There is only tests on what is added by image.high_variance_confounds()
    # compared to signal.high_variance_confounds()

    shape = (40, 41, 42)
    length = 17
    n_confounds = 10

    img, mask_img = testing.generate_fake_fmri(shape=shape, length=length)

    confounds1 = image.high_variance_confounds(img, mask_img=mask_img,
                                               percentile=10.,
                                               n_confounds=n_confounds)
    assert_true(confounds1.shape == (length, n_confounds))

    # No mask.
    confounds2 = image.high_variance_confounds(img, percentile=10.,
                                               n_confounds=n_confounds)
    assert_true(confounds2.shape == (length, n_confounds))


def test_smooth():
    # This function only checks added functionalities compared
    # to _smooth_array()
    shapes = ((10, 11, 12), (13, 14, 15))
    lengths = (17, 18)
    fwhm = (1., 2., 3.)

    img1, mask1 = testing.generate_fake_fmri(shape=shapes[0],
                                             length=lengths[0])
    img2, mask2 = testing.generate_fake_fmri(shape=shapes[1],
                                             length=lengths[1])

    for create_files in (False, True):
        with testing.write_tmp_imgs(img1, img2,
                                    create_files=create_files) as imgs:
            # List of images as input
            out = image.smooth(imgs, fwhm)
            assert_true(isinstance(out, list))
            assert_true(len(out) == 2)
            for o, s, l in zip(out, shapes, lengths):
                assert_true(o.shape == (s + (l,)))

            # Single image as input
            out = image.smooth(imgs[0], fwhm)
            assert_true(isinstance(out, nibabel.Nifti1Image))
            assert_true(out.shape == (shapes[0] + (lengths[0],)))


def test__crop_img_to():
    data = np.zeros((5, 6, 7))
    data[2:4, 1:5, 3:6] = 1
    affine = np.diag((4, 3, 2, 1))
    niimg = nibabel.Nifti1Image(data, affine=affine)

    slices = [slice(2, 4), slice(1, 5), slice(3, 6)]
    cropped_niimg = image._crop_img_to(niimg, slices, copy=False)

    new_origin = np.array((4, 3, 2)) * np.array((2, 1, 3))

    # check that correct part was extracted:
    assert_true((cropped_niimg.get_data() == 1).all())
    assert_true(cropped_niimg.shape == (2, 4, 3))

    # check that affine was adjusted correctly
    assert_true((cropped_niimg.get_affine()[:3, 3] == new_origin).all())

    # check that data was really not copied
    data[2:4, 1:5, 3:6] = 2
    assert_true((cropped_niimg.get_data() == 2).all())

    # check that copying works
    copied_cropped_niimg = image._crop_img_to(niimg, slices)
    data[2:4, 1:5, 3:6] = 1
    assert_true((copied_cropped_niimg.get_data() == 2).all())


def test_crop_img():
    data = np.zeros((5, 6, 7))
    data[2:4, 1:5, 3:6] = 1
    affine = np.diag((4, 3, 2, 1))
    niimg = nibabel.Nifti1Image(data, affine=affine)

    cropped_niimg = image.crop_img(niimg)

    # correction for padding with "-1"
    new_origin = np.array((4, 3, 2)) * np.array((2 - 1, 1 - 1, 3 - 1))

    # check that correct part was extracted:
    # This also corrects for padding
    assert_true((cropped_niimg.get_data()[1:-1, 1:-1, 1:-1] == 1).all())
    assert_true(cropped_niimg.shape == (2 + 2, 4 + 2, 3 + 2))

