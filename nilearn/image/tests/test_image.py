"""
Test image pre-processing functions
"""
from nose.tools import assert_true, assert_false, assert_equal, assert_raises
from nose import SkipTest

import platform
import os
import nibabel
from nibabel import Nifti1Image
import numpy as np
from numpy.testing import assert_array_equal, assert_allclose
from nilearn._utils.testing import assert_raises_regex
from nilearn._utils.exceptions import DimensionError

from nilearn import signal
from nilearn.image import image
from nilearn.image import resampling
from nilearn.image import concat_imgs
from nilearn._utils import testing, niimg_conversions, compat
from nilearn.image import new_img_like
from nilearn.image import threshold_img
from nilearn.image import iter_img
from nilearn.image import math_img
from nilearn.image import largest_connected_component_img

X64 = (platform.architecture()[0] == '64bit')

currdir = os.path.dirname(os.path.abspath(__file__))
datadir = os.path.join(currdir, 'data')


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


def test__fast_smooth_array():
    N = 4
    shape = (N, N, N)
    # hardcoded in _fast_smooth_array
    neighbor_weight = 0.2
    # 6 neighbors in 3D if you are not on an edge
    nb_neighbors_max = 6

    data = np.ones(shape)
    smooth_data = image._fast_smooth_array(data)

    # this contains the number of neighbors for each cell in the array
    nb_neighbors_arr = np.empty(shape)
    for (i, j, k), __ in np.ndenumerate(nb_neighbors_arr):
        nb_neighbors_arr[i, j, k] = (3 + (0 < i < N - 1) +
                                     (0 < j < N - 1) + (0 < k < N - 1))

    expected = ((1 + neighbor_weight * nb_neighbors_arr) /
                (1 + neighbor_weight * nb_neighbors_max))
    np.testing.assert_allclose(smooth_data, expected)


def test__smooth_array():
    """Test smoothing of images: _smooth_array()"""
    # Impulse in 3D
    data = np.zeros((40, 41, 42))
    data[20, 20, 20] = 1

    # fwhm divided by any test affine must be odd. Otherwise assertion below
    # will fail. ( 9 / 0.6 = 15 is fine)
    fwhm = 9
    test_affines = (np.eye(4), np.diag((1, 1, -1, 1)),
                    np.diag((.6, 1, .6, 1)))
    for affine in test_affines:
        filtered = image._smooth_array(data, affine,
                                         fwhm=fwhm, copy=True)
        assert_false(np.may_share_memory(filtered, data))

        # We are expecting a full-width at half maximum of
        # fwhm / voxel_size:
        vmax = filtered.max()
        above_half_max = filtered > .5 * vmax
        for axis in (0, 1, 2):
            proj = np.any(np.any(np.rollaxis(above_half_max,
                          axis=axis), axis=-1), axis=-1)
            np.testing.assert_equal(proj.sum(),
                                    fwhm / np.abs(affine[axis, axis]))

    # Check that NaNs in the data do not propagate
    data[10, 10, 10] = np.NaN
    filtered = image._smooth_array(data, affine, fwhm=fwhm,
                                   ensure_finite=True, copy=True)
    assert_true(np.all(np.isfinite(filtered)))

    # Check copy=False.
    for affine in test_affines:
        data = np.zeros((40, 41, 42))
        data[20, 20, 20] = 1
        image._smooth_array(data, affine, fwhm=fwhm, copy=False)

        # We are expecting a full-width at half maximum of
        # fwhm / voxel_size:
        vmax = data.max()
        above_half_max = data > .5 * vmax
        for axis in (0, 1, 2):
            proj = np.any(np.any(np.rollaxis(above_half_max,
                          axis=axis), axis=-1), axis=-1)
            np.testing.assert_equal(proj.sum(),
                                    fwhm / np.abs(affine[axis, axis]))

    # Check fwhm='fast'
    for affine in test_affines:
        np.testing.assert_equal(image._smooth_array(data, affine, fwhm='fast'),
                                image._fast_smooth_array(data))


def test_smooth_img():
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
            out = image.smooth_img(imgs, fwhm)
            assert_true(isinstance(out, list))
            assert_true(len(out) == 2)
            for o, s, l in zip(out, shapes, lengths):
                assert_true(o.shape == (s + (l,)))

            # Single image as input
            out = image.smooth_img(imgs[0], fwhm)
            assert_true(isinstance(out, nibabel.Nifti1Image))
            assert_true(out.shape == (shapes[0] + (lengths[0],)))


def test__crop_img_to():
    data = np.zeros((5, 6, 7))
    data[2:4, 1:5, 3:6] = 1
    affine = np.diag((4, 3, 2, 1))
    img = nibabel.Nifti1Image(data, affine=affine)

    slices = [slice(2, 4), slice(1, 5), slice(3, 6)]
    cropped_img = image._crop_img_to(img, slices, copy=False)

    new_origin = np.array((4, 3, 2)) * np.array((2, 1, 3))

    # check that correct part was extracted:
    assert_true((cropped_img.get_data() == 1).all())
    assert_true(cropped_img.shape == (2, 4, 3))

    # check that affine was adjusted correctly
    assert_true((compat.get_affine(cropped_img)[:3, 3] == new_origin).all())

    # check that data was really not copied
    data[2:4, 1:5, 3:6] = 2
    assert_true((cropped_img.get_data() == 2).all())

    # check that copying works
    copied_cropped_img = image._crop_img_to(img, slices)
    data[2:4, 1:5, 3:6] = 1
    assert_true((copied_cropped_img.get_data() == 2).all())


def test_crop_img():
    data = np.zeros((5, 6, 7))
    data[2:4, 1:5, 3:6] = 1
    affine = np.diag((4, 3, 2, 1))
    img = nibabel.Nifti1Image(data, affine=affine)

    cropped_img = image.crop_img(img)

    # correction for padding with "-1"
    new_origin = np.array((4, 3, 2)) * np.array((2 - 1, 1 - 1, 3 - 1))

    # check that correct part was extracted:
    # This also corrects for padding
    assert_true((cropped_img.get_data()[1:-1, 1:-1, 1:-1] == 1).all())
    assert_true(cropped_img.shape == (2 + 2, 4 + 2, 3 + 2))


def test_crop_threshold_tolerance():
    """Check to see whether crop can skip values that are extremely
    close to zero in a relative sense and will crop them away"""

    data = np.zeros([10, 14, 12])
    data[3:7, 3:7, 5:9] = 1.
    active_shape = (4 + 2, 4 + 2, 4 + 2)  # add padding

    # add an infinitesimal outside this block
    data[3, 3, 3] = 1e-12
    affine = np.eye(4)
    img = nibabel.Nifti1Image(data, affine=affine)

    cropped_img = image.crop_img(img)
    assert_true(cropped_img.shape == active_shape)


def test_mean_img():
    rng = np.random.RandomState(42)
    data1 = np.zeros((5, 6, 7))
    data2 = rng.rand(5, 6, 7)
    data3 = rng.rand(5, 6, 7, 3)
    affine = np.diag((4, 3, 2, 1))
    img1 = nibabel.Nifti1Image(data1, affine=affine)
    img2 = nibabel.Nifti1Image(data2, affine=affine)
    img3 = nibabel.Nifti1Image(data3, affine=affine)
    for imgs in ([img1, ],
                   [img1, img2],
                   [img2, img1, img2],
                   [img3, img1, img2],  # Mixture of 4D and 3D images
                  ):

        arrays = list()
        # Ground-truth:
        for img in imgs:
            img = img.get_data()
            if img.ndim == 4:
                img = np.mean(img, axis=-1)
            arrays.append(img)
        truth = np.mean(arrays, axis=0)

        mean_img = image.mean_img(imgs)
        assert_array_equal(compat.get_affine(mean_img), affine)
        assert_array_equal(mean_img.get_data(), truth)

        # Test with files
        with testing.write_tmp_imgs(*imgs) as imgs:
            mean_img = image.mean_img(imgs)
            assert_array_equal(compat.get_affine(mean_img), affine)
            if X64:
                assert_array_equal(mean_img.get_data(), truth)
            else:
                # We don't really understand but arrays are not
                # exactly equal on 32bit. Given that you can not do
                # much real world data analysis with nilearn on a
                # 32bit machine it is not worth investigating more
                assert_allclose(mean_img.get_data(), truth,
                                rtol=np.finfo(truth.dtype).resolution,
                                atol=0)


def test_mean_img_resample():
    # Test resampling in mean_img with a permutation of the axes
    rng = np.random.RandomState(42)
    data = rng.rand(5, 6, 7, 40)
    affine = np.diag((4, 3, 2, 1))
    img = nibabel.Nifti1Image(data, affine=affine)
    mean_img = nibabel.Nifti1Image(data.mean(axis=-1), affine=affine)

    target_affine = affine[:, [1, 0, 2, 3]]  # permutation of axes
    mean_img_with_resampling = image.mean_img(img,
                                              target_affine=target_affine)
    resampled_mean_image = resampling.resample_img(mean_img,
                                              target_affine=target_affine)
    assert_array_equal(resampled_mean_image.get_data(),
                       mean_img_with_resampling.get_data())
    assert_array_equal(compat.get_affine(resampled_mean_image),
                       compat.get_affine(mean_img_with_resampling))
    assert_array_equal(compat.get_affine(mean_img_with_resampling), target_affine)


def test_swap_img_hemispheres():
    # make sure input image data is not overwritten inside function
    data = np.random.randn(4, 5, 7)
    data_img = nibabel.Nifti1Image(data, np.eye(4))
    image.swap_img_hemispheres(data_img)
    np.testing.assert_array_equal(data_img.get_data(), data)

    # swapping operations work
    np.testing.assert_array_equal(  # one turn
        image.swap_img_hemispheres(data_img).get_data(),
        data[::-1])
    np.testing.assert_array_equal(  # two turns -> back to original data
        image.swap_img_hemispheres(
            image.swap_img_hemispheres(data_img)).get_data(),
        data)


def test_concat_imgs():
    assert_true(concat_imgs is niimg_conversions.concat_niimgs)


def test_index_img():
    img_3d = nibabel.Nifti1Image(np.ones((3, 4, 5)), np.eye(4))
    testing.assert_raises_regex(TypeError,
                                "Input data has incompatible dimensionality: "
                                "Expected dimension is 4D and you provided "
                                "a 3D image.",
                                image.index_img, img_3d, 0)

    affine = np.array([[1., 2., 3., 4.],
                       [5., 6., 7., 8.],
                       [9., 10., 11., 12.],
                       [0., 0., 0., 1.]])
    img_4d, _ = testing.generate_fake_fmri(affine=affine)

    fourth_dim_size = img_4d.shape[3]
    tested_indices = (list(range(fourth_dim_size)) +
                      [slice(2, 8, 2), [1, 2, 3, 2], [],
                       (np.arange(fourth_dim_size) % 3) == 1])
    for i in tested_indices:
        this_img = image.index_img(img_4d, i)
        expected_data_3d = img_4d.get_data()[..., i]
        assert_array_equal(this_img.get_data(),
                           expected_data_3d)
        assert_array_equal(compat.get_affine(this_img),
                           compat.get_affine(img_4d))

    for i in [fourth_dim_size, - fourth_dim_size - 1,
              [0, fourth_dim_size],
              np.repeat(True, fourth_dim_size + 1)]:
        testing.assert_raises_regex(
            IndexError,
            'out of bounds|invalid index|out of range|boolean index',
            image.index_img, img_4d, i)


def test_iter_img():
    img_3d = nibabel.Nifti1Image(np.ones((3, 4, 5)), np.eye(4))
    testing.assert_raises_regex(TypeError,
                                "Input data has incompatible dimensionality: "
                                "Expected dimension is 4D and you provided "
                                "a 3D image.",
                                image.iter_img, img_3d)

    affine = np.array([[1., 2., 3., 4.],
                       [5., 6., 7., 8.],
                       [9., 10., 11., 12.],
                       [0., 0., 0., 1.]])
    img_4d, _ = testing.generate_fake_fmri(affine=affine)

    for i, img in enumerate(image.iter_img(img_4d)):
        expected_data_3d = img_4d.get_data()[..., i]
        assert_array_equal(img.get_data(),
                           expected_data_3d)
        assert_array_equal(compat.get_affine(img),
                           compat.get_affine(img_4d))

    with testing.write_tmp_imgs(img_4d) as img_4d_filename:
        for i, img in enumerate(image.iter_img(img_4d_filename)):
            expected_data_3d = img_4d.get_data()[..., i]
            assert_array_equal(img.get_data(),
                               expected_data_3d)
            assert_array_equal(compat.get_affine(img),
                               compat.get_affine(img_4d))
        # enables to delete "img_4d_filename" on windows
        del img

    img_3d_list = list(image.iter_img(img_4d))
    for i, img in enumerate(image.iter_img(img_3d_list)):
        expected_data_3d = img_4d.get_data()[..., i]
        assert_array_equal(img.get_data(),
                           expected_data_3d)
        assert_array_equal(compat.get_affine(img),
                           compat.get_affine(img_4d))

    with testing.write_tmp_imgs(*img_3d_list) as img_3d_filenames:
        for i, img in enumerate(image.iter_img(img_3d_filenames)):
            expected_data_3d = img_4d.get_data()[..., i]
            assert_array_equal(img.get_data(),
                               expected_data_3d)
            assert_array_equal(compat.get_affine(img),
                               compat.get_affine(img_4d))
        # enables to delete "img_3d_filename" on windows
        del img


def test_new_img_like_mgz():
    """Check that new images can be generated with bool MGZ type
    This is usually when computing masks using MGZ inputs, e.g.
    when using plot_stap_map
    """

    ref_img = nibabel.load(os.path.join(datadir, 'test.mgz'))
    data = np.ones(ref_img.get_data().shape, dtype=np.bool)
    affine = compat.get_affine(ref_img)
    new_img_like(ref_img, data, affine, copy_header=False)


def test_new_img_like():
    # Give a list to new_img_like
    data = np.zeros((5, 6, 7))
    data[2:4, 1:5, 3:6] = 1
    affine = np.diag((4, 3, 2, 1))
    img = nibabel.Nifti1Image(data, affine=affine)
    img2 = new_img_like([img, ], data)
    np.testing.assert_array_equal(img.get_data(), img2.get_data())


def test_validity_threshold_value_in_threshold_img():
    shape = (6, 8, 10)
    maps, _ = testing.generate_maps(shape, n_regions=2)

    # testing to raise same error when threshold=None case
    testing.assert_raises_regex(ValueError,
                                "The input parameter 'threshold' is empty. ",
                                threshold_img, maps, threshold=None)

    invalid_threshold_values = ['90t%', 's%', 't', '0.1']
    name = 'threshold'
    for thr in invalid_threshold_values:
        testing.assert_raises_regex(ValueError,
                                    '{0}.+should be a number followed by '
                                    'the percent sign'.format(name),
                                    threshold_img, maps, threshold=thr)


def test_threshold_img():
    # to check whether passes with valid threshold inputs
    shape = (10, 20, 30)
    maps, _ = testing.generate_maps(shape, n_regions=4)
    affine = np.eye(4)
    mask_img = nibabel.Nifti1Image(np.ones((shape), dtype=np.int8), affine)

    for img in iter_img(maps):
        # when threshold is a float value
        thr_maps_img = threshold_img(img, threshold=0.8)
        # when we provide mask image
        thr_maps_percent = threshold_img(img, threshold=1, mask_img=mask_img)
        # when threshold is a percentile
        thr_maps_percent2 = threshold_img(img, threshold='2%')


def test_isnan_threshold_img_data():
    shape = (10, 10, 10)
    maps, _ = testing.generate_maps(shape, n_regions=2)
    data = maps.get_data()
    data[:, :, 0] = np.nan

    maps_img = nibabel.Nifti1Image(data, np.eye(4))
    # test threshold_img to converge properly when input image has nans.
    threshold_img(maps_img, threshold=0.8)


def test_math_img_exceptions():
    img1 = Nifti1Image(np.ones((10, 10, 10, 10)), np.eye(4))
    img2 = Nifti1Image(np.zeros((10, 20, 10, 10)), np.eye(4))
    img3 = Nifti1Image(np.ones((10, 10, 10, 10)), np.eye(4))
    img4 = Nifti1Image(np.ones((10, 10, 10, 10)), np.eye(4) * 2)

    formula = "np.mean(img1, axis=-1) - np.mean(img2, axis=-1)"
    # Images with different shapes should raise a ValueError exception.
    assert_raises_regex(ValueError,
                        "Input images cannot be compared",
                        math_img, formula, img1=img1, img2=img2)

    # Images with different affines should raise a ValueError exception.
    assert_raises_regex(ValueError,
                        "Input images cannot be compared",
                        math_img, formula, img1=img1, img2=img4)

    bad_formula = "np.toto(img1, axis=-1) - np.mean(img3, axis=-1)"
    assert_raises_regex(AttributeError,
                        "Input formula couldn't be processed",
                        math_img, bad_formula, img1=img1, img3=img3)


def test_math_img():
    img1 = Nifti1Image(np.ones((10, 10, 10, 10)), np.eye(4))
    img2 = Nifti1Image(np.zeros((10, 10, 10, 10)), np.eye(4))
    expected_result = Nifti1Image(np.ones((10, 10, 10)), np.eye(4))

    formula = "np.mean(img1, axis=-1) - np.mean(img2, axis=-1)"
    for create_files in (True, False):
        with testing.write_tmp_imgs(img1, img2,
                                    create_files=create_files) as imgs:
            result = math_img(formula, img1=imgs[0], img2=imgs[1])
            assert_array_equal(result.get_data(),
                               expected_result.get_data())
            assert_array_equal(compat.get_affine(result),
                               compat.get_affine(expected_result))
            assert_equal(result.shape, expected_result.shape)


def test_clean_img():

    rng = np.random.RandomState(0)

    data = rng.randn(10, 10, 10, 100) + .5
    data_flat = data.T.reshape(100, -1)
    data_img = nibabel.Nifti1Image(data, np.eye(4))

    data_img_ = image.clean_img(
        data_img, detrend=True, standardize=False, low_pass=0.1)
    data_flat_ = signal.clean(
        data_flat, detrend=True, standardize=False, low_pass=0.1)

    np.testing.assert_almost_equal(data_img_.get_data().T.reshape(100, -1),
                                   data_flat_)
    # if NANs
    data[:, 9, 9] = np.nan
    # if infinity
    data[:, 5, 5] = np.inf
    nan_img = nibabel.Nifti1Image(data, np.eye(4))
    clean_im = image.clean_img(nan_img, ensure_finite=True)
    assert_true(np.any(np.isfinite(clean_im.get_data())), True)


def test_largest_cc_img():
    """ Check the extraction of the largest connected component, for niftis

    Similiar to smooth_img tests for largest connected_component_img, here also
    only the added features for largest_connected_component are tested.
    """

    # Test whether dimension of 3Dimg and list of 3Dimgs are kept.
    shapes = ((10, 11, 12), (13, 14, 15))
    regions = [1, 3]

    img1 = testing.generate_labeled_regions(shape=shapes[0],
                                            n_regions=regions[0])
    img2 = testing.generate_labeled_regions(shape=shapes[1],
                                            n_regions=regions[1])

    for create_files in (False, True):
        with testing.write_tmp_imgs(img1, img2,
                                    create_files=create_files) as imgs:
            # List of images as input
            out = largest_connected_component_img(imgs)
            assert_true(isinstance(out, list))
            assert_true(len(out) == 2)
            for o, s in zip(out, shapes):
                assert_true(o.shape == (s))

            # Single image as input
            out = largest_connected_component_img(imgs[0])
            assert_true(isinstance(out, Nifti1Image))
            assert_true(out.shape == (shapes[0]))

        # Test whether 4D Nifti throws the right error.
        img_4D = testing.generate_fake_fmri(shapes[0], length=17)
        assert_raises(DimensionError, largest_connected_component_img, img_4D)

    # tests adapted to non-native endian data dtype
    img1_change_dtype = nibabel.Nifti1Image(img1.get_data().astype('>f8'),
                                            affine=img1.get_affine())
    img2_change_dtype = nibabel.Nifti1Image(img2.get_data().astype('>f8'),
                                            affine=img2.get_affine())

    for create_files in (False, True):
        with testing.write_tmp_imgs(img1_change_dtype, img2_change_dtype,
                                    create_files=create_files) as imgs:
            # List of images as input
            out = largest_connected_component_img(imgs)
            assert_true(isinstance(out, list))
            assert_true(len(out) == 2)
            for o, s in zip(out, shapes):
                assert_true(o.shape == (s))

            # Single image as input
            out = largest_connected_component_img(imgs[0])
            assert_true(isinstance(out, Nifti1Image))
            assert_true(out.shape == (shapes[0]))

    # Test the output with native and without native
    out_native = largest_connected_component_img(img1)

    out_non_native = largest_connected_component_img(img1_change_dtype)
    np.testing.assert_equal(out_native.get_data(), out_non_native.get_data())
