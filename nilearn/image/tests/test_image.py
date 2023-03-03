"""Test image pre-processing functions"""
import os
import platform
import sys
import tempfile

import nibabel
import numpy as np
import pytest
from nibabel import AnalyzeImage, Nifti1Image
from nibabel.freesurfer import MGHImage
from nilearn import signal
from nilearn._utils import data_gen, niimg_conversions, testing
from nilearn._utils.exceptions import DimensionError
from nilearn.image import (
    binarize_img,
    concat_imgs,
    get_data,
    image,
    index_img,
    iter_img,
    largest_connected_component_img,
    math_img,
    new_img_like,
    resampling,
    threshold_img,
)
from numpy.testing import assert_allclose, assert_array_equal

try:
    import pandas as pd
except Exception:
    pass

X64 = platform.architecture()[0] == "64bit"

currdir = os.path.dirname(os.path.abspath(__file__))
datadir = os.path.join(currdir, "data")


def test_get_data():
    img, *_ = data_gen.generate_fake_fmri(shape=(10, 11, 12))
    data = get_data(img)
    assert data.shape == img.shape
    assert data is img._data_cache
    mask_img = new_img_like(img, data > 0)
    data = get_data(mask_img)
    assert data.dtype == np.dtype("uint8")
    img_3d = index_img(img, 0)
    with tempfile.TemporaryDirectory() as tempdir:
        filename = os.path.join(tempdir, "img_{}.nii.gz")
        img_3d.to_filename(filename.format("a"))
        img_3d.to_filename(filename.format("b"))
        data = get_data(filename.format("a"))
        assert len(data.shape) == 3
        data = get_data(filename.format("*"))
        assert len(data.shape) == 4


def test_high_variance_confounds():
    # See also test_signals.test_high_variance_confounds()
    # There is only tests on what is added by image.high_variance_confounds()
    # compared to signal.high_variance_confounds()

    shape = (40, 41, 42)
    length = 17
    n_confounds = 10

    img, mask_img = data_gen.generate_fake_fmri(shape=shape, length=length)

    confounds1 = image.high_variance_confounds(
        img, mask_img=mask_img, percentile=10.0, n_confounds=n_confounds
    )
    assert confounds1.shape == (length, n_confounds)

    # No mask.
    confounds2 = image.high_variance_confounds(
        img, percentile=10.0, n_confounds=n_confounds
    )
    assert confounds2.shape == (length, n_confounds)


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
        nb_neighbors_arr[i, j, k] = (
            3 + (0 < i < N - 1) + (0 < j < N - 1) + (0 < k < N - 1)
        )

    expected = (1 + neighbor_weight * nb_neighbors_arr) / (
        1 + neighbor_weight * nb_neighbors_max
    )
    np.testing.assert_allclose(smooth_data, expected)


def test__smooth_array():
    """Test smoothing of images: _smooth_array()"""
    # Impulse in 3D
    data = np.zeros((40, 41, 42))
    data[20, 20, 20] = 1

    # fwhm divided by any test affine must be odd. Otherwise assertion below
    # will fail. ( 9 / 0.6 = 15 is fine)
    fwhm = 9
    test_affines = (
        np.eye(4),
        np.diag((1, 1, -1, 1)),
        np.diag((0.6, 1, 0.6, 1)),
    )
    for affine in test_affines:
        filtered = image._smooth_array(data, affine, fwhm=fwhm, copy=True)
        assert not np.may_share_memory(filtered, data)

        # We are expecting a full-width at half maximum of
        # fwhm / voxel_size:
        vmax = filtered.max()
        above_half_max = filtered > 0.5 * vmax
        for axis in (0, 1, 2):
            proj = np.any(
                np.any(np.rollaxis(above_half_max, axis=axis), axis=-1),
                axis=-1,
            )
            np.testing.assert_equal(
                proj.sum(), fwhm / np.abs(affine[axis, axis])
            )

    # Check that NaNs in the data do not propagate
    data[10, 10, 10] = np.NaN
    filtered = image._smooth_array(
        data, affine, fwhm=fwhm, ensure_finite=True, copy=True
    )
    assert np.all(np.isfinite(filtered))

    # Check copy=False.
    for affine in test_affines:
        data = np.zeros((40, 41, 42))
        data[20, 20, 20] = 1
        image._smooth_array(data, affine, fwhm=fwhm, copy=False)

        # We are expecting a full-width at half maximum of
        # fwhm / voxel_size:
        vmax = data.max()
        above_half_max = data > 0.5 * vmax
        for axis in (0, 1, 2):
            proj = np.any(
                np.any(np.rollaxis(above_half_max, axis=axis), axis=-1),
                axis=-1,
            )
            np.testing.assert_equal(
                proj.sum(), fwhm / np.abs(affine[axis, axis])
            )

    # Check fwhm='fast'
    for affine in test_affines:
        np.testing.assert_equal(
            image._smooth_array(data, affine, fwhm="fast"),
            image._fast_smooth_array(data),
        )

    # Check corner case when fwhm=0. See #1537
    # Test whether function _smooth_array raises a warning when fwhm=0.
    with pytest.warns(UserWarning):
        image._smooth_array(data, affine, fwhm=0.0)

    # Test output equal when fwhm=None and fwhm=0
    out_fwhm_none = image._smooth_array(data, affine, fwhm=None)
    out_fwhm_zero = image._smooth_array(data, affine, fwhm=0.0)
    assert_array_equal(out_fwhm_none, out_fwhm_zero)


def test_smooth_img():
    # This function only checks added functionalities compared
    # to _smooth_array()
    shapes = ((10, 11, 12), (13, 14, 15))
    lengths = (17, 18)
    fwhm = (1.0, 2.0, 3.0)

    img1, mask1 = data_gen.generate_fake_fmri(
        shape=shapes[0], length=lengths[0]
    )
    img2, mask2 = data_gen.generate_fake_fmri(
        shape=shapes[1], length=lengths[1]
    )

    for create_files in (False, True):
        with testing.write_tmp_imgs(
            img1, img2, create_files=create_files
        ) as imgs:
            # List of images as input
            out = image.smooth_img(imgs, fwhm)
            assert isinstance(out, list)
            assert len(out) == 2
            for o, s, l in zip(out, shapes, lengths):
                assert o.shape == (s + (l,))

            # Single image as input
            out = image.smooth_img(imgs[0], fwhm)
            assert isinstance(out, nibabel.Nifti1Image)
            assert out.shape == (shapes[0] + (lengths[0],))

    # Check corner case situations when fwhm=0, See issue #1537
    # Test whether function smooth_img raises a warning when fwhm=0.
    with pytest.warns(UserWarning):
        image.smooth_img(img1, fwhm=0.0)

    # Test output equal when fwhm=None and fwhm=0
    out_fwhm_none = image.smooth_img(img1, fwhm=None)
    out_fwhm_zero = image.smooth_img(img1, fwhm=0.0)
    assert_array_equal(get_data(out_fwhm_none), get_data(out_fwhm_zero))

    data1 = np.zeros((10, 11, 12))
    data1[2:4, 1:5, 3:6] = 1
    data2 = np.zeros((13, 14, 15))
    data2[2:4, 1:5, 3:6] = 9
    img1_nifti2 = nibabel.Nifti2Image(data1, affine=np.eye(4))
    img2_nifti2 = nibabel.Nifti2Image(data2, affine=np.eye(4))
    out = image.smooth_img([img1_nifti2, img2_nifti2], fwhm=1.0)


def test__crop_img_to():
    data = np.zeros((5, 6, 7))
    data[2:4, 1:5, 3:6] = 1
    affine = np.diag((4, 3, 2, 1))
    img = nibabel.Nifti1Image(data, affine=affine)

    slices = [slice(2, 4), slice(1, 5), slice(3, 6)]
    cropped_img = image._crop_img_to(img, slices, copy=False)

    new_origin = np.array((4, 3, 2)) * np.array((2, 1, 3))

    # check that correct part was extracted:
    assert (get_data(cropped_img) == 1).all()
    assert cropped_img.shape == (2, 4, 3)

    # check that affine was adjusted correctly
    assert (cropped_img.affine[:3, 3] == new_origin).all()

    # check that data was really not copied
    data[2:4, 1:5, 3:6] = 2
    assert (get_data(cropped_img) == 2).all()

    # check that copying works
    copied_cropped_img = image._crop_img_to(img, slices)
    data[2:4, 1:5, 3:6] = 1
    assert (get_data(copied_cropped_img) == 2).all()


def test_crop_img():
    data = np.zeros((5, 6, 7))
    data[2:4, 1:5, 3:6] = 1
    affine = np.diag((4, 3, 2, 1))
    img = nibabel.Nifti1Image(data, affine=affine)

    cropped_img = image.crop_img(img)

    # correction for padding with "-1"

    # check that correct part was extracted:
    # This also corrects for padding
    assert (get_data(cropped_img)[1:-1, 1:-1, 1:-1] == 1).all()
    assert cropped_img.shape == (2 + 2, 4 + 2, 3 + 2)


def test_crop_threshold_tolerance():
    """Check if crop can skip values that are extremely close to zero.

    In a relative sense and will crop them away
    """
    data = np.zeros([10, 14, 12])
    data[3:7, 3:7, 5:9] = 1.0
    active_shape = (4 + 2, 4 + 2, 4 + 2)  # add padding

    # add an infinitesimal outside this block
    data[3, 3, 3] = 1e-12
    affine = np.eye(4)
    img = nibabel.Nifti1Image(data, affine=affine)

    cropped_img = image.crop_img(img)
    assert cropped_img.shape == active_shape


def test_mean_img():
    rng = np.random.RandomState(42)
    data1 = np.zeros((5, 6, 7))
    data2 = rng.uniform(size=(5, 6, 7))
    data3 = rng.uniform(size=(5, 6, 7, 3))
    affine = np.diag((4, 3, 2, 1))
    img1 = nibabel.Nifti1Image(data1, affine=affine)
    img2 = nibabel.Nifti1Image(data2, affine=affine)
    img3 = nibabel.Nifti1Image(data3, affine=affine)
    for imgs in (
        [
            img1,
        ],
        [img1, img2],
        [img2, img1, img2],
        [img3, img1, img2],  # Mixture of 4D and 3D images
    ):
        arrays = []
        # Ground-truth:
        for img in imgs:
            img = get_data(img)
            if img.ndim == 4:
                img = np.mean(img, axis=-1)
            arrays.append(img)
        truth = np.mean(arrays, axis=0)

        mean_img = image.mean_img(imgs)
        assert_array_equal(mean_img.affine, affine)
        assert_array_equal(get_data(mean_img), truth)

        # Test with files
        with testing.write_tmp_imgs(*imgs) as imgs:
            mean_img = image.mean_img(imgs)
            assert_array_equal(mean_img.affine, affine)
            if X64:
                assert_array_equal(get_data(mean_img), truth)
            else:
                # We don't really understand but arrays are not
                # exactly equal on 32bit. Given that you can not do
                # much real world data analysis with nilearn on a
                # 32bit machine it is not worth investigating more
                assert_allclose(
                    get_data(mean_img),
                    truth,
                    rtol=np.finfo(truth.dtype).resolution,
                    atol=0,
                )


def test_mean_img_resample():
    # Test resampling in mean_img with a permutation of the axes
    rng = np.random.RandomState(42)
    data = rng.uniform(size=(5, 6, 7, 40))
    affine = np.diag((4, 3, 2, 1))
    img = nibabel.Nifti1Image(data, affine=affine)
    mean_img = nibabel.Nifti1Image(data.mean(axis=-1), affine=affine)

    target_affine = affine[:, [1, 0, 2, 3]]  # permutation of axes
    mean_img_with_resampling = image.mean_img(img, target_affine=target_affine)
    resampled_mean_image = resampling.resample_img(
        mean_img, target_affine=target_affine
    )
    assert_array_equal(
        get_data(resampled_mean_image), get_data(mean_img_with_resampling)
    )
    assert_array_equal(
        resampled_mean_image.affine, mean_img_with_resampling.affine
    )
    assert_array_equal(mean_img_with_resampling.affine, target_affine)


def test_swap_img_hemispheres():
    rng = np.random.RandomState(42)

    # make sure input image data is not overwritten inside function
    data = rng.standard_normal(size=(4, 5, 7))
    data_img = nibabel.Nifti1Image(data, np.eye(4))
    image.swap_img_hemispheres(data_img)
    np.testing.assert_array_equal(get_data(data_img), data)

    # swapping operations work
    np.testing.assert_array_equal(  # one turn
        get_data(image.swap_img_hemispheres(data_img)), data[::-1]
    )
    np.testing.assert_array_equal(  # two turns -> back to original data
        get_data(
            image.swap_img_hemispheres(image.swap_img_hemispheres(data_img))
        ),
        data,
    )


def test_concat_imgs():
    assert concat_imgs is niimg_conversions.concat_niimgs


def test_index_img():
    img_3d = nibabel.Nifti1Image(np.ones((3, 4, 5)), np.eye(4))
    expected_error_msg = (
        "Input data has incompatible dimensionality: "
        "Expected dimension is 4D and you provided "
        "a 3D image."
    )
    with pytest.raises(TypeError, match=expected_error_msg):
        image.index_img(img_3d, 0)

    affine = np.array(
        [
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0, 12.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    img_4d, _ = data_gen.generate_fake_fmri(affine=affine)

    fourth_dim_size = img_4d.shape[3]
    tested_indices = list(range(fourth_dim_size)) + [
        slice(2, 8, 2),
        [1, 2, 3, 2],
        [],
        (np.arange(fourth_dim_size) % 3) == 1,
    ]
    for i in tested_indices:
        this_img = image.index_img(img_4d, i)
        expected_data_3d = get_data(img_4d)[..., i]
        assert_array_equal(get_data(this_img), expected_data_3d)
        assert_array_equal(this_img.affine, img_4d.affine)

    for i in [
        fourth_dim_size,
        -fourth_dim_size - 1,
        [0, fourth_dim_size],
        np.repeat(True, fourth_dim_size + 1),
    ]:
        with pytest.raises(
            IndexError,
            match="out of bounds|invalid index|out of range|" "boolean index",
        ):
            image.index_img(img_4d, i)


def test_pd_index_img():
    # confirm indices from pandas dataframes are handled correctly
    if "pandas" not in sys.modules:
        raise pytest.skip(msg="Pandas not available")

    affine = np.array(
        [
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0, 12.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    img_4d, _ = data_gen.generate_fake_fmri(affine=affine)

    fourth_dim_size = img_4d.shape[3]

    rng = np.random.RandomState(42)
    arr = rng.uniform(size=fourth_dim_size) > 0.5
    df = pd.DataFrame({"arr": arr})

    np_index_img = image.index_img(img_4d, arr)
    pd_index_img = image.index_img(img_4d, df)
    assert_array_equal(get_data(np_index_img), get_data(pd_index_img))


def test_iter_img():
    img_3d = nibabel.Nifti1Image(np.ones((3, 4, 5)), np.eye(4))
    expected_error_msg = (
        "Input data has incompatible dimensionality: "
        "Expected dimension is 4D and you provided "
        "a 3D image."
    )
    with pytest.raises(TypeError, match=expected_error_msg):
        image.iter_img(img_3d)

    affine = np.array(
        [
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0, 12.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    img_4d, _ = data_gen.generate_fake_fmri(affine=affine)

    for i, img in enumerate(image.iter_img(img_4d)):
        expected_data_3d = get_data(img_4d)[..., i]
        assert_array_equal(get_data(img), expected_data_3d)
        assert_array_equal(img.affine, img_4d.affine)

    with testing.write_tmp_imgs(img_4d) as img_4d_filename:
        for i, img in enumerate(image.iter_img(img_4d_filename)):
            expected_data_3d = get_data(img_4d)[..., i]
            assert_array_equal(get_data(img), expected_data_3d)
            assert_array_equal(img.affine, img_4d.affine)
        # enables to delete "img_4d_filename" on windows
        del img

    img_3d_list = list(image.iter_img(img_4d))
    for i, img in enumerate(image.iter_img(img_3d_list)):
        expected_data_3d = get_data(img_4d)[..., i]
        assert_array_equal(get_data(img), expected_data_3d)
        assert_array_equal(img.affine, img_4d.affine)

    with testing.write_tmp_imgs(*img_3d_list) as img_3d_filenames:
        for i, img in enumerate(image.iter_img(img_3d_filenames)):
            expected_data_3d = get_data(img_4d)[..., i]
            assert_array_equal(get_data(img), expected_data_3d)
            assert_array_equal(img.affine, img_4d.affine)
        # enables to delete "img_3d_filename" on windows
        del img


def test_new_img_like_mgz():
    """Check that new images can be generated with bool MGZ type.

    This is usually when computing masks using MGZ inputs, e.g.
    when using plot_stap_map
    """
    ref_img = nibabel.load(os.path.join(datadir, "test.mgz"))
    data = np.ones(get_data(ref_img).shape, dtype=bool)
    affine = ref_img.affine
    new_img_like(ref_img, data, affine, copy_header=False)


def test_new_img_like():
    # Give a list to new_img_like
    data = np.zeros((5, 6, 7))
    data[2:4, 1:5, 3:6] = 1
    affine = np.diag((4, 3, 2, 1))
    img = nibabel.Nifti1Image(data, affine=affine)
    img2 = new_img_like(
        [
            img,
        ],
        data,
    )
    np.testing.assert_array_equal(get_data(img), get_data(img2))

    # test_new_img_like_with_nifti2image_copy_header
    img_nifti2 = nibabel.Nifti2Image(data, affine=affine)
    img2_nifti2 = new_img_like(
        [
            img_nifti2,
        ],
        data,
        copy_header=True,
    )
    np.testing.assert_array_equal(get_data(img_nifti2), get_data(img2_nifti2))


def test_new_img_like_non_iterable_header():
    """
    Tests that when an niimg's header is not iterable
    & it is set to be copied, an error is not raised.
    """
    rng = np.random.RandomState(42)
    fake_fmri_data = rng.uniform(size=(10, 10, 10, 10))
    fake_affine = rng.uniform(size=(4, 4))
    fake_spatial_image = nibabel.spatialimages.SpatialImage(
        fake_fmri_data, fake_affine
    )
    assert new_img_like(
        fake_spatial_image, data=fake_fmri_data, copy_header=True
    )


@pytest.mark.parametrize("no_int64_nifti", ["allow for this test"])
def test_new_img_like_int64():
    img = data_gen.generate_labeled_regions((3, 3, 3), 2)
    data = image.get_data(img).astype("int32")
    with pytest.warns(None) as record:
        new_img = new_img_like(img, data)
    assert not record
    assert image.get_data(new_img).dtype == "int32"
    data = data.astype("int64")
    with pytest.warns(UserWarning, match=r".*array.*contains.*64.*"):
        new_img = new_img_like(img, data)
    assert image.get_data(new_img).dtype == "int32"
    data[:] = 2**40
    with pytest.warns(UserWarning, match=r".*64.*too large.*"):
        new_img = new_img_like(img, data, copy_header=True)
    assert image.get_data(new_img).dtype == "int64"


def test_validity_threshold_value_in_threshold_img():
    """Check that invalid values to threshold_img's threshold parameter raise
    Exceptions.
    """
    shape = (6, 8, 10)
    maps, _ = data_gen.generate_maps(shape, n_regions=2)

    # testing to raise same error when threshold=None case
    with pytest.raises(
        TypeError,
        match="threshold should be either a number or a string",
    ):
        threshold_img(maps, threshold=None)

    invalid_threshold_values = ["90t%", "s%", "t", "0.1"]
    name = "threshold"
    for thr in invalid_threshold_values:
        with pytest.raises(
            ValueError,
            match=f"{name}.+should be a number followed by the percent sign",
        ):
            threshold_img(maps, threshold=thr)


def test_threshold_img():
    """Smoke test for threshold_img with valid threshold inputs."""
    shape = (10, 20, 30)
    maps, _ = data_gen.generate_maps(shape, n_regions=4)
    affine = np.eye(4)
    mask_img = nibabel.Nifti1Image(np.ones((shape), dtype=np.int8), affine)

    for img in iter_img(maps):
        # when threshold is a float value
        threshold_img(img, threshold=0.8)

        # when we provide mask image
        threshold_img(img, threshold=1, mask_img=mask_img)

        # when threshold is a percentile
        threshold_img(img, threshold="2%")


def test_threshold_img_with_cluster_threshold():
    """Check that threshold_img behaves as expected with cluster_threshold
    and/or two_sided.
    """
    # First we create a statistical image with specific characteristics
    shape = (20, 20, 30)
    affine = np.eye(4)
    data = np.zeros(shape, dtype="int32")
    data[:2, :2, :2] = 4  # 8-voxel positive cluster
    data[4:6, :2, :2] = -4  # 8-voxel negative cluster
    data[8:11, 0, 0] = 5  # 3-voxel positive cluster
    data[13:16, 0, 0] = -5  # 3-voxel positive cluster
    data[:6, 4:10, :6] = 1  # 216-voxel positive cluster with low value
    data[13:19, 4:10, :6] = -1  # 216-voxel negative cluster with low value

    stat_img = nibabel.Nifti1Image(data, affine)

    # The standard approach should retain any clusters with values > 2
    thr_img = threshold_img(stat_img, threshold=2, two_sided=False, copy=True)
    assert np.array_equal(np.unique(thr_img.get_fdata()), np.array([0, 4, 5]))

    # With two-sided we get any clusters with |values| > 2
    thr_img = threshold_img(stat_img, threshold=2, two_sided=True, copy=True)
    assert np.array_equal(
        np.unique(thr_img.get_fdata()),
        np.array([-5, -4, 0, 4, 5]),
    )

    # With a cluster threshold of 5 we get clusters with |values| > 2 and
    # cluster sizes > 5
    thr_img = threshold_img(
        stat_img,
        threshold=2,
        two_sided=True,
        cluster_threshold=5,
        copy=True,
    )
    assert np.array_equal(np.unique(thr_img.get_fdata()), np.array([-4, 0, 4]))
    assert np.sum(thr_img.get_fdata() == 4) == 8
    # With a cluster threshold of 5 we get clusters with |values| > 0.5 and
    # cluster sizes > 5
    thr_img = threshold_img(
        stat_img,
        threshold=0.5,
        two_sided=True,
        cluster_threshold=5,
        copy=True,
    )
    assert np.array_equal(
        np.unique(thr_img.get_fdata()),
        np.array([-4, -1, 0, 1, 4]),
    )

    # Now we disable two_sided again to get clusters with values > 0.5 and
    # cluster sizes > 5
    thr_img = threshold_img(
        stat_img,
        threshold=0.5,
        two_sided=False,
        cluster_threshold=5,
        copy=True,
    )
    assert np.array_equal(np.unique(thr_img.get_fdata()), np.array([0, 1, 4]))


def test_threshold_img_copy():
    """Test the behavior of threshold_img's copy parameter."""
    img_ones = Nifti1Image(np.ones((10, 10, 10, 10)), np.eye(4))

    # Check that copy does not mutate. It returns modified copy.
    thresholded = threshold_img(img_ones, 2)  # threshold 2 > 1

    # Original img_ones should have all ones.
    assert_array_equal(get_data(img_ones), np.ones((10, 10, 10, 10)))
    # Thresholded should have all zeros.
    assert_array_equal(get_data(thresholded), np.zeros((10, 10, 10, 10)))

    # Check that not copying does mutate.
    img_to_mutate = Nifti1Image(np.ones((10, 10, 10, 10)), np.eye(4))
    thresholded = threshold_img(img_to_mutate, 2, copy=False)
    # Check that original mutates
    assert_array_equal(get_data(img_to_mutate), np.zeros((10, 10, 10, 10)))
    # And that returned value is also thresholded.
    assert_array_equal(get_data(img_to_mutate), get_data(thresholded))


def test_isnan_threshold_img_data():
    """Check threshold_img converges properly when input image has nans."""
    shape = (10, 10, 10)
    maps, _ = data_gen.generate_maps(shape, n_regions=2)
    data = get_data(maps)
    data[:, :, 0] = np.nan

    maps_img = nibabel.Nifti1Image(data, np.eye(4))
    threshold_img(maps_img, threshold=0.8)


def test_math_img_exceptions():
    img1 = Nifti1Image(np.ones((10, 10, 10, 10)), np.eye(4))
    img2 = Nifti1Image(np.zeros((10, 20, 10, 10)), np.eye(4))
    img3 = Nifti1Image(np.ones((10, 10, 10, 10)), np.eye(4))
    img4 = Nifti1Image(np.ones((10, 10, 10, 10)), np.eye(4) * 2)

    formula = "np.mean(img1, axis=-1) - np.mean(img2, axis=-1)"
    # Images with different shapes should raise a ValueError exception.
    with pytest.raises(ValueError, match="Input images cannot be compared"):
        math_img(formula, img1=img1, img2=img2)

    # Images with different affines should raise a ValueError exception.
    with pytest.raises(ValueError, match="Input images cannot be compared"):
        math_img(formula, img1=img1, img2=img4)

    bad_formula = "np.toto(img1, axis=-1) - np.mean(img3, axis=-1)"
    with pytest.raises(
        AttributeError, match="Input formula couldn't be processed"
    ):
        math_img(bad_formula, img1=img1, img3=img3)


def test_math_img():
    img1 = Nifti1Image(np.ones((10, 10, 10, 10)), np.eye(4))
    img2 = Nifti1Image(np.zeros((10, 10, 10, 10)), np.eye(4))
    expected_result = Nifti1Image(np.ones((10, 10, 10)), np.eye(4))

    formula = "np.mean(img1, axis=-1) - np.mean(img2, axis=-1)"
    for create_files in (True, False):
        with testing.write_tmp_imgs(
            img1, img2, create_files=create_files
        ) as imgs:
            result = math_img(formula, img1=imgs[0], img2=imgs[1])
            assert_array_equal(get_data(result), get_data(expected_result))
            assert_array_equal(result.affine, expected_result.affine)
            assert result.shape == expected_result.shape


def test_binarize_img():
    img = Nifti1Image(np.random.rand(10, 10, 10, 10), np.eye(4))
    # Test that all output values are 1.
    img1 = binarize_img(img)
    np.testing.assert_array_equal(np.unique(img1.dataobj), np.array([1]))
    # Test that it works with threshold
    img2 = binarize_img(img, threshold=0.5)
    np.testing.assert_array_equal(np.unique(img2.dataobj), np.array([0, 1]))
    # Test that manual binarization equals binarize_img results.
    img3 = Nifti1Image(np.random.rand(10, 10, 10, 10), np.eye(4))
    img3.dataobj[img.dataobj < 0.5] = 0
    img3.dataobj[img.dataobj >= 0.5] = 1
    np.testing.assert_array_equal(img2.dataobj, img3.dataobj)


def test_clean_img():
    rng = np.random.RandomState(42)
    data = rng.standard_normal(size=(10, 10, 10, 100)) + 0.5
    data_flat = data.T.reshape(100, -1)
    data_img = nibabel.Nifti1Image(data, np.eye(4))

    pytest.raises(
        ValueError, image.clean_img, data_img, t_r=None, low_pass=0.1
    )

    data_img_ = image.clean_img(
        data_img, detrend=True, standardize=False, low_pass=0.1, t_r=1.0
    )
    data_flat_ = signal.clean(
        data_flat, detrend=True, standardize=False, low_pass=0.1, t_r=1.0
    )

    np.testing.assert_almost_equal(
        get_data(data_img_).T.reshape(100, -1), data_flat_
    )
    # if NANs
    data[:, 9, 9] = np.nan
    # if infinity
    data[:, 5, 5] = np.inf
    nan_img = nibabel.Nifti1Image(data, np.eye(4))
    clean_im = image.clean_img(nan_img, ensure_finite=True)
    assert np.any(np.isfinite(get_data(clean_im))), True

    # test_clean_img_passing_nifti2image
    data_img_nifti2 = nibabel.Nifti2Image(data, np.eye(4))

    image.clean_img(
        data_img_nifti2, detrend=True, standardize=False, low_pass=0.1, t_r=1.0
    )

    # if mask_img
    img, mask_img = data_gen.generate_fake_fmri(shape=(10, 10, 10), length=10)
    data_img_mask_ = image.clean_img(img, mask_img=mask_img)

    # Checks that output with full mask and without is equal
    data_img_ = image.clean_img(img)
    np.testing.assert_almost_equal(
        get_data(data_img_), get_data(data_img_mask_)
    )


def test_largest_cc_img():
    """Check the extraction of the largest connected component, for niftis.

    Similar to smooth_img tests for largest connected_component_img, here also
    only the added features for largest_connected_component are tested.
    """
    # Test whether dimension of 3Dimg and list of 3Dimgs are kept.
    shapes = ((10, 11, 12), (13, 14, 15))
    regions = [1, 3]

    img1 = data_gen.generate_labeled_regions(
        shape=shapes[0], n_regions=regions[0]
    )
    img2 = data_gen.generate_labeled_regions(
        shape=shapes[1], n_regions=regions[1]
    )

    for create_files in (False, True):
        with testing.write_tmp_imgs(
            img1, img2, create_files=create_files
        ) as imgs:
            # List of images as input
            out = largest_connected_component_img(imgs)
            assert isinstance(out, list)
            assert len(out) == 2
            for o, s in zip(out, shapes):
                assert o.shape == (s)

            # Single image as input
            out = largest_connected_component_img(imgs[0])
            assert isinstance(out, Nifti1Image)
            assert out.shape == (shapes[0])

        # Test whether 4D Nifti throws the right error.
        img_4D = data_gen.generate_fake_fmri(shapes[0], length=17)
        pytest.raises(DimensionError, largest_connected_component_img, img_4D)

    # tests adapted to non-native endian data dtype
    img1_change_dtype = nibabel.Nifti1Image(
        get_data(img1).astype(">f8"), affine=img1.affine
    )
    img2_change_dtype = nibabel.Nifti1Image(
        get_data(img2).astype(">f8"), affine=img2.affine
    )

    for create_files in (False, True):
        with testing.write_tmp_imgs(
            img1_change_dtype, img2_change_dtype, create_files=create_files
        ) as imgs:
            # List of images as input
            out = largest_connected_component_img(imgs)
            assert isinstance(out, list)
            assert len(out) == 2
            for o, s in zip(out, shapes):
                assert o.shape == (s)

            # Single image as input
            out = largest_connected_component_img(imgs[0])
            assert isinstance(out, Nifti1Image)
            assert out.shape == (shapes[0])

    # Test the output with native and without native
    out_native = largest_connected_component_img(img1)

    out_non_native = largest_connected_component_img(img1_change_dtype)
    np.testing.assert_equal(get_data(out_native), get_data(out_non_native))


def test_new_img_like_mgh_image():
    data = np.zeros((5, 5, 5), dtype=np.uint8)
    niimg = nibabel.freesurfer.MGHImage(dataobj=data, affine=np.eye(4))
    new_img_like(niimg, data.astype(float), niimg.affine, copy_header=True)


@pytest.mark.parametrize("image", [MGHImage, AnalyzeImage])
def test_new_img_like_boolean_data(image):
    """Check defaulting boolean input data to np.uint8 dtype is valid for
    encoding with nibabel image classes MGHImage and AnalyzeImage.
    """
    data = np.random.randn(5, 5, 5).astype("uint8")
    in_img = image(dataobj=data, affine=np.eye(4))
    out_img = new_img_like(in_img, data=in_img.get_fdata() > 0.5)
    assert get_data(out_img).dtype == "uint8"
