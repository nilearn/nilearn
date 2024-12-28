"""Test image pre-processing functions."""

import platform
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from nibabel import AnalyzeImage, Nifti1Image, Nifti2Image, load, spatialimages
from nibabel.freesurfer import MGHImage
from numpy.testing import (
    assert_allclose,
    assert_almost_equal,
    assert_array_almost_equal,
    assert_array_equal,
    assert_equal,
)

from nilearn import signal
from nilearn._utils import testing
from nilearn._utils.data_gen import (
    _basic_confounds,
    generate_fake_fmri,
    generate_labeled_regions,
    generate_maps,
)
from nilearn._utils.exceptions import DimensionError
from nilearn.conftest import _affine_eye, _img_3d_rand, _rng, _shape_4d_default
from nilearn.image import (
    binarize_img,
    clean_img,
    concat_imgs,
    copy_img,
    crop_img,
    get_data,
    high_variance_confounds,
    image,
    index_img,
    iter_img,
    largest_connected_component_img,
    math_img,
    mean_img,
    new_img_like,
    resampling,
    smooth_img,
    swap_img_hemispheres,
    threshold_img,
)
from nilearn.image.tests._testing import match_headers_keys

X64 = platform.architecture()[0] == "64bit"


AFFINE_TO_TEST = [
    _affine_eye(),
    np.diag((1, 1, -1, 1)),
    np.diag((0.6, 1, 0.6, 1)),
]

NON_EYE_AFFINE = np.array(
    [
        [1.0, 2.0, 3.0, 4.0],
        [5.0, 6.0, 7.0, 8.0],
        [9.0, 10.0, 11.0, 12.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
)


def _new_data_for_smooth_array():
    # Impulse in 3D
    data = np.zeros((40, 41, 42))
    data[20, 20, 20] = 1
    return data


def _make_largest_cc_img_test_data():
    shapes = ((10, 11, 12), (13, 14, 15))
    regions = [1, 3]

    img1 = generate_labeled_regions(shape=shapes[0], n_regions=regions[0])
    img2 = generate_labeled_regions(shape=shapes[1], n_regions=regions[1])
    return img1, img2, shapes


def _images_to_mean():
    """Return a mixture of 4D and 3D images."""
    rng = _rng()

    data1 = np.zeros((5, 6, 7))
    data2 = rng.uniform(size=(5, 6, 7))
    data3 = rng.uniform(size=(5, 6, 7, 3))

    affine = np.diag((4, 3, 2, 1))

    img1 = Nifti1Image(data1, affine=affine)
    img2 = Nifti1Image(data2, affine=affine)
    img3 = Nifti1Image(data3, affine=affine)

    imgs = (
        [img1],
        [img1, img2],
        [img2, img1, img2],
        [img3, img1, img2],
    )
    return imgs


def _check_fwhm(data, affine, fwhm):
    """Expect a full-width at half maximum of fwhm / voxel_size."""
    vmax = data.max()
    above_half_max = data > 0.5 * vmax
    for axis in [0, 1, 2]:
        proj = np.any(
            np.any(np.rollaxis(above_half_max, axis=axis), axis=-1),
            axis=-1,
        )
        assert_equal(proj.sum(), fwhm / np.abs(affine[axis, axis]))


def _mean_ground_truth(imgs):
    arrays = []
    for img in imgs:
        img = get_data(img)
        if img.ndim == 4:
            img = np.mean(img, axis=-1)
        arrays.append(img)
    return np.mean(arrays, axis=0)


@pytest.fixture(scope="session")
def smooth_array_data():
    return _new_data_for_smooth_array()


@pytest.fixture(scope="session")
def stat_img_test_data():
    shape = (20, 20, 30)
    affine = _affine_eye()
    data = np.zeros(shape, dtype="int32")
    data[:2, :2, :2] = 4  # 8-voxel positive cluster
    data[4:6, :2, :2] = -4  # 8-voxel negative cluster
    data[8:11, 0, 0] = 5  # 3-voxel positive cluster
    data[13:16, 0, 0] = -5  # 3-voxel positive cluster
    data[:6, 4:10, :6] = 1  # 216-voxel positive cluster with low value
    data[13:19, 4:10, :6] = -1  # 216-voxel negative cluster with low value

    stat_img = Nifti1Image(data, affine)

    return stat_img


def test_get_data(tmp_path, shape_3d_default):
    img, *_ = generate_fake_fmri(shape=shape_3d_default)

    data = get_data(img)

    assert data.shape == img.shape
    assert data is img._data_cache

    mask_img = new_img_like(img, data > 0)
    data = get_data(mask_img)

    assert data.dtype == np.dtype("uint8")

    img_3d = index_img(img, 0)

    filename = str(tmp_path / "img_{}.nii.gz")
    img_3d.to_filename(filename.format("a"))
    img_3d.to_filename(filename.format("b"))

    data = get_data(filename.format("a"))

    assert len(data.shape) == 3

    data = get_data(filename.format("*"))

    assert len(data.shape) == 4


def test_high_variance_confounds(shape_3d_default):
    # See also test_signals.test_high_variance_confounds()
    # There is only tests on what is added by high_variance_confounds()
    # compared to signal.high_variance_confounds()

    length = 17
    n_confounds = 10

    img, mask_img = generate_fake_fmri(shape=shape_3d_default, length=length)

    confounds1 = high_variance_confounds(
        img, mask_img=mask_img, percentile=10.0, n_confounds=n_confounds
    )

    assert confounds1.shape == (length, n_confounds)

    # No mask.
    confounds2 = high_variance_confounds(
        img, percentile=10.0, n_confounds=n_confounds
    )

    assert confounds2.shape == (length, n_confounds)


def test_fast_smooth_array():
    N = 4
    shape = (N, N, N)
    # hardcoded in _fast_smooth_array
    neighbor_weight = 0.2
    # 6 neighbors in 3D if you are not on an edge
    n_neighbors_max = 6

    data = np.ones(shape)
    smooth_data = image._fast_smooth_array(data)

    # this contains the number of neighbors for each cell in the array
    n_neighbors_arr = np.empty(shape)
    for (i, j, k), __ in np.ndenumerate(n_neighbors_arr):
        n_neighbors_arr[i, j, k] = (
            3 + (0 < i < N - 1) + (0 < j < N - 1) + (0 < k < N - 1)
        )

    expected = (1 + neighbor_weight * n_neighbors_arr) / (
        1 + neighbor_weight * n_neighbors_max
    )
    assert_allclose(smooth_data, expected)


@pytest.mark.parametrize("affine", AFFINE_TO_TEST)
def test_smooth_array_fwhm_is_odd_with_copy(smooth_array_data, affine):
    """Test that fwhm divided by any affine is odd.

    Otherwise assertion below will fail.
    ( 9 / 0.6 = 15 is fine)
    """
    data = smooth_array_data
    fwhm = 9

    filtered = image.smooth_array(data, affine, fwhm=fwhm, copy=True)

    assert not np.may_share_memory(filtered, data)

    _check_fwhm(filtered, affine, fwhm)


@pytest.mark.parametrize("affine", AFFINE_TO_TEST)
def test_smooth_array_fwhm_is_odd_no_copy(affine):
    """Test that fwhm divided by any affine is odd.

    Otherwise assertion below will fail.
    ( 9 / 0.6 = 15 is fine)
    """
    data = _new_data_for_smooth_array()
    fwhm = 9

    image.smooth_array(data, affine, fwhm=fwhm, copy=False)

    _check_fwhm(data, affine, fwhm)


def test_smooth_array_nan_do_not_propagate():
    data = _new_data_for_smooth_array()
    data[10, 10, 10] = np.nan
    fwhm = 9
    affine = AFFINE_TO_TEST[2]

    filtered = image.smooth_array(
        data, affine, fwhm=fwhm, ensure_finite=True, copy=True
    )

    assert np.all(np.isfinite(filtered))


def test_smooth_array_same_result_with_fwhm_none_or_zero(
    smooth_array_data,
):
    affine = AFFINE_TO_TEST[2]

    out_fwhm_none = image.smooth_array(smooth_array_data, affine, fwhm=None)
    out_fwhm_zero = image.smooth_array(smooth_array_data, affine, fwhm=0.0)

    assert_array_equal(out_fwhm_none, out_fwhm_zero)


@pytest.mark.parametrize("affine", AFFINE_TO_TEST)
def test_fast_smooth_array_give_same_result_as_smooth_array(
    smooth_array_data, affine
):
    assert_equal(
        image.smooth_array(smooth_array_data, affine, fwhm="fast"),
        image._fast_smooth_array(smooth_array_data),
    )


def test_smooth_array_raise_warning_if_fwhm_is_zero(smooth_array_data):
    """See https://github.com/nilearn/nilearn/issues/1537."""
    affine = AFFINE_TO_TEST[2]
    with pytest.warns(UserWarning):
        image.smooth_array(smooth_array_data, affine, fwhm=0.0)


def test_smooth_img(affine_eye, tmp_path):
    """Checks added functionalities compared to image._smooth_array()."""
    shapes = ((10, 11, 12), (13, 14, 15))
    lengths = (17, 18)
    fwhm = (1.0, 2.0, 3.0)

    img1, _ = generate_fake_fmri(shape=shapes[0], length=lengths[0])
    img2, _ = generate_fake_fmri(shape=shapes[1], length=lengths[1])

    for create_files in (False, True):
        imgs = testing.write_imgs_to_path(
            img1, img2, file_path=tmp_path, create_files=create_files
        )
        # List of images as input
        out = smooth_img(imgs, fwhm)

        assert isinstance(out, list)
        assert len(out) == 2
        for o, s, l in zip(out, shapes, lengths):
            assert o.shape == (*s, l)

        # Single image as input
        out = smooth_img(imgs[0], fwhm)

        assert isinstance(out, Nifti1Image)
        assert out.shape == (shapes[0] + (lengths[0],))

    # Check corner case situations when fwhm=0, See issue #1537
    # Test whether function smooth_img raises a warning when fwhm=0.
    with pytest.warns(UserWarning):
        smooth_img(img1, fwhm=0.0)

    # Test output equal when fwhm=None and fwhm=0
    out_fwhm_none = smooth_img(img1, fwhm=None)
    out_fwhm_zero = smooth_img(img1, fwhm=0.0)

    assert_array_equal(get_data(out_fwhm_none), get_data(out_fwhm_zero))

    data1 = np.zeros((10, 11, 12))
    data1[2:4, 1:5, 3:6] = 1
    data2 = np.zeros((13, 14, 15))
    data2[2:4, 1:5, 3:6] = 9
    img1_nifti2 = Nifti2Image(data1, affine=affine_eye)
    img2_nifti2 = Nifti2Image(data2, affine=affine_eye)
    out = smooth_img([img1_nifti2, img2_nifti2], fwhm=1.0)


def test_crop_img_to():
    data = np.zeros((5, 6, 7))
    data[2:4, 1:5, 3:6] = 1
    affine = np.diag((4, 3, 2, 1))
    img = Nifti1Image(data, affine=affine)

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
    img = Nifti1Image(data, affine=affine)

    cropped_img = crop_img(img, copy_header=True)

    # correction for padding with "-1"
    # check that correct part was extracted:
    # This also corrects for padding
    assert (get_data(cropped_img)[1:-1, 1:-1, 1:-1] == 1).all()
    assert cropped_img.shape == (2 + 2, 4 + 2, 3 + 2)


def test_crop_img_copied_header(img_4d_mni_tr2):
    # Test equality of header fields between input and output
    # create zero padded data
    data = np.zeros((10, 10, 10, 10))
    data[0:4, 0:4, 0:4, :] = 1
    # replace the img_4d_mni_tr2 values with data
    img_4d_mni_tr2_zero_padded = new_img_like(
        img_4d_mni_tr2,
        data=data,
        affine=img_4d_mni_tr2.affine,
        copy_header=True,
    )
    cropped_img = crop_img(img_4d_mni_tr2_zero_padded, copy_header=True)
    # only dim[1:4] should be different
    assert (
        cropped_img.header["dim"][1:4]
        != img_4d_mni_tr2_zero_padded.header["dim"][1:4]
    ).all()
    # other dim indices should be the same
    assert (
        cropped_img.header["dim"][0]
        == img_4d_mni_tr2_zero_padded.header["dim"][0]
    )
    assert (
        cropped_img.header["dim"][4:]
        == img_4d_mni_tr2_zero_padded.header["dim"][4:]
    ).all()
    # other header fields should also be same
    match_headers_keys(
        cropped_img, img_4d_mni_tr2_zero_padded, except_keys=["dim"]
    )


def test_crop_threshold_tolerance(affine_eye):
    """Check if crop can skip values that are extremely close to zero.

    In a relative sense and will crop them away
    """
    data = np.zeros([10, 14, 12])
    data[3:7, 3:7, 5:9] = 1.0
    active_shape = (4 + 2, 4 + 2, 4 + 2)  # add padding

    # add an infinitesimal outside this block
    data[3, 3, 3] = 1e-12
    img = Nifti1Image(data, affine=affine_eye)

    cropped_img = crop_img(img, copy_header=True)

    assert cropped_img.shape == active_shape


@pytest.mark.parametrize("images_to_mean", _images_to_mean())
def test_mean_img(images_to_mean, tmp_path):
    affine = np.diag((4, 3, 2, 1))

    truth = _mean_ground_truth(images_to_mean)

    mean_img = image.mean_img(images_to_mean, copy_header=True)

    assert_array_equal(mean_img.affine, affine)
    assert_array_equal(get_data(mean_img), truth)

    # Test with files
    imgs = testing.write_imgs_to_path(*images_to_mean, file_path=tmp_path)
    mean_img = image.mean_img(imgs, copy_header=True)

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


def test_mean_img_resample(rng):
    # Test resampling in mean_img with a permutation of the axes
    data = rng.uniform(size=(5, 6, 7, 40))
    affine = np.diag((4, 3, 2, 1))
    img = Nifti1Image(data, affine=affine)
    mean_img = Nifti1Image(data.mean(axis=-1), affine=affine)

    target_affine = affine[:, [1, 0, 2, 3]]  # permutation of axes

    mean_img_with_resampling = image.mean_img(
        img, target_affine=target_affine, copy_header=True
    )

    resampled_mean_image = resampling.resample_img(
        mean_img,
        target_affine=target_affine,
        copy_header=True,
        force_resample=True,
    )

    assert_array_equal(
        get_data(resampled_mean_image), get_data(mean_img_with_resampling)
    )
    assert_array_equal(
        resampled_mean_image.affine, mean_img_with_resampling.affine
    )
    assert_array_equal(mean_img_with_resampling.affine, target_affine)


def test_mean_img_copied_header(img_4d_mni_tr2):
    # Test equality of header fields between input and output
    result = image.mean_img(img_4d_mni_tr2, copy_header=True)
    match_headers_keys(
        result,
        img_4d_mni_tr2,
        except_keys=["dim", "pixdim", "cal_max", "cal_min"],
    )


def test_swap_img_hemispheres(affine_eye, shape_3d_default, rng):
    # make sure input image data is not overwritten inside function
    data = rng.standard_normal(size=shape_3d_default)
    data_img = Nifti1Image(data, affine_eye)

    swap_img_hemispheres(data_img)

    assert_array_equal(get_data(data_img), data)
    # swapping operations work
    assert_array_equal(  # one turn
        get_data(swap_img_hemispheres(data_img)), data[::-1]
    )
    assert_array_equal(  # two turns -> back to original data
        get_data(swap_img_hemispheres(swap_img_hemispheres(data_img))),
        data,
    )


def test_index_img_error_3d(affine_eye):
    img_3d = Nifti1Image(np.ones((3, 4, 5)), affine_eye)
    expected_error_msg = (
        "Input data has incompatible dimensionality: "
        "Expected dimension is 4D and you provided "
        "a 3D image."
    )
    with pytest.raises(TypeError, match=expected_error_msg):
        index_img(img_3d, 0)


def test_index_img():
    img_4d, _ = generate_fake_fmri(affine=NON_EYE_AFFINE)

    fourth_dim_size = img_4d.shape[3]
    tested_indices = [
        *range(fourth_dim_size),
        slice(2, 8, 2),
        [1, 2, 3, 2],
        [],
        (np.arange(fourth_dim_size) % 3) == 1,
    ]
    for i in tested_indices:
        this_img = index_img(img_4d, i)

        expected_data_3d = get_data(img_4d)[..., i]
        assert_array_equal(get_data(this_img), expected_data_3d)
        assert_array_equal(this_img.affine, img_4d.affine)


def test_index_img_error_4d(affine_eye):
    img_4d, _ = generate_fake_fmri(affine=affine_eye)
    fourth_dim_size = img_4d.shape[3]
    for i in [
        fourth_dim_size,
        -fourth_dim_size - 1,
        [0, fourth_dim_size],
        np.repeat(True, fourth_dim_size + 1),
    ]:
        with pytest.raises(
            IndexError,
            match="out of bounds|invalid index|out of range|boolean index",
        ):
            index_img(img_4d, i)


def test_pd_index_img(rng, img_4d_rand_eye):
    # confirm indices from pandas dataframes are handled correctly
    fourth_dim_size = img_4d_rand_eye.shape[3]

    arr = rng.uniform(size=fourth_dim_size) > 0.5

    np_index_img = index_img(img_4d_rand_eye, arr)
    pd_index_img = index_img(img_4d_rand_eye, pd.DataFrame({"arr": arr}))

    assert_array_equal(get_data(np_index_img), get_data(pd_index_img))


def test_iter_img_3d_imag_error(affine_eye):
    img_3d = Nifti1Image(np.ones((3, 4, 5)), affine_eye)
    expected_error_msg = (
        "Input data has incompatible dimensionality: "
        "Expected dimension is 4D and you provided "
        "a 3D image."
    )
    with pytest.raises(TypeError, match=expected_error_msg):
        iter_img(img_3d)


def test_iter_img(tmp_path):
    img_4d, _ = generate_fake_fmri(affine=NON_EYE_AFFINE)

    for i, img in enumerate(iter_img(img_4d)):
        expected_data_3d = get_data(img_4d)[..., i]

        assert_array_equal(get_data(img), expected_data_3d)
        assert_array_equal(img.affine, img_4d.affine)

    img_4d_filename = testing.write_imgs_to_path(img_4d, file_path=tmp_path)
    for i, img in enumerate(iter_img(img_4d_filename)):
        expected_data_3d = get_data(img_4d)[..., i]

        assert_array_equal(get_data(img), expected_data_3d)
        assert_array_equal(img.affine, img_4d.affine)

    # enables to delete "img_4d_filename" on windows
    del img

    img_3d_list = list(iter_img(img_4d))
    for i, img in enumerate(iter_img(img_3d_list)):
        expected_data_3d = get_data(img_4d)[..., i]

        assert_array_equal(get_data(img), expected_data_3d)
        assert_array_equal(img.affine, img_4d.affine)

    img_3d_filenames = testing.write_imgs_to_path(
        *img_3d_list, file_path=tmp_path
    )
    for i, img in enumerate(iter_img(img_3d_filenames)):
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
    img_filename = Path(__file__).parent / "data" / "test.mgz"
    ref_img = load(img_filename)
    data = np.ones(get_data(ref_img).shape, dtype=bool)
    affine = ref_img.affine
    new_img_like(ref_img, data, affine, copy_header=False)


def test_new_img_like():
    # Give a list to new_img_like
    data = np.zeros((5, 6, 7))
    data[2:4, 1:5, 3:6] = 1
    affine = np.diag((4, 3, 2, 1))
    img = Nifti1Image(data, affine=affine)

    img2 = new_img_like([img], data)

    assert_array_equal(get_data(img), get_data(img2))

    # test_new_img_like_with_nifti2image_copy_header
    img_nifti2 = Nifti2Image(data, affine=affine)

    img2_nifti2 = new_img_like([img_nifti2], data, copy_header=True)

    assert_array_equal(get_data(img_nifti2), get_data(img2_nifti2))


def test_new_img_like_accepts_paths(affine_eye, tmp_path, rng):
    """Check that new_img_like can accept instances of pathlib.Path."""
    nifti_path = tmp_path / "sample.nii"
    assert isinstance(nifti_path, Path)

    data = rng.random((10, 10, 10))
    img = Nifti1Image(data, affine_eye)
    img.to_filename(nifti_path)

    new_data = rng.random((10, 10, 10))
    new_img = new_img_like(nifti_path, new_data)
    assert new_img.shape == (10, 10, 10)

    # Check that list of pathlib.Path also accepted
    new_img = new_img_like([nifti_path], new_data)
    assert new_img.shape == (10, 10, 10)


def test_new_img_like_non_iterable_header(rng):
    """Tests that when an niimg's header is not iterable \
       and it is set to be copied, an error is not raised.
    """
    fake_fmri_data = rng.uniform(size=_shape_4d_default())
    fake_affine = rng.uniform(size=(4, 4))
    fake_spatial_image = spatialimages.SpatialImage(
        fake_fmri_data, fake_affine
    )

    assert new_img_like(
        fake_spatial_image, data=fake_fmri_data, copy_header=True
    )


@pytest.mark.parametrize("no_int64_nifti", ["allow for this test"])
def test_new_img_like_int64(shape_3d_default):
    img = generate_labeled_regions(shape=shape_3d_default, n_regions=2)

    data = get_data(img).astype("int32")

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        new_img = new_img_like(img, data)
    assert get_data(new_img).dtype == "int32"

    data = data.astype("int64")

    with pytest.warns(UserWarning, match=r".*array.*contains.*64.*"):
        new_img = new_img_like(img, data)
    assert get_data(new_img).dtype == "int32"

    data[:] = 2**40

    with pytest.warns(UserWarning, match=r".*64.*too large.*"):
        new_img = new_img_like(img, data, copy_header=True)
    assert get_data(new_img).dtype == "int64"


def test_validity_threshold_value_in_threshold_img(shape_3d_default):
    """Check that invalid values to threshold_img's threshold parameter \
       raise Exceptions.
    """
    maps, _ = generate_maps(shape_3d_default, n_regions=2)

    # testing to raise same error when threshold=None case
    with pytest.raises(
        TypeError,
        match="threshold should be either a number or a string",
    ):
        threshold_img(maps, threshold=None, copy_header=True)

    invalid_threshold_values = ["90t%", "s%", "t", "0.1"]
    name = "threshold"
    for thr in invalid_threshold_values:
        with pytest.raises(
            ValueError,
            match=f"{name}.+should be a number followed by the percent sign",
        ):
            threshold_img(maps, threshold=thr, copy_header=True)


def test_threshold_img(affine_eye):
    """Smoke test for threshold_img with valid threshold inputs."""
    shape = (10, 20, 30)
    maps, _ = generate_maps(shape, n_regions=4)
    mask_img = Nifti1Image(np.ones((shape), dtype=np.int8), affine_eye)

    for img in iter_img(maps):
        # when threshold is a float value
        threshold_img(img, threshold=0.8, copy_header=True)

        # when we provide mask image
        threshold_img(img, threshold=1, mask_img=mask_img, copy_header=True)

        # when threshold is a percentile
        threshold_img(img, threshold="2%", copy_header=True)


@pytest.mark.parametrize(
    "threshold,two_sided,cluster_threshold,expected",
    [
        (2, False, 0, [0, 4, 5]),  # cluster with values > 2
        (2, True, 0, [-5, -4, 0, 4, 5]),  # cluster with |values| > 2
        (2, True, 5, [-4, 0, 4]),  # cluster: |values| > 2 & size > 5
        (2, True, 5, [-4, 0, 4]),  # cluster: |values| > 2 & size > 5
        (0.5, True, 5, [-4, -1, 0, 1, 4]),  # cluster: |values| > 0.5 & size>5
        (0.5, False, 5, [0, 1, 4]),  # cluster: values > 0.5 & size > 5
    ],
)
def test_threshold_img_with_cluster_threshold(
    stat_img_test_data, threshold, two_sided, cluster_threshold, expected
):
    """Check that passing specific threshold and cluster threshold values \
    only gives cluster the right number of voxels with the right values.
    """
    thr_img = threshold_img(
        img=stat_img_test_data,
        threshold=threshold,
        two_sided=two_sided,
        cluster_threshold=cluster_threshold,
        copy=True,
        copy_header=True,
    )

    assert np.array_equal(np.unique(thr_img.get_fdata()), np.array(expected))


def test_threshold_img_threshold_n_clusters(stat_img_test_data):
    """With a cluster threshold of 5 we get 8 clusters with |values| > 2 \
       and cluster sizes > 5.
    """
    thr_img = threshold_img(
        img=stat_img_test_data,
        threshold=2,
        two_sided=True,
        cluster_threshold=5,
        copy=True,
        copy_header=True,
    )

    assert np.sum(thr_img.get_fdata() == 4) == 8


def test_threshold_img_copy(img_4d_ones_eye):
    """Test the behavior of threshold_img's copy parameter."""
    # Check that copy does not mutate. It returns modified copy.
    thresholded = threshold_img(
        img_4d_ones_eye, 2, copy_header=True
    )  # threshold 2 > 1

    # Original img_ones should have all ones.
    assert_array_equal(get_data(img_4d_ones_eye), np.ones(_shape_4d_default()))
    # Thresholded should have all zeros.
    assert_array_equal(get_data(thresholded), np.zeros(_shape_4d_default()))

    # Check that not copying does mutate.
    img_to_mutate = img_4d_ones_eye

    thresholded = threshold_img(img_to_mutate, 2, copy=False, copy_header=True)

    # Check that original mutates
    assert_array_equal(get_data(img_to_mutate), np.zeros(_shape_4d_default()))
    # And that returned value is also thresholded.
    assert_array_equal(get_data(img_to_mutate), get_data(thresholded))


def test_isnan_threshold_img_data(affine_eye, shape_3d_default):
    """Check threshold_img converges properly when input image has nans."""
    maps, _ = generate_maps(shape_3d_default, n_regions=2)
    data = get_data(maps)
    data[:, :, 0] = np.nan

    maps_img = Nifti1Image(data, affine_eye)

    threshold_img(maps_img, threshold=0.8, copy_header=True)


def test_threshold_img_copied_header(img_4d_mni_tr2):
    # Test equality of header fields between input and output
    result = threshold_img(img_4d_mni_tr2, threshold=0.5, copy_header=True)
    # only the min value should be different
    match_headers_keys(
        result,
        img_4d_mni_tr2,
        except_keys=["cal_min"],
    )
    # min value should be 0 in the result
    assert result.header["cal_min"] == 0


def test_math_img_exceptions(affine_eye, img_4d_ones_eye):
    img1 = img_4d_ones_eye
    img2 = Nifti1Image(np.zeros((10, 20, 10, 10)), affine_eye)
    img3 = img_4d_ones_eye
    img4 = Nifti1Image(np.ones(_shape_4d_default()), affine_eye * 2)

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

    # Test copy_header_from parameter
    # Copying header from 4d image to a result that is 3d should raise a
    # ValueError
    formula = "np.mean(img1, axis=-1) - np.mean(img3, axis=-1)"
    with pytest.raises(ValueError, match="Cannot copy the header."):
        math_img(formula, img1=img1, img3=img3, copy_header_from="img1")

    # Passing an 'img*' variable (to copy_header_from) that is not in the
    # formula or an img* argument should raise a KeyError exception.
    with pytest.raises(KeyError):
        math_img(formula, img1=img1, img3=img3, copy_header_from="img2")


def test_math_img(
    affine_eye, img_4d_ones_eye, img_4d_zeros_eye, shape_3d_default, tmp_path
):
    img1 = img_4d_ones_eye
    img2 = img_4d_zeros_eye
    expected_result = Nifti1Image(np.ones(shape_3d_default), affine_eye)

    formula = "np.mean(img1, axis=-1) - np.mean(img2, axis=-1)"
    for create_files in (True, False):
        imgs = testing.write_imgs_to_path(
            img1, img2, file_path=tmp_path, create_files=create_files
        )
        result = math_img(formula, img1=imgs[0], img2=imgs[1])
        assert_array_equal(get_data(result), get_data(expected_result))
        assert_array_equal(result.affine, expected_result.affine)
        assert result.shape == expected_result.shape


def test_math_img_copy_default_header(
    img_4d_ones_eye_default_header, img_4d_ones_eye_tr2
):
    # case where data values are not changed and header values are not copied
    # the result should have default header values
    formula_no_change = "img * 1"
    # using img_4d_ones_eye_tr2 with edited header in the formula
    result = math_img(
        formula_no_change, img=img_4d_ones_eye_tr2, copy_header_from=None
    )
    # header values should instead match default header values
    assert result.header == img_4d_ones_eye_default_header.header


def test_math_img_copied_header_from_img(img_4d_mni_tr2):
    # case where data values are not changed but header values are copied
    # the result should have the same header values as the given image
    formula_no_change = "img * 1"
    result = math_img(
        formula_no_change, img=img_4d_mni_tr2, copy_header_from="img"
    )
    # all header values should be the same
    assert result.header == img_4d_mni_tr2.header


def test_math_img_copied_header_data_values_changed(
    img_4d_ones_eye_default_header, img_4d_ones_eye_tr2
):
    # case where data values are changed and header values are copied from one
    # of the input images
    # the result should have the same header values as img_4d_ones_eye_tr2,
    # except for cal_max and cal_min that should be different
    formula_change_min_max = "img1 - img2"
    result = math_img(
        formula_change_min_max,
        img1=img_4d_ones_eye_default_header,
        img2=img_4d_ones_eye_tr2,
        copy_header_from="img2",
    )
    for key in img_4d_ones_eye_tr2.header:
        # cal_max and cal_min should be different in result
        if key in ["cal_max", "cal_min"]:
            assert result.header[key] != img_4d_ones_eye_tr2.header[key]
        # other header values should be the same
        elif isinstance(result.header[key], np.ndarray):
            assert_array_equal(
                result.header[key], img_4d_ones_eye_tr2.header[key]
            )
        else:
            assert result.header[key] == img_4d_ones_eye_tr2.header[key]


def test_binarize_img(img_4d_rand_eye):
    # Test that all output values are 1.
    img1 = binarize_img(img_4d_rand_eye, copy_header=True)

    assert_array_equal(np.unique(img1.dataobj), np.array([1]))

    # Test that it works with threshold
    img2 = binarize_img(img_4d_rand_eye, threshold=0.5, copy_header=True)

    assert_array_equal(np.unique(img2.dataobj), np.array([0, 1]))
    # Test that manual binarization equals binarize_img results.
    img3 = copy_img(img_4d_rand_eye)
    img3.dataobj[img_4d_rand_eye.dataobj < 0.5] = 0
    img3.dataobj[img_4d_rand_eye.dataobj >= 0.5] = 1

    assert_array_equal(img2.dataobj, img3.dataobj)


def test_binarize_negative_img(img_4d_rand_eye, rng):
    # Test option to use original or absolute values
    img_data = img_4d_rand_eye.dataobj
    # Create a mask for half of the values and make them negative
    neg_mask = rng.choice(
        [True, False], size=img_4d_rand_eye.shape, p=[0.5, 0.5]
    )
    img_data[neg_mask] *= -1
    img = new_img_like(img_4d_rand_eye, img_data)
    # Binarize using original and absolute values
    img_original = binarize_img(
        img, threshold=0, two_sided=False, copy_header=True
    )
    img_absolute = binarize_img(
        img, threshold=0, two_sided=True, copy_header=True
    )
    # Check that all values are 1 for absolute valued threshold
    assert_array_equal(np.unique(img_absolute.dataobj), np.array([1]))
    # Check that binarized image contains 0 and 1 for original threshold
    assert_array_equal(np.unique(img_original.dataobj), np.array([0, 1]))


def test_binarize_img_copied_header(img_4d_mni_tr2):
    # Test equality of header fields between input and output
    result = binarize_img(img_4d_mni_tr2, threshold=0.5, copy_header=True)
    # only the min value should be different
    match_headers_keys(
        result,
        img_4d_mni_tr2,
        except_keys=["cal_min", "cal_max"],
    )
    # min value should be 0 in the result
    assert result.header["cal_min"] == 0
    # max value should be 1 in the result
    assert result.header["cal_max"] == 1


def test_binarize_img_no_userwarning(img_4d_rand_eye):
    # Test that a UserWarning is not thrown for a float64 img
    with warnings.catch_warnings():
        warnings.simplefilter("error", category=UserWarning)
        binarize_img(img_4d_rand_eye)


@pytest.mark.parametrize(
    "func, input_img",
    [
        (binarize_img, "img_4d_mni_tr2"),
        (crop_img, "img_4d_mni_tr2"),
        (mean_img, "img_4d_mni_tr2"),
        (threshold_img, "img_4d_mni_tr2"),
    ],
)
def test_warning_copy_header_false(request, func, input_img):
    # Use the request fixture to get the actual fixture value
    actual_input_img = request.getfixturevalue(input_img)
    with pytest.warns(FutureWarning, match="From release 0.13.0 onwards*"):
        if func is threshold_img:
            func(actual_input_img, threshold=0.5, copy_header=False)
        else:
            func(actual_input_img, copy_header=False)


def test_clean_img(affine_eye, shape_3d_default, rng):
    data = rng.standard_normal(size=(10, 10, 10, 100)) + 0.5
    data_flat = data.T.reshape(100, -1)
    data_img = Nifti1Image(data, affine_eye)

    with pytest.raises(ValueError, match="t_r.*must be specified"):
        clean_img(data_img, t_r=None, low_pass=0.1)

    data_img_ = clean_img(
        data_img, detrend=True, standardize=False, low_pass=0.1, t_r=1.0
    )
    data_flat_ = signal.clean(
        data_flat, detrend=True, standardize=False, low_pass=0.1, t_r=1.0
    )

    assert_almost_equal(get_data(data_img_).T.reshape(100, -1), data_flat_)
    # if NANs
    data[:, 9, 9] = np.nan
    # if infinity
    data[:, 5, 5] = np.inf
    nan_img = Nifti1Image(data, affine_eye)

    clean_im = clean_img(nan_img, ensure_finite=True)

    assert np.any(np.isfinite(get_data(clean_im)))

    # test_clean_img_passing_nifti2image
    data_img_nifti2 = Nifti2Image(data, affine_eye)

    clean_img(
        data_img_nifti2, detrend=True, standardize=False, low_pass=0.1, t_r=1.0
    )

    # if mask_img
    img, mask_img = generate_fake_fmri(shape=shape_3d_default, length=10)

    data_img_mask_ = clean_img(img, mask_img=mask_img)

    # Checks that output with full mask and without is equal
    data_img_ = clean_img(img)

    assert_almost_equal(get_data(data_img_), get_data(data_img_mask_))


@pytest.mark.parametrize("create_files", [True, False])
def test_largest_cc_img(create_files, tmp_path):
    """Check the extraction of the largest connected component, for niftis.

    Similar to smooth_img tests for largest connected_component_img, here also
    only the added features for largest_connected_component are tested.
    """
    # Test whether dimension of 3Dimg and list of 3Dimgs are kept.
    img1, img2, shapes = _make_largest_cc_img_test_data()

    imgs = testing.write_imgs_to_path(
        img1, img2, file_path=tmp_path, create_files=create_files
    )
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


@pytest.mark.parametrize("create_files", [True, False])
def test_largest_cc_img_non_native_endian_type(create_files, tmp_path):
    # Test whether dimension of 3Dimg and list of 3Dimgs are kept.
    img1, img2, shapes = _make_largest_cc_img_test_data()

    # tests adapted to non-native endian data dtype
    img1_change_dtype = Nifti1Image(
        get_data(img1).astype(">f8"), affine=img1.affine
    )
    img2_change_dtype = Nifti1Image(
        get_data(img2).astype(">f8"), affine=img2.affine
    )

    imgs = testing.write_imgs_to_path(
        img1_change_dtype,
        img2_change_dtype,
        file_path=tmp_path,
        create_files=create_files,
    )
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
    assert_equal(get_data(out_native), get_data(out_non_native))


def test_largest_cc_img_error(shape_3d_default):
    # Test whether 4D Nifti throws the right error.
    img_4D = generate_fake_fmri(shape_3d_default)

    with pytest.raises(DimensionError, match="dimension"):
        largest_connected_component_img(img_4D)


def test_new_img_like_mgh_image(affine_eye, shape_3d_default):
    data = np.zeros(shape_3d_default, dtype=np.uint8)
    niimg = MGHImage(dataobj=data, affine=affine_eye)

    new_img_like(niimg, data.astype(float), niimg.affine, copy_header=True)


@pytest.mark.parametrize("image", [MGHImage, AnalyzeImage])
def test_new_img_like_boolean_data(affine_eye, image, shape_3d_default, rng):
    """Checks defaulting boolean input data to np.uint8 dtype is valid \
       forencoding with nibabel image classes MGHImage and AnalyzeImage.
    """
    data = rng.standard_normal(shape_3d_default).astype("uint8")
    in_img = image(dataobj=data, affine=affine_eye)

    out_img = new_img_like(in_img, data=in_img.get_fdata() > 0.5)

    assert get_data(out_img).dtype == "uint8"


def test_clean_img_sample_mask(img_4d_rand_eye, shape_4d_default):
    """Check sample mask can be passed as a kwarg and used correctly."""
    length = shape_4d_default[3]
    confounds = _basic_confounds(length)
    # exclude last time point
    sample_mask = np.arange(length - 1)

    img = image.clean_img(
        img_4d_rand_eye,
        confounds=confounds,
        clean__sample_mask=sample_mask,
    )
    assert img.shape == (*shape_4d_default[:3], length - 1)


def test_clean_img_sample_mask_mask_img(shape_3d_default):
    """Check sample_mask and mask_img can be correctly used together."""
    length = 10
    confounds = _basic_confounds(length)
    img_4d, mask_img = generate_fake_fmri(
        shape=shape_3d_default, length=length
    )

    # exclude last time point
    sample_mask = np.arange(length - 1)

    # test with sample mask
    img = image.clean_img(
        img_4d,
        confounds=confounds,
        mask_img=mask_img,
        clean__sample_mask=sample_mask,
    )
    assert img.shape == (*shape_3d_default, length - 1)


def test_concat_niimgs_errors(affine_eye, shape_3d_default):
    img1 = Nifti1Image(np.ones(shape_3d_default), affine_eye)
    img2 = Nifti1Image(np.ones(shape_3d_default), 2 * affine_eye)
    img4d = Nifti1Image(np.ones((*shape_3d_default, 2)), affine_eye)

    # check error for non-forced but necessary resampling
    with pytest.raises(ValueError, match="Field of view of image"):
        concat_imgs([img1, img2], auto_resample=False)

    # Regression test for #601.
    # Dimensionality of first image was not checked properly.
    _dimension_error_msg = (
        "Input data has incompatible dimensionality: "
        "Expected dimension is 4D and you provided "
        "a list of 4D images \\(5D\\)"
    )
    with pytest.raises(DimensionError, match=_dimension_error_msg):
        concat_imgs([img4d], ensure_ndim=4)

    with pytest.raises(DimensionError, match=_dimension_error_msg):
        concat_imgs([img1, img4d])

    img5d = Nifti1Image(np.ones((2, 2, 2, 2, 2)), affine_eye)
    with pytest.raises(
        TypeError,
        match="Concatenated images must be 3D or 4D. "
        "You gave a list of 5D images",
    ):
        concat_imgs([img5d, img5d])


def test_concat_niimgs(affine_eye, tmp_path):
    # create images different in affine and 3D/4D shape
    shape = (10, 11, 12)
    img1 = Nifti1Image(np.ones(shape), affine_eye)
    img2 = Nifti1Image(np.zeros(shape), affine_eye)

    shape2 = (12, 11, 10)
    img1b = Nifti1Image(np.ones(shape2), affine_eye)

    shape3 = (11, 22, 33)
    img1c = Nifti1Image(np.ones(shape3), affine_eye)

    # check basic concatenation with equal shape/affine
    # verbose for coverage
    concatenated = concat_imgs((img1, img2, img1), verbose=1)

    # smoke-test auto_resample
    concatenated = concat_imgs((img1, img1b, img1c), auto_resample=True)
    assert concatenated.shape == (*img1.shape, 3)

    # test list of 4D niimgs as input
    img1.to_filename(tmp_path / "1.nii")
    img2.to_filename(tmp_path / "2.nii")
    concatenated = concat_imgs(tmp_path / "*")
    assert_array_equal(get_data(concatenated)[..., 0], get_data(img1))
    assert_array_equal(get_data(concatenated)[..., 1], get_data(img2))


def test_concat_niimg_dtype(affine_eye):
    shape = [2, 3, 4]
    vols = [
        Nifti1Image(np.zeros([*shape, n_scans]).astype(np.int16), affine_eye)
        for n_scans in [1, 5]
    ]
    nimg = concat_imgs(vols)
    assert get_data(nimg).dtype == np.float32
    nimg = concat_imgs(vols, dtype=None)
    assert get_data(nimg).dtype == np.int16


def nifti_generator(buffer):
    for _ in range(10):
        buffer.append(_img_3d_rand())
        yield buffer[-1]


def test_iterator_generator(img_3d_rand_eye):
    # Create a list of random images
    list_images = [img_3d_rand_eye for _ in range(10)]
    cc = concat_imgs(list_images)
    assert cc.shape[-1] == 10
    assert_array_almost_equal(get_data(cc)[..., 0], get_data(list_images[0]))

    # Same with iteration
    i = image.iter_img(list_images)
    cc = concat_imgs(i)
    assert cc.shape[-1] == 10
    assert_array_almost_equal(get_data(cc)[..., 0], get_data(list_images[0]))

    # Now, a generator
    b = []
    g = nifti_generator(b)
    cc = concat_imgs(g)
    assert cc.shape[-1] == 10
    assert len(b) == 10


def test_copy_img():
    with pytest.raises(ValueError, match="Input value is not an image"):
        copy_img(3)


def test_copy_img_side_effect(img_4d_ones_eye):
    hash1 = joblib.hash(img_4d_ones_eye)
    copy_img(img_4d_ones_eye)
    hash2 = joblib.hash(img_4d_ones_eye)
    assert hash1 == hash2
