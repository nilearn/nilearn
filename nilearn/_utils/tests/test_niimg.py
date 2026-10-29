import os
import warnings
from pathlib import Path
from tempfile import mkstemp

import joblib
import numpy as np
import pytest
from nibabel import Nifti1Header, Nifti1Image, load
from numpy.testing import assert_almost_equal, assert_equal

from nilearn._utils.niimg import (
    _get_data,
    ensure_finite_data,
    has_non_finite,
    img_data_dtype,
    is_binary_niimg,
    load_niimg,
    repr_niimgs,
    trim_maps,
)
from nilearn._utils.numpy_conversions import get_target_dtype
from nilearn._utils.testing import write_imgs_to_path
from nilearn.image import get_data, new_img_like

N_REGIONS = 8


@pytest.fixture
def img1(affine_eye) -> Nifti1Image:
    data = np.ones((2, 2, 2, 2))
    return Nifti1Image(data, affine=affine_eye)


@pytest.fixture
def img_binary(affine_eye, has_inf, has_nan, non_bin) -> Nifti1Image:
    data = np.ones((3, 3, 3, 3))

    if has_inf:
        data[0, 0, 0, 0] = np.inf
    if has_nan:
        data[1, 1, 1, 1] = np.nan
    if non_bin:
        data[2, 2, 2, 2] = 5
    return Nifti1Image(data, affine=affine_eye)


def test_new_img_like_side_effect(img1):
    hash1 = joblib.hash(img1)
    new_img_like(img1, np.ones((2, 2, 2, 2)), img1.affine.copy())
    hash2 = joblib.hash(img1)
    assert hash1 == hash2


def test_get_target_dtype(affine_eye):
    img = Nifti1Image(np.ones((2, 2, 2), dtype=np.float64), affine=affine_eye)
    assert get_data(img).dtype.kind == "f"
    dtype_kind_float = get_target_dtype(
        get_data(img).dtype, target_dtype="auto"
    )
    assert dtype_kind_float == np.float32
    # Passing dtype or header is required when using int64
    # https://nipy.org/nibabel/changelog.html#api-changes-and-deprecations
    hdr = Nifti1Header()
    hdr.set_data_dtype(np.int64)
    data = np.ones((2, 2, 2), dtype=np.int64)
    img2 = Nifti1Image(data, affine=affine_eye, header=hdr)
    assert get_data(img2).dtype.kind == img2.get_data_dtype().kind == "i"
    dtype_kind_int = get_target_dtype(
        get_data(img2).dtype, target_dtype="auto"
    )
    assert dtype_kind_int == np.int32


def test_img_data_dtype(rng, affine_eye, tmp_path):
    # Ignoring complex, binary, 128+ bit, RGBA
    nifti1_dtypes = (
        np.uint8,
        np.uint16,
        np.uint32,
        np.uint64,
        np.int8,
        np.int16,
        np.int32,
        np.float32,
        np.float64,
    )
    dtype_matches = []
    # Passing dtype or header is required when using int64
    # https://nipy.org/nibabel/changelog.html#api-changes-and-deprecations
    hdr = Nifti1Header()
    for logical_dtype in nifti1_dtypes:
        dataobj = rng.uniform(0, 255, (2, 2, 2)).astype(logical_dtype)
        for on_disk_dtype in nifti1_dtypes:
            hdr.set_data_dtype(on_disk_dtype)
            img = Nifti1Image(dataobj, affine_eye, header=hdr)
            img.to_filename(tmp_path / "test.nii")
            loaded = load(tmp_path / "test.nii")
            # To verify later that sometimes these differ meaningfully
            dtype_matches.append(
                loaded.get_data_dtype() == img_data_dtype(loaded)
            )
            # TODO (numpy > 2.*) deal with this DeprecationWarning
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=DeprecationWarning)
                assert np.array(loaded.dataobj).dtype == img_data_dtype(loaded)
    # Verify that the distinction is worth making
    assert any(dtype_matches)
    assert not all(dtype_matches)


@pytest.mark.thread_unsafe
def test_load_niimg(img1, tmp_path):
    filename = write_imgs_to_path(img1, file_path=tmp_path, create_files=True)
    filename = Path(filename)
    load_niimg(filename)


def test_repr_niimgs():
    """Test repr_niimgs.

    - with file path
    - shortening long names (default)
    - explicit shortening of long names
    - explicit shortening of list 3 of long names
    - explicit shortening of list longer than 3 names
    """
    # Tests with file path
    assert repr_niimgs("test") == "test"
    assert repr_niimgs("test", shorten=False) == "test"

    # Shortening long names by default
    long_name = "this-is-a-very-long-name-for-a-nifti-file.nii"
    short_name = "this-is-a-very-lon..."
    assert repr_niimgs(long_name) == short_name

    # Explicit shortening of long names
    assert repr_niimgs(long_name, shorten=True) == short_name

    # Lists of long names up to length 3
    list_of_size_3 = [
        "this-is-a-very-long-name-for-a-nifti-file.nii",
        "this-is-another-very-long-name-for-a-nifti-file.nii",
        "this-is-again-another-very-long-name-for-a-nifti-file.nii",
    ]
    # Explicit shortening, all 3 names are displayed, but shortened
    shortened_rep_list_of_size_3 = (
        "[\n "
        + "this-is-a-very-lon..., "
        + "this-is-another-ve..., "
        + "this-is-again-anot...,"
        + "\n]"
    )

    assert (
        repr_niimgs(list_of_size_3, shorten=True)
        == shortened_rep_list_of_size_3
    )

    # Lists longer than 3
    # Small names - Explicit shortening
    long_list_small_names = ["test", "retest", "reretest", "rereretest"]
    shortened_rep_long_list_small_names = (
        "[\n test,\n         ...\n rereretest,\n]"
    )

    assert (
        repr_niimgs(long_list_small_names, shorten=True)
        == shortened_rep_long_list_small_names
    )

    # Long names - Explicit shortening
    list_of_size_4 = [
        *list_of_size_3,
        "this-is-again-another-super-very-long-name-for-a-nifti-file.nii",
    ]
    shortened_rep_long_list_long_names = (
        "[\n this-is-a-very-lon...,\n         ...\n this-is-again-anot...,\n]"
    )

    assert (
        repr_niimgs(list_of_size_4, shorten=True)
        == shortened_rep_long_list_long_names
    )


def test_repr_niimgs_force_long_names():
    """Test repr_niimgs without shortening."""
    long_name = "this-is-a-very-long-name-for-a-nifti-file.nii"
    # Force long display of long names
    assert repr_niimgs(long_name, shorten=False) == long_name

    # Tests with list of file paths
    assert repr_niimgs(["test", "retest"]) == "[\n test, retest,\n]"
    assert (
        repr_niimgs(["test", "retest"], shorten=False)
        == "[\n test, retest,\n]"
    )

    # Force display, all 3 names are displayed
    list_of_size_3 = [
        "this-is-a-very-long-name-for-a-nifti-file.nii",
        "this-is-another-very-long-name-for-a-nifti-file.nii",
        "this-is-again-another-very-long-name-for-a-nifti-file.nii",
    ]
    long_rep_list_of_size_3 = (
        "[\n this-is-a-very-long-name-for-a-nifti-file.nii,"
        " this-is-another-very-long-name-for-a-nifti-file.nii,"
        " this-is-again-another-very-long-name-for-a-nifti-file.nii,\n]"
    )
    assert (
        repr_niimgs(list_of_size_3, shorten=False) == long_rep_list_of_size_3
    )

    long_list_small_names = ["test", "retest", "reretest", "rereretest"]
    long_rep_long_list_small_names = (
        "[\n test,\n retest,\n reretest,\n rereretest,\n]"
    )

    assert (
        repr_niimgs(long_list_small_names, shorten=False)
        == long_rep_long_list_small_names
    )

    # Long names - Force full display in pretty print style for readability
    list_of_size_4 = [
        *list_of_size_3,
        "this-is-again-another-super-very-long-name-for-a-nifti-file.nii",
    ]
    long_rep_long_list_long_names = (
        long_rep_list_of_size_3[:-2].replace(",", ",\n")
        + " this-is-again-another-super-very-long-name-for-a-nifti-file.nii,"
        + "\n]"
    )

    assert (
        repr_niimgs(list_of_size_4, shorten=False)
        == long_rep_long_list_long_names
    )


def test_repr_niimgs_with_niimg_pathlib():
    """Test repr_niimgs with Path."""
    # Tests with pathlib
    # Case with very long path and small filename
    long_path = Path("/this/is/a/fake/long/path/to/file.nii")
    short_path = Path(".../path/to/file.nii")
    assert repr_niimgs(long_path, shorten=True) == str(short_path)
    assert repr_niimgs(long_path, shorten=False) == str(long_path)

    # Case with very long path but very long filename
    long_path_long_name = Path(
        "/this/is/a/fake/long/path/to/my_file_with_a_very_long_name.nii"
    )
    short_name = "my_file_with_a_ver..."
    assert repr_niimgs(long_path_long_name, shorten=True) == short_name
    assert repr_niimgs(long_path_long_name, shorten=False) == str(
        long_path_long_name
    )

    # Case with lists
    list_of_paths = [
        Path("/this/is/a/fake/long/path/to/file.nii"),
        Path("/this/is/a/fake/long/path/to/another/file2.nii"),
        Path("/again/another/fake/long/path/to/file3.nii"),
        Path("/this/is/a/fake/long/path/to/a-very-long-file-name.nii"),
    ]

    shortened_list_of_paths = (
        f"[\n ...{Path('/path/to/file.nii')!s},\n"
        f"         ...\n"
        f" a-very-long-file-n...,\n]"
    )

    assert repr_niimgs(list_of_paths, shorten=True) == shortened_list_of_paths
    long_list_of_paths = ",\n ".join([str(_) for _ in list_of_paths])
    long_list_of_paths = f"[\n {long_list_of_paths},\n]"
    assert repr_niimgs(list_of_paths, shorten=False) == long_list_of_paths


@pytest.mark.thread_unsafe
@pytest.mark.parametrize("shorten", [True, False])
def test_repr_niimgs_with_niimg(
    shorten, tmp_path, affine_eye, img_3d_ones_eye, shape_3d_default
):
    """Test repr_niimgs with actual image objects.

    Shorten has no effect in this case.
    """
    assert repr_niimgs(img_3d_ones_eye, shorten=shorten).replace(
        "10L", "10"
    ) == (
        f"{img_3d_ones_eye.__class__.__name__}(\nshape={shape_3d_default!r},\naffine={affine_eye!r}\n)"
    )

    # Add filename long enough to qualify for shortening
    fd, tmpimg1 = mkstemp(suffix="_very_long.nii", dir=str(tmp_path))
    os.close(fd)
    img_3d_ones_eye.to_filename(tmpimg1)
    class_name = img_3d_ones_eye.__class__.__name__
    filename = Path(img_3d_ones_eye.get_filename())
    assert (
        repr_niimgs(img_3d_ones_eye, shorten=False)
        == f"{class_name}('{filename}')"
    )
    assert (
        repr_niimgs(img_3d_ones_eye, shorten=True)
        == f"{class_name}('{Path(filename).name[:18]}...')"
    )


@pytest.mark.parametrize("shorten", [True, False])
def test_repr_niimgs_with_1d_surface_image(surf_img_1d, shorten):
    img_repr = "<SurfaceImage (9,)>"

    assert repr(surf_img_1d) == img_repr
    assert repr_niimgs(surf_img_1d, shorten=shorten) == img_repr


@pytest.mark.parametrize("shorten", [True, False])
def test_repr_niimgs_with_2d_surface_image(surf_img_2d, shorten):
    img = surf_img_2d(n_samples=2)
    img_repr = "<SurfaceImage (9, 2)>"

    assert repr(img) == img_repr
    assert repr_niimgs(img, shorten=shorten) == img_repr


@pytest.mark.parametrize("shorten", [True, False])
def test_repr_niimgs_with_surface_image_list(
    surf_img_1d, surf_img_2d, shorten
):
    imgs = [surf_img_1d, surf_img_2d(n_samples=2)]

    expected = f"[\n {imgs[0]!r}, {imgs[1]!r},\n]"
    assert repr_niimgs(imgs, shorten=shorten) == expected


@pytest.mark.parametrize("shorten", [True, False])
def test_repr_niimgs_with_long_surface_image_list(
    surf_img_1d, surf_img_2d, shorten
):
    imgs = [
        surf_img_1d,
        surf_img_2d(n_samples=2),
        surf_img_1d,
        surf_img_2d(n_samples=3),
    ]

    if shorten:
        expected = f"[\n {imgs[0]!r},\n         ...\n {imgs[-1]!r},\n]"
    else:
        image_reprs = ",\n ".join(repr(img) for img in imgs)
        expected = f"[\n {image_reprs},\n]"
    assert repr_niimgs(imgs, shorten=shorten) == expected


@pytest.mark.parametrize(
    "has_inf,has_nan,non_bin,expected",
    [
        (False, False, False, False),
        (True, False, False, True),
        (False, True, False, True),
        (False, False, True, False),
        (True, True, True, True),
    ],
)
def test_has_non_finite(img_binary, expected):
    """Test nilearn._utils.niimg.has_non_finite."""
    data = _get_data(img_binary)
    has_not_finite, _ = has_non_finite(data)
    assert has_not_finite == expected


@pytest.mark.parametrize("has_inf,has_nan,non_bin", [(True, True, False)])
def test_ensure_finite_data(img_binary):
    data = _get_data(img_binary)
    data_returned = ensure_finite_data(data)
    expected_data = np.ones((3, 3, 3, 3))
    expected_data[0, 0, 0, 0] = 0
    expected_data[1, 1, 1, 1] = 0

    assert np.array_equal(data_returned, expected_data)


@pytest.mark.parametrize(
    "has_inf,has_nan,non_bin,accept_non_finite,expected",
    [
        (False, False, False, True, True),
        (False, False, False, False, True),
        (True, False, False, True, True),
        (True, False, False, False, False),
        (False, True, False, True, True),
        (False, True, False, False, False),
        (True, True, False, True, True),
        (True, True, False, False, False),
        (False, False, True, True, False),
        (False, False, True, False, False),
        (True, True, True, True, False),
        (True, True, True, False, False),
    ],
)
def test_is_binary_niimg(img_binary, accept_non_finite, expected):
    """Test nilearn._utils.niimg.is_binary_niimg."""
    assert (
        is_binary_niimg(img_binary, accept_non_finite=accept_non_finite)
        == expected
    )


def test_trim_maps_all_regions(shape_3d_default):
    """Use mask intersecting all regions."""
    maps_data = np.zeros((*shape_3d_default, N_REGIONS), dtype=np.float32)
    h0, h1, h2 = (s // 2 for s in shape_3d_default)
    maps_data[:h0, :h1, :h2, 0] = 1
    maps_data[:h0, :h1, h2:, 1] = 1.1
    maps_data[:h0, h1:, :h2, 2] = 1
    maps_data[:h0, h1:, h2:, 3] = 0.5
    maps_data[h0:, :h1, :h2, 4] = 1
    maps_data[h0:, :h1, h2:, 5] = 1.4
    maps_data[h0:, h1:, :h2, 6] = 1
    maps_data[h0:, h1:, h2:, 7] = 1

    mask_data = np.zeros(shape_3d_default, dtype=np.int8)
    mask_data[1:-1, 1:-1, 1:-1] = 1

    maps_i, maps_i_mask, maps_i_indices = trim_maps(maps_data, mask_data)

    assert maps_i.flags["F_CONTIGUOUS"]
    assert len(maps_i_indices) == maps_i.shape[-1]
    assert maps_i.shape == maps_data.shape

    maps_i_correct = maps_data.copy()
    maps_i_correct[np.logical_not(mask_data), :] = 0
    assert_almost_equal(maps_i_correct, maps_i)

    assert_equal(mask_data, maps_i_mask)
    assert_equal(np.asarray(list(range(8))), maps_i_indices)


def test_trim_maps_half_regions(shape_3d_default):
    """Use mask intersecting half of the regions."""
    # maps
    maps_data = np.zeros((*shape_3d_default, N_REGIONS), dtype=np.float32)
    h0, h1, h2 = (s // 2 for s in shape_3d_default)
    maps_data[:h0, :h1, :h2, 0] = 1
    maps_data[:h0, :h1, h2:, 1] = 1.1
    maps_data[:h0, h1:, :h2, 2] = 1
    maps_data[:h0, h1:, h2:, 3] = 0.5
    maps_data[h0:, :h1, :h2, 4] = 1
    maps_data[h0:, :h1, h2:, 5] = 1.4
    maps_data[h0:, h1:, :h2, 6] = 1
    maps_data[h0:, h1:, h2:, 7] = 1
    maps_data[1, 1, 1, 0] = 0  # remove one point inside mask

    mask_data = np.zeros(shape_3d_default, dtype=np.int8)
    mask_data[1:2, 1:-1, 1:-1] = 1

    maps_i, maps_i_mask, maps_i_indices = trim_maps(maps_data, mask_data)

    assert maps_i.flags["F_CONTIGUOUS"]
    assert len(maps_i_indices) == maps_i.shape[-1]
    assert maps_i.shape == ((*maps_data.shape[:3], 4))

    maps_i_correct = maps_data[..., :4].copy()
    maps_i_correct[np.logical_not(mask_data), :] = 0
    assert_almost_equal(maps_i_correct, maps_i)

    mask_data[1, 1, 1] = 0  # for test to succeed
    assert_equal(mask_data, maps_i_mask)

    mask_data[1, 1, 1] = 1  # reset, just in case.
    assert_equal(np.asarray(list(range(4))), maps_i_indices)


@pytest.mark.parametrize("weight", [3.0, -3.0])
def test_trim_maps_support_covers_negative_weights(shape_3d_default, weight):
    """A kept map must contribute its voxels to the support, whatever its sign.

    Which maps are kept is decided with abs(), so a negative-weighted map
    survives the trim. The support must agree, otherwise the map is declared
    present while covering no voxel at all.
    """
    maps_data = np.zeros((*shape_3d_default, 2), dtype=np.float32)
    maps_data[0, 0, 0, 0] = weight
    maps_data[1, 1, 1, 1] = 5.0

    mask_data = np.ones(shape_3d_default, dtype=np.int8)

    _, maps_i_mask, maps_i_indices = trim_maps(maps_data, mask_data)

    assert len(maps_i_indices) == 2
    assert maps_i_mask.sum() == 2
