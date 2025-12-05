import os
import warnings
from pathlib import Path
from tempfile import mkstemp

import joblib
import numpy as np
import pytest
from nibabel import Nifti1Header, Nifti1Image, load

from nilearn._utils.niimg import (
    _get_target_dtype,
    img_data_dtype,
    load_niimg,
    repr_niimgs,
)
from nilearn._utils.testing import write_imgs_to_path
from nilearn.image import get_data, new_img_like


@pytest.fixture
def img1(affine_eye):
    data = np.ones((2, 2, 2, 2))
    return Nifti1Image(data, affine=affine_eye)


def test_new_img_like_side_effect(img1):
    hash1 = joblib.hash(img1)
    new_img_like(img1, np.ones((2, 2, 2, 2)), img1.affine.copy())
    hash2 = joblib.hash(img1)
    assert hash1 == hash2


@pytest.mark.parametrize("no_int64_nifti", ["allow for this test"])
def test_get_target_dtype(affine_eye):
    img = Nifti1Image(np.ones((2, 2, 2), dtype=np.float64), affine=affine_eye)
    assert get_data(img).dtype.kind == "f"
    dtype_kind_float = _get_target_dtype(
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
    dtype_kind_int = _get_target_dtype(
        get_data(img2).dtype, target_dtype="auto"
    )
    assert dtype_kind_int == np.int32


@pytest.mark.parametrize("no_int64_nifti", ["allow for this test"])
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
        "[this-is-a-very-lon..., this-is-another-ve..., this-is-again-anot...]"
    )

    assert (
        repr_niimgs(list_of_size_3, shorten=True)
        == shortened_rep_list_of_size_3
    )

    # Lists longer than 3
    # Small names - Explicit shortening
    long_list_small_names = ["test", "retest", "reretest", "rereretest"]
    shortened_rep_long_list_small_names = "[test,\n         ...\n rereretest]"

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
        "[this-is-a-very-lon...,\n         ...\n this-is-again-anot...]"
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
    assert repr_niimgs(["test", "retest"]) == "[test, retest]"
    assert repr_niimgs(["test", "retest"], shorten=False) == "[test, retest]"

    # Force display, all 3 names are displayed
    list_of_size_3 = [
        "this-is-a-very-long-name-for-a-nifti-file.nii",
        "this-is-another-very-long-name-for-a-nifti-file.nii",
        "this-is-again-another-very-long-name-for-a-nifti-file.nii",
    ]
    long_rep_list_of_size_3 = (
        "[this-is-a-very-long-name-for-a-nifti-file.nii,"
        " this-is-another-very-long-name-for-a-nifti-file.nii,"
        " this-is-again-another-very-long-name-for-a-nifti-file.nii]"
    )
    assert (
        repr_niimgs(list_of_size_3, shorten=False) == long_rep_list_of_size_3
    )

    long_list_small_names = ["test", "retest", "reretest", "rereretest"]
    long_rep_long_list_small_names = (
        "[test,\n retest,\n reretest,\n rereretest]"
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
        long_rep_list_of_size_3[:-1].replace(",", ",\n")
        + ",\n "
        + "this-is-again-another-super-very-long-name-for-a-nifti-file.nii]"
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
        f"[...{Path('/path/to/file.nii')!s},\n"
        f"         ...\n"
        f" a-very-long-file-n...]"
    )

    assert repr_niimgs(list_of_paths, shorten=True) == shortened_list_of_paths
    long_list_of_paths = ",\n ".join([str(_) for _ in list_of_paths])
    long_list_of_paths = f"[{long_list_of_paths}]"
    assert repr_niimgs(list_of_paths, shorten=False) == long_list_of_paths


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
