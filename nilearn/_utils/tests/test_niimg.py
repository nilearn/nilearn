from pathlib import Path

import joblib
import numpy as np
import pytest
from nibabel import Nifti1Header, Nifti1Image, load

from nilearn._utils import load_niimg, niimg, testing
from nilearn.image import get_data, new_img_like


@pytest.fixture
def img1(affine_eye):
    data = np.ones((2, 2, 2, 2))
    return Nifti1Image(data, affine=affine_eye)


def test_new_img_like_side_effect(img1):
    hash1 = joblib.hash(img1)
    new_img_like(
        img1, np.ones((2, 2, 2, 2)), img1.affine.copy(), copy_header=True
    )
    hash2 = joblib.hash(img1)
    assert hash1 == hash2


@pytest.mark.parametrize("no_int64_nifti", ["allow for this test"])
def test_get_target_dtype(affine_eye):
    img = Nifti1Image(np.ones((2, 2, 2), dtype=np.float64), affine=affine_eye)
    assert get_data(img).dtype.kind == "f"
    dtype_kind_float = niimg._get_target_dtype(
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
    dtype_kind_int = niimg._get_target_dtype(
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
                loaded.get_data_dtype() == niimg.img_data_dtype(loaded)
            )
            assert np.array(loaded.dataobj).dtype == niimg.img_data_dtype(
                loaded
            )
    # Verify that the distinction is worth making
    assert any(dtype_matches)
    assert not all(dtype_matches)


def test_load_niimg(img1, tmp_path):
    filename = testing.write_imgs_to_path(
        img1, file_path=tmp_path, create_files=True
    )
    filename = Path(filename)
    load_niimg(filename)
