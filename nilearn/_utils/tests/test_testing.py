import numpy as np
import pytest
from nibabel import Nifti1Image

from nilearn._utils.testing import (
    assert_memory_less_than,
    with_memory_profiler,
    write_imgs_to_path,
)


def create_object(size):
    """Just create and return an object containing `size` bytes."""
    mem_use = b"a" * size
    return mem_use


@with_memory_profiler
def test_memory_usage():
    # Valid measures (larger objects)
    for mem in (500, 200):
        assert_memory_less_than(mem, 0.1, create_object, mem * 1024**2)

    # Ensure an exception is raised with too small objects as
    # memory_profiler can return non trustable memory measure in this case.
    with pytest.raises(
        ValueError, match="Memory profiler measured an untrustable memory"
    ):
        assert_memory_less_than(50, 0.1, create_object, 25 * 1024**2)

    # Ensure ValueError is raised if memory used is above expected memory
    # limit.
    with pytest.raises(ValueError, match="Memory consumption measured"):
        assert_memory_less_than(100, 0.1, create_object, 200 * 1024**2)


def test_int64_niftis(affine_eye, tmp_path):
    data = np.ones((3, 3, 3), dtype=bool)
    for dtype in "uint8", "int32", "float32":
        img = Nifti1Image(data.astype(dtype), affine_eye)
        img.to_filename(tmp_path.joinpath("img.nii.gz"))
    for dtype in "int64", "uint64":
        with pytest.raises(AssertionError):
            Nifti1Image(data.astype(dtype), affine_eye)


@pytest.mark.parametrize("create_files", [True, False])
@pytest.mark.parametrize("use_wildcards", [True, False])
def test_write_tmp_imgs_default(
    monkeypatch, tmp_path, img_3d_mni, create_files, use_wildcards
):
    """Write imgs to default location."""
    monkeypatch.chdir(tmp_path)

    write_imgs_to_path(
        img_3d_mni,
        create_files=create_files,
        use_wildcards=use_wildcards,
    )


@pytest.mark.parametrize("create_files", [True, False])
@pytest.mark.parametrize("use_wildcards", [True, False])
def test_write_tmp_imgs_set_path(
    tmp_path, img_3d_mni, create_files, use_wildcards
):
    """Write imgs to a specified location."""
    write_imgs_to_path(
        img_3d_mni,
        file_path=tmp_path,
        create_files=create_files,
        use_wildcards=use_wildcards,
    )
