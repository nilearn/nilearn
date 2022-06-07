import itertools

import numpy as np
import pytest
import nibabel

from nilearn._utils.testing import with_memory_profiler
from nilearn._utils.testing import assert_memory_less_than
from nilearn._utils.data_gen import generate_fake_fmri


def create_object(size):
    """Just create and return an object containing `size` bytes."""
    mem_use = b'a' * size
    return mem_use


@with_memory_profiler
def test_memory_usage():
    # Valid measures (larger objects)
    for mem in (500, 200):
        assert_memory_less_than(mem, 0.1, create_object, mem * 1024 ** 2)

    # Ensure an exception is raised with too small objects as
    # memory_profiler can return non trustable memory measure in this case.
    with pytest.raises(
            ValueError,
            match="Memory profiler measured an untrustable memory"):
        assert_memory_less_than(50, 0.1,
                                create_object, 25 * 1024 ** 2)

    # Ensure ValueError is raised if memory used is above expected memory
    # limit.
    with pytest.raises(ValueError, match="Memory consumption measured"):
        assert_memory_less_than(100, 0.1, create_object, 200 * 1024 ** 2)


def test_generate_fake_fmri():
    shapes = [(6, 6, 7), (10, 11, 12)]
    lengths = [16, 20]
    kinds = ['noise', 'step']
    n_blocks = [None, 1, 4]
    block_size = [None, 4]
    block_type = ['classification', 'regression']

    rand_gen = np.random.RandomState(3)

    for shape, length, kind, n_block, bsize, btype in itertools.product(
            shapes, lengths, kinds, n_blocks, block_size, block_type):

        if n_block is None:
            fmri, mask = generate_fake_fmri(
                shape=shape, length=length, kind=kind,
                n_blocks=n_block, block_size=bsize,
                block_type=btype,
                random_state=rand_gen)
        else:
            fmri, mask, target = generate_fake_fmri(
                shape=shape, length=length, kind=kind,
                n_blocks=n_block, block_size=bsize,
                block_type=btype,
                random_state=rand_gen)

        assert fmri.shape[:-1] == shape
        assert fmri.shape[-1] == length

        if n_block is not None:
            assert target.size == length

    pytest.raises(ValueError, generate_fake_fmri, length=10, n_blocks=10,
                  block_size=None, random_state=rand_gen)


def test_int64_niftis(tmp_path):
    data = np.ones((3, 3, 3), dtype=bool)
    affine = np.eye(4)
    for dtype in "uint8", "int32", "float32":
        img = nibabel.Nifti1Image(data.astype(dtype), affine)
        img.to_filename(tmp_path.joinpath("img.nii.gz"))
    for dtype in "int64", "uint64":
        with pytest.raises(AssertionError):
            nibabel.Nifti1Image(data.astype(dtype), affine)
