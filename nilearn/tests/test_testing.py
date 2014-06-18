import itertools

import numpy as np

from nose.tools import assert_equals, assert_raises

from .._utils.testing import generate_fake_fmri


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
                rand_gen=rand_gen)
        else:
            fmri, mask, target = generate_fake_fmri(
                shape=shape, length=length, kind=kind,
                n_blocks=n_block, block_size=bsize,
                block_type=btype,
                rand_gen=rand_gen)

        assert_equals(fmri.shape[:-1], shape)
        assert_equals(fmri.shape[-1], length)

        if n_block is not None:
            assert_equals(target.size, length)

    assert_raises(ValueError, generate_fake_fmri, length=10, n_blocks=10,
                  block_size=None, rand_gen=rand_gen)
