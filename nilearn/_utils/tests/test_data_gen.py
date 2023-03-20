import numpy as np
import pytest
from nilearn._utils.data_gen import (
    generate_fake_fmri,
    generate_labeled_regions,
    generate_maps,
    generate_regions_ts,
)
from nilearn.image import get_data


@pytest.mark.parametrize("window", ["boxcar", "hamming"])
def test_generate_regions_ts_no_overlap(window):
    n_voxels = 50
    n_regions = 10

    regions = generate_regions_ts(
        n_voxels, n_regions, overlap=0, window=window
    )

    assert regions.shape == (n_regions, n_voxels)
    # check no overlap
    np.testing.assert_array_less(
        (regions > 0).sum(axis=0) - 0.1, np.ones(regions.shape[1])
    )
    # check: a region everywhere
    np.testing.assert_array_less(
        np.zeros(regions.shape[1]), (regions > 0).sum(axis=0)
    )


@pytest.mark.parametrize("window", ["boxcar", "hamming"])
def test_generate_regions_ts_with_overlap(window):
    n_voxels = 50
    n_regions = 10

    regions = generate_regions_ts(
        n_voxels, n_regions, overlap=1, window=window
    )

    assert regions.shape == (n_regions, n_voxels)
    # check overlap
    assert np.any((regions > 0).sum(axis=-1) > 1.9)
    # check: a region everywhere
    np.testing.assert_array_less(
        np.zeros(regions.shape[1]), (regions > 0).sum(axis=0)
    )


def test_generate_labeled_regions():
    """Minimal testing of generate_labeled_regions."""
    shape = (3, 4, 5)
    n_regions = 10
    regions = generate_labeled_regions(shape, n_regions)
    assert regions.shape == shape
    assert len(np.unique(get_data(regions))) == n_regions + 1


def test_generate_maps():
    # Basic testing of generate_maps()
    shape = (10, 11, 12)
    n_regions = 9
    maps_img, _ = generate_maps(shape, n_regions, border=1)
    maps = get_data(maps_img)
    assert maps.shape == shape + (n_regions,)
    # no empty map
    assert np.all(abs(maps).sum(axis=0).sum(axis=0).sum(axis=0) > 0)
    # check border
    assert np.all(maps[0, ...] == 0)
    assert np.all(maps[:, 0, ...] == 0)
    assert np.all(maps[:, :, 0, :] == 0)


@pytest.mark.parametrize("shape", [(10, 11, 12), (6, 6, 7)])
@pytest.mark.parametrize("length", [16, 20])
@pytest.mark.parametrize("kind", ["noise", "step"])
@pytest.mark.parametrize("n_block", [None, 1, 4])
@pytest.mark.parametrize("block_size", [None, 4])
@pytest.mark.parametrize("block_type", ["classification", "regression"])
def test_generate_fake_fmri(
    shape, length, kind, n_block, block_size, block_type
):
    rand_gen = np.random.RandomState(3)

    fake_fmri = generate_fake_fmri(
        shape=shape,
        length=length,
        kind=kind,
        n_blocks=n_block,
        block_size=block_size,
        block_type=block_type,
        random_state=rand_gen,
    )

    assert fake_fmri[0].shape[:-1] == shape
    assert fake_fmri[0].shape[-1] == length
    if n_block is not None:
        assert fake_fmri[2].size == length


def test_generate_fake_fmri_error():
    with pytest.raises(ValueError, match="10 is too small"):
        generate_fake_fmri(
            length=10,
            n_blocks=10,
            block_size=None,
            random_state=np.random.RandomState(3),
        )
