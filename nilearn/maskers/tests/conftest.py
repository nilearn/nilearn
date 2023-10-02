import pytest

from nilearn._utils.data_gen import (
    generate_fake_fmri,
    generate_labeled_regions,
    generate_maps,
)
from nilearn.conftest import _affine_eye, _shape_3d_default


@pytest.fixture()
def n_regions():
    return 9


@pytest.fixture()
def length():
    return 3


@pytest.fixture()
def shape2():
    return (8, 9, 10)


@pytest.fixture()
def fmri_img(length):
    img, _ = generate_fake_fmri(
        shape=_shape_3d_default(), affine=_affine_eye(), length=length
    )
    return img


@pytest.fixture()
def labels_img(n_regions):
    img = generate_labeled_regions(
        shape=_shape_3d_default(), affine=_affine_eye(), n_regions=n_regions
    )
    return img


@pytest.fixture()
def maps_img(n_regions):
    img, _ = generate_maps(
        shape=_shape_3d_default(), n_regions=n_regions, affine=_affine_eye()
    )
    return img


@pytest.fixture()
def mask2(shape2, length):
    _, mask_img = generate_fake_fmri(
        shape=shape2, affine=_affine_eye(), length=length
    )
    return mask_img
