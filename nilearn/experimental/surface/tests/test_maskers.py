import numpy as np
import pytest

from nilearn.experimental.surface import SurfaceImage, SurfaceMasker


def test_mask_img_fit_shape_mismatch(
    flip_img, mini_mask, make_mini_img, assert_img_equal
):
    img = make_mini_img()
    masker = SurfaceMasker(mini_mask)
    with pytest.raises(ValueError, match="number of vertices"):
        masker.fit(flip_img(img))
    # fitting with the number of vertices is ok
    masker.fit(img)
    # fitting with the same number of vertices and extra dimensions (surface
    # timeseries) is ok
    masker.fit(make_mini_img((2,)))
    assert_img_equal(mini_mask, masker.mask_img_)


def test_mask_img_fit_keys_mismatch(mini_mask, drop_img_part):
    masker = SurfaceMasker(mini_mask)
    with pytest.raises(ValueError, match="key"):
        masker.fit(drop_img_part(mini_mask))


def test_none_mask_img(mini_mask):
    masker = SurfaceMasker(None)
    with pytest.raises(ValueError, match="provide either"):
        masker.fit(None)
    # no mask_img but fit argument is ok
    masker.fit(mini_mask)
    # no fit argument but a mask_img is ok
    SurfaceMasker(mini_mask).fit(None)


def test_unfitted_masker(mini_mask):
    masker = SurfaceMasker(mini_mask)
    with pytest.raises(ValueError, match="fitted"):
        masker.transform(mini_mask)


def test_mask_img_transform_shape_mismatch(flip_img, mini_img, mini_mask):
    masker = SurfaceMasker(mini_mask).fit()
    with pytest.raises(ValueError, match="number of vertices"):
        masker.transform(flip_img(mini_img))
    # non-flipped is ok
    masker.transform(mini_img)


def test_mask_img_transform_keys_mismatch(mini_mask, mini_img, drop_img_part):
    masker = SurfaceMasker(mini_mask).fit()
    with pytest.raises(ValueError, match="key"):
        masker.transform(drop_img_part(mini_img))
    # full img is ok
    masker.transform(mini_img)


@pytest.mark.parametrize("shape", [(), (1,), (3,), (3, 2)])
def test_transform_inverse_transform(shape, make_mini_img, assert_img_equal):
    img = make_mini_img(shape)
    masker = SurfaceMasker().fit(img)
    masked_img = masker.transform(img)
    assert np.array_equal(
        masked_img.ravel()[:9], [1, 2, 3, 4, 10, 20, 30, 40, 50]
    )
    assert masked_img.shape == shape + (img.shape[-1],)
    unmasked_img = masker.inverse_transform(masked_img)
    assert_img_equal(img, unmasked_img)


@pytest.mark.parametrize("shape", [(), (1,), (3,), (3, 2)])
def test_transform_inverse_transform_with_mask(
    shape, make_mini_img, assert_img_equal, mini_mask
):
    img = make_mini_img(shape)
    masker = SurfaceMasker(mini_mask).fit(img)
    masked_img = masker.transform(img)
    assert masked_img.shape == shape + (img.shape[-1] - 2,)
    assert np.array_equal(masked_img.ravel()[:7], [2, 3, 4, 20, 30, 40, 50.0])
    unmasked_img = masker.inverse_transform(masked_img)
    expected_data = {k: v.copy() for (k, v) in img.data.items()}
    for v in expected_data.values():
        v[..., 0] = 0.0
    expected_img = SurfaceImage(img.mesh, expected_data)
    assert_img_equal(expected_img, unmasked_img)
