import warnings

import numpy as np
import pytest

from nilearn._utils.helpers import is_matplotlib_installed
from nilearn.experimental.surface import (
    SurfaceImage,
    SurfaceLabelsMasker,
    SurfaceMasker,
)


# test with only one surface image and with 2 surface images (surface time
# series)
@pytest.mark.parametrize("shape", [(1,), (2,)])
def test_mask_img_fit_shape_mismatch(
    flip_img, make_surface_mask, make_surface_img, shape, assert_img_equal
):
    img = make_surface_img(shape)
    mask = make_surface_mask()
    masker = SurfaceMasker(mask)
    with pytest.raises(ValueError, match="number of vertices"):
        masker.fit(flip_img(img))
    masker.fit(img)
    assert_img_equal(mask, masker.mask_img_)


def test_mask_img_fit_keys_mismatch(make_surface_mask, drop_img_part):
    mask = make_surface_mask()
    masker = SurfaceMasker(mask)
    with pytest.raises(ValueError, match="key"):
        masker.fit(drop_img_part(mask))


def test_none_mask_img(make_surface_mask):
    masker = SurfaceMasker(None)
    with pytest.raises(ValueError, match="provide either"):
        masker.fit(None)
    mask = make_surface_mask()
    # no mask_img but fit argument is ok
    masker.fit(mask)
    # no fit argument but a mask_img is ok
    SurfaceMasker(mask).fit(None)


def test_unfitted_masker(make_surface_mask):
    masker = SurfaceMasker(make_surface_mask)
    with pytest.raises(ValueError, match="fitted"):
        masker.transform(make_surface_mask)


def test_mask_img_transform_shape_mismatch(
    flip_img, make_surface_img, make_surface_mask
):
    img = make_surface_img()
    mask = make_surface_mask()
    masker = SurfaceMasker(mask).fit()
    with pytest.raises(ValueError, match="number of vertices"):
        masker.transform(flip_img(img))
    # non-flipped is ok
    masker.transform(img)


def test_mask_img_transform_keys_mismatch(
    make_surface_mask, make_surface_img, drop_img_part
):
    img = make_surface_img()
    mask = make_surface_mask()
    masker = SurfaceMasker(mask).fit()
    with pytest.raises(ValueError, match="key"):
        masker.transform(drop_img_part(img))
    # full img is ok
    masker.transform(img)


@pytest.mark.parametrize("shape", [(), (1,), (3,), (3, 2)])
def test_transform_inverse_transform(
    make_surface_img, shape, assert_img_equal
):
    img = make_surface_img(shape)
    masker = SurfaceMasker().fit(img)
    masked_img = masker.transform(img)
    assert np.array_equal(
        masked_img.ravel()[:9], [1, 2, 3, 4, 10, 20, 30, 40, 50]
    )
    assert masked_img.shape == (*shape, img.shape[-1])
    unmasked_img = masker.inverse_transform(masked_img)
    assert_img_equal(img, unmasked_img)


@pytest.mark.xfail(
    reason="Will implement a different surface image inside the test."
)
@pytest.mark.parametrize("shape", [(), (1,), (3,), (3, 2)])
@pytest.mark.parametrize("n_zeros_in_mask", [4, 6])
def test_transform_inverse_transform_with_mask(
    make_surface_img,
    assert_img_equal,
    make_surface_mask,
    shape,
    n_zeros_in_mask,
):
    img = make_surface_img(shape)
    mask = make_surface_mask(n_zeros_in_mask)
    masker = SurfaceMasker(mask).fit(img)
    masked_img = masker.transform(img)
    assert masked_img.shape == (*shape, img.shape[-1] - n_zeros_in_mask)
    assert np.array_equal(masked_img.ravel()[:7], [2, 3, 4, 20, 30, 40, 50])
    unmasked_img = masker.inverse_transform(masked_img)
    expected_data = {k: v.copy() for (k, v) in img.data.parts.items()}
    for v in expected_data.values():
        v[..., 0] = 0.0
    expected_img = SurfaceImage(img.mesh, expected_data)
    assert_img_equal(expected_img, unmasked_img)


@pytest.mark.xfail(reason="Fixture for label image not implemented.")
@pytest.mark.skipif(
    is_matplotlib_installed(),
    reason="Test requires matplotlib not to be installed.",
)
def test_masker_reporting_mpl_warning(make_surface_mask, mini_label_img):
    """Raise warning after exception if matplotlib is not installed."""
    with warnings.catch_warnings(record=True) as warning_list:
        SurfaceMasker(make_surface_mask).fit().generate_report()

    assert len(warning_list) == 1
    assert issubclass(warning_list[0].category, ImportWarning)

    with warnings.catch_warnings(record=True) as warning_list:
        SurfaceLabelsMasker(mini_label_img).fit().generate_report()

    assert len(warning_list) == 1
    assert issubclass(warning_list[0].category, ImportWarning)
