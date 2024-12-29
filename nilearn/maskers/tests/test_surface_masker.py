import warnings

import numpy as np
import pytest

from nilearn._utils.class_inspect import check_estimator
from nilearn._utils.helpers import is_matplotlib_installed
from nilearn.maskers import SurfaceMasker
from nilearn.surface import SurfaceImage
from nilearn.surface._testing import (
    assert_polydata_equal,
    assert_surface_image_equal,
)

extra_valid_checks = [
    "check_do_not_raise_errors_in_init_or_set_params",
    "check_no_attributes_set_in_init",
    "check_parameters_default_constructible",
    "check_transformer_n_iter",
    "check_transformers_unfitted",
    "check_estimators_unfitted",
]


@pytest.mark.parametrize(
    "estimator, check, name",
    check_estimator(
        estimator=[SurfaceMasker()], extra_valid_checks=extra_valid_checks
    ),
)
def test_check_estimator(estimator, check, name):  # noqa: ARG001
    """Check compliance with sklearn estimators."""
    check(estimator)


@pytest.mark.xfail(reason="invalid checks should fail")
@pytest.mark.parametrize(
    "estimator, check, name",
    check_estimator(
        estimator=[SurfaceMasker()],
        valid=False,
        extra_valid_checks=extra_valid_checks,
    ),
)
def test_check_estimator_invalid(estimator, check, name):  # noqa: ARG001
    """Check compliance with sklearn estimators."""
    check(estimator)


def test_fit_list_surf_images(surf_img_2d):
    """Test fit on list of surface images.

    resulting mask should have a single 'timepoint'.
    """
    masker = SurfaceMasker()
    masker.fit([surf_img_2d(3), surf_img_2d(5)])
    assert masker.mask_img_.shape == (surf_img_2d(1).shape[0], 1)


def test_fit_list_surf_images_with_mask(surf_mask_1d, surf_img_2d):
    """Test fit on list of surface images when masker has a mask."""
    masker = SurfaceMasker(mask_img=surf_mask_1d)
    masker.fit([surf_img_2d(3), surf_img_2d(5)])
    assert masker.mask_img_.shape == (surf_img_2d(1).shape[0],)


# test with only one surface image and with 2 surface images (surface time
# series)
@pytest.mark.parametrize("shape", [1, 2])
def test_mask_img_fit_shape_mismatch(
    flip_surf_img, surf_mask_1d, surf_img_2d, shape
):
    masker = SurfaceMasker(surf_mask_1d)
    with pytest.raises(ValueError, match="number of vertices"):
        masker.fit(flip_surf_img(surf_img_2d(shape)))
    masker.fit(surf_img_2d(shape))
    assert_polydata_equal(surf_mask_1d.data, masker.mask_img_.data)


def test_mask_img_fit_keys_mismatch(surf_mask_1d, drop_surf_img_part):
    masker = SurfaceMasker(surf_mask_1d)
    with pytest.raises(ValueError, match="key"):
        masker.fit(drop_surf_img_part(surf_mask_1d))


def test_none_mask_img(surf_mask_1d):
    masker = SurfaceMasker(None)
    with pytest.raises(ValueError, match="provide either"):
        masker.fit(None)
    # no mask_img but fit argument is ok
    masker.fit(surf_mask_1d)
    # no fit argument but a mask_img is ok
    SurfaceMasker(surf_mask_1d).fit(None)


@pytest.mark.parametrize("surf_mask_dim", [1, 2])
def test_transform_list_surf_images(
    surf_mask_dim,
    surf_mask_1d,
    surf_mask_2d,
    surf_img_1d,
    surf_img_2d,
):
    """Test transform on list of surface images."""
    surf_mask = surf_mask_1d if surf_mask_dim == 1 else surf_mask_2d()
    masker = SurfaceMasker(surf_mask).fit()
    signals = masker.transform([surf_img_1d, surf_img_1d, surf_img_1d])
    assert signals.shape == (3, masker.output_dimension_)
    signals = masker.transform([surf_img_2d(5), surf_img_2d(4)])
    assert signals.shape == (9, masker.output_dimension_)


@pytest.mark.parametrize("surf_mask_dim", [1, 2])
def test_inverse_transform_list_surf_images(
    surf_mask_dim, surf_mask_1d, surf_mask_2d, surf_img_2d
):
    """Test inverse_transform on list of surface images."""
    surf_mask = surf_mask_1d if surf_mask_dim == 1 else surf_mask_2d()
    masker = SurfaceMasker(surf_mask).fit()
    signals = masker.transform([surf_img_2d(3), surf_img_2d(4)])
    img = masker.inverse_transform(signals)
    assert img.shape == (surf_mask.mesh.n_vertices, 7)


def test_unfitted_masker(surf_mask_1d):
    masker = SurfaceMasker(surf_mask_1d)
    with pytest.raises(ValueError, match="fitted"):
        masker.transform(surf_mask_1d)


def test_check_is_fitted(surf_mask_1d):
    masker = SurfaceMasker(surf_mask_1d)
    assert not masker.__sklearn_is_fitted__()


def test_mask_img_transform_shape_mismatch(
    flip_surf_img, surf_img_1d, surf_mask_1d
):
    masker = SurfaceMasker(surf_mask_1d).fit()
    with pytest.raises(ValueError, match="number of vertices"):
        masker.transform(flip_surf_img(surf_img_1d))
    # non-flipped is ok
    masker.transform(surf_img_1d)


def test_mask_img_transform_clean(surf_img_2d, surf_mask_1d):
    """Smoke test for clean args."""
    masker = SurfaceMasker(
        surf_mask_1d,
        t_r=2.0,
        high_pass=1 / 128,
        clean_args={"filter": "cosine"},
    ).fit()
    masker.transform(surf_img_2d(50))


def test_mask_img_generate_report(surf_img_1d, surf_mask_1d):
    """Smoke test generate report."""
    masker = SurfaceMasker(surf_mask_1d, reports=True).fit()

    assert masker._reporting_data is not None
    assert masker._reporting_data["images"] is None

    masker.transform(surf_img_1d)

    assert isinstance(masker._reporting_data["images"], SurfaceImage)

    masker.generate_report()


def test_mask_img_generate_no_report(surf_img_2d, surf_mask_1d):
    """Smoke test generate report."""
    masker = SurfaceMasker(surf_mask_1d, reports=False).fit()

    assert masker._reporting_data is None

    img = surf_img_2d(5)
    masker.transform(img)

    masker.generate_report()


def test_warning_smoothing(surf_img_1d, surf_mask_1d):
    """Smooth during transform not implemented."""
    masker = SurfaceMasker(surf_mask_1d, smoothing_fwhm=1)
    masker = masker.fit()
    with pytest.warns(UserWarning, match="not yet supported"):
        masker.transform(surf_img_1d)


def test_mask_img_transform_keys_mismatch(
    surf_mask_1d, surf_img_1d, drop_surf_img_part
):
    masker = SurfaceMasker(surf_mask_1d).fit()
    with pytest.raises(ValueError, match="key"):
        masker.transform(drop_surf_img_part(surf_img_1d))
    # full img is ok
    masker.transform(surf_img_1d)


def test_error_inverse_transform_shape(surf_img_1d, surf_mask_1d, rng):
    masker = SurfaceMasker(surf_mask_1d).fit()
    signals = masker.transform(surf_img_1d)
    signals_wrong_shape = rng.random(
        size=(signals.shape[0] + 1, signals.shape[1] + 1)
    )
    with pytest.raises(
        ValueError, match="Input to 'inverse_transform' has wrong shape"
    ):
        masker.inverse_transform(signals_wrong_shape)


@pytest.mark.parametrize("n_timepoints", [3])
def test_transform_inverse_transform_no_mask(surf_mesh, n_timepoints):
    # make a sample image with data on the first timepoint/sample 1-4 on
    # left part and 10-50 on right part
    mesh = surf_mesh()
    img_data = {}
    for i, (key, val) in enumerate(mesh.parts.items()):
        data_shape = (val.n_vertices, n_timepoints)
        data_part = (
            np.arange(np.prod(data_shape)).reshape(data_shape[::-1]) + 1.0
        ) * 10**i
        img_data[key] = data_part.T

    img = SurfaceImage(mesh, img_data)
    masker = SurfaceMasker().fit(img)
    signals = masker.transform(img)

    # make sure none of the data has been removed
    assert signals.shape == (n_timepoints, img.shape[0])
    assert np.array_equal(signals[0], [1, 2, 3, 4, 10, 20, 30, 40, 50])
    unmasked_img = masker.inverse_transform(signals)
    assert_polydata_equal(img.data, unmasked_img.data)


@pytest.mark.parametrize("n_timepoints", [1, 3])
def test_transform_inverse_transform_with_mask(surf_mesh, n_timepoints):
    # make a sample image with data on the first timepoint/sample 1-4 on
    # left part and 10-50 on right part
    mesh = surf_mesh()
    img_data = {}
    for i, (key, val) in enumerate(mesh.parts.items()):
        data_shape = (val.n_vertices, n_timepoints)
        data_part = (
            np.arange(np.prod(data_shape)).reshape(data_shape[::-1]) + 1.0
        ) * 10**i
        img_data[key] = data_part.T
    img = SurfaceImage(mesh, img_data)

    # make a mask that removes first vertex of each part
    # total 2 removed
    mask_data = {
        "left": np.asarray([False, True, True, True]),
        "right": np.asarray([False, True, True, True, True]),
    }
    mask = SurfaceImage(mesh, mask_data)

    masker = SurfaceMasker(mask).fit(img)
    signals = masker.transform(img)

    # check mask shape is as expected
    assert signals.shape == (n_timepoints, masker.output_dimension_)

    # check the data for first seven vertices is as expected
    assert np.array_equal(signals.ravel()[:7], [2, 3, 4, 20, 30, 40, 50])

    # check whether inverse transform does not change the img
    unmasked_img = masker.inverse_transform(signals)
    # recreate data that we expect after unmasking
    expected_data = {k: v.copy() for (k, v) in img.data.parts.items()}
    for v in expected_data.values():
        v[0] = 0.0
    expected_img = SurfaceImage(img.mesh, expected_data)
    assert_surface_image_equal(unmasked_img, expected_img)


@pytest.mark.skipif(
    is_matplotlib_installed(),
    reason="Test requires matplotlib not to be installed.",
)
def test_masker_reporting_mpl_warning(surf_mask_1d):
    """Raise warning after exception if matplotlib is not installed."""
    with warnings.catch_warnings(record=True) as warning_list:
        SurfaceMasker(surf_mask_1d, cmap="gray").fit().generate_report()

    assert len(warning_list) == 1
    assert issubclass(warning_list[0].category, ImportWarning)
