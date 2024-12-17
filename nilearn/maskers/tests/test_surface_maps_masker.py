from os.path import join
from pathlib import Path

import numpy as np
import pytest

from nilearn._utils.class_inspect import check_estimator
from nilearn.maskers import SurfaceMapsMasker
from nilearn.surface import SurfaceImage


@pytest.fixture
def surf_maps_img(surf_mesh, rng):
    """Return a sample surface map image using the sample mesh.
    Has 6 regions in total: 3 in both, 1 only in left and 2 only in right.
    Later we multiply the data with random "probability" values to make it
    more realistic.
    """
    data = {
        "left": np.asarray(
            [
                [1, 1, 0, 1, 0, 0],
                [0, 1, 1, 1, 0, 0],
                [1, 0, 1, 1, 0, 0],
                [1, 1, 1, 0, 0, 0],
            ]
        ),
        "right": np.asarray(
            [
                [1, 0, 0, 0, 1, 1],
                [1, 1, 0, 0, 1, 1],
                [0, 1, 1, 0, 1, 1],
                [1, 1, 1, 0, 0, 1],
                [0, 0, 1, 0, 0, 1],
            ]
        ),
    }
    # multiply with random "probability" values
    data = {part: data[part] * rng.random(data[part].shape) for part in data}
    return SurfaceImage(surf_mesh(), data)


# tests for scikit-learn compatibility
extra_valid_checks = [
    "check_no_attributes_set_in_init",
    "check_parameters_default_constructible",
    "check_transformer_n_iter",
    "check_transformers_unfitted",
    "check_estimator_repr",
    "check_estimator_cloneable",
    "check_do_not_raise_errors_in_init_or_set_params",
    "check_estimators_unfitted",
    "check_mixin_order",
    "check_estimator_tags_renamed",
]


@pytest.mark.parametrize(
    "estimator, check, name",
    check_estimator(
        estimator=[SurfaceMapsMasker(surf_maps_img)],
        extra_valid_checks=extra_valid_checks,
    ),
)
def test_check_estimator(estimator, check, name):  # noqa: ARG001
    """Check compliance with sklearn estimators."""
    check(estimator)


@pytest.mark.xfail(reason="invalid checks should fail")
@pytest.mark.parametrize(
    "estimator, check, name",
    check_estimator(
        estimator=[SurfaceMapsMasker(surf_maps_img)],
        valid=False,
        extra_valid_checks=extra_valid_checks,
    ),
)
def test_check_estimator_invalid(estimator, check, name):  # noqa: ARG001
    """Check compliance with sklearn estimators."""
    check(estimator)


@pytest.mark.parametrize("surf_mask_dim", [1, 2])
def test_surface_maps_masker_fit_transform_shape(
    surf_maps_img, surf_img_2d, surf_mask_1d, surf_mask_2d, surf_mask_dim
):
    """Test that the fit_transform method returns the expected shape."""
    surf_mask = surf_mask_1d if surf_mask_dim == 1 else surf_mask_2d()
    masker = SurfaceMapsMasker(surf_maps_img, surf_mask).fit()
    region_signals = masker.transform(surf_img_2d(50))
    # surf_img_2d has shape (n_vertices, n_timepoints) = (9, 50)
    # surf_maps_img has shape (n_vertices, n_regions) = (9, 6)
    # region_signals should have shape (n_timepoints, n_regions) = (50, 6)
    assert region_signals.shape == (
        surf_img_2d(50).shape[-1],
        surf_maps_img.shape[-1],
    )


def test_surface_maps_masker_fit_transform_mask_vs_no_mask(
    surf_maps_img, surf_img_2d, surf_mask_1d
):
    """Test that fit_transform returns the different results when a mask is
    used vs. when no mask is used.
    """
    masker_with_mask = SurfaceMapsMasker(surf_maps_img, surf_mask_1d).fit()
    region_signals_with_mask = masker_with_mask.transform(surf_img_2d(50))

    masker_no_mask = SurfaceMapsMasker(surf_maps_img).fit()
    region_signals_no_mask = masker_no_mask.transform(surf_img_2d(50))

    assert not (region_signals_with_mask == region_signals_no_mask).all()


def test_surface_maps_masker_fit_transform_actual_output(surf_mesh, rng):
    """Test that fit_transform returns the expected output.
    Meaning that the SurfaceMapsMasker gives the solution to equation Ax = B,
    where A is the maps_img, x is the region_signals, and B is the img.
    """
    # create a maps_img with 9 vertices and 2 regions
    A = rng.random((9, 2))
    maps_data = {"left": A[:4, :], "right": A[4:, :]}
    surf_maps_img = SurfaceImage(surf_mesh(), maps_data)

    # random region signals x
    expected_region_signals = rng.random((50, 2))

    # create an img with 9 vertices and 50 timepoints as B = A @ x
    B = np.dot(A, expected_region_signals.T)
    img_data = {"left": B[:4, :], "right": B[4:, :]}
    surf_img = SurfaceImage(surf_mesh(), img_data)

    # get the region signals x using the SurfaceMapsMasker
    region_signals = SurfaceMapsMasker(surf_maps_img).fit_transform(surf_img)

    assert region_signals.shape == expected_region_signals.shape
    assert np.allclose(region_signals, expected_region_signals)


@pytest.mark.parametrize("surf_mask_dim", [1, 2])
def test_surface_maps_masker_inverse_transform_shape(
    surf_maps_img, surf_img_2d, surf_mask_1d, surf_mask_2d, surf_mask_dim
):
    """Test that inverse_transform returns an image with the same shape as the
    input.
    """
    surf_mask = surf_mask_1d if surf_mask_dim == 1 else surf_mask_2d()
    masker = SurfaceMapsMasker(surf_maps_img, surf_mask).fit()
    region_signals = masker.fit_transform(surf_img_2d(50))
    X_inverse_transformed = masker.inverse_transform(region_signals)
    assert X_inverse_transformed.shape == surf_img_2d(50).shape


def test_surface_maps_masker_inverse_transform_actual_output(surf_mesh, rng):
    """Test that inverse_transform returns the expected output."""
    # create a maps_img with 9 vertices and 2 regions
    A = rng.random((9, 2))
    maps_data = {"left": A[:4, :], "right": A[4:, :]}
    surf_maps_img = SurfaceImage(surf_mesh(), maps_data)

    # random region signals x
    expected_region_signals = rng.random((50, 2))

    # create an img with 9 vertices and 50 timepoints as B = A @ x
    B = np.dot(A, expected_region_signals.T)
    img_data = {"left": B[:4, :], "right": B[4:, :]}
    surf_img = SurfaceImage(surf_mesh(), img_data)

    # get the region signals x using the SurfaceMapsMasker
    masker = SurfaceMapsMasker(surf_maps_img).fit()
    region_signals = masker.fit_transform(surf_img)
    X_inverse_transformed = masker.inverse_transform(region_signals)

    assert np.allclose(
        X_inverse_transformed.data.parts["left"], img_data["left"]
    )
    assert np.allclose(
        X_inverse_transformed.data.parts["right"], img_data["right"]
    )


def test_surface_maps_masker_inverse_transform_wrong_region_signals_shape(
    surf_maps_img, surf_img_2d
):
    """Test that an error is raised when the region_signals shape is wrong."""
    masker = SurfaceMapsMasker(surf_maps_img).fit()
    region_signals = masker.fit_transform(surf_img_2d(50))
    wrong_region_signals = region_signals[:, :-1]

    with pytest.raises(
        ValueError,
        match="Expected 6 regions, but got 5",
    ):
        masker.inverse_transform(wrong_region_signals)


def test_surface_maps_masker_1d_maps_img(surf_img_1d):
    """Test that an error is raised when maps_img has 1D data."""
    with pytest.raises(
        ValueError,
        match="maps_img should be 2D",
    ):
        SurfaceMapsMasker(maps_img=surf_img_1d).fit()


def test_surface_maps_masker_1d_img(surf_maps_img, surf_img_1d):
    """Test that an error is raised when img has 1D data."""
    with pytest.raises(
        ValueError,
        match="img should be 2D",
    ):
        masker = SurfaceMapsMasker(maps_img=surf_maps_img).fit()
        masker.transform(surf_img_1d)


def test_surface_maps_masker_not_fitted_error(surf_maps_img):
    """Test that an error is raised when transform or inverse_transform is
    called before fit.
    """
    masker = SurfaceMapsMasker(surf_maps_img)
    with pytest.raises(
        ValueError,
        match="SurfaceMapsMasker has not been fitted",
    ):
        masker.transform(None)
    with pytest.raises(
        ValueError,
        match="SurfaceMapsMasker has not been fitted",
    ):
        masker.inverse_transform(None)


def test_surface_maps_masker_smoothing_not_supported_error(
    surf_maps_img, surf_img_2d
):
    """Test that an error is raised when smoothing_fwhm is not None."""
    masker = SurfaceMapsMasker(maps_img=surf_maps_img, smoothing_fwhm=1).fit()
    with pytest.warns(match="smoothing_fwhm is not yet supported"):
        masker.transform(surf_img_2d(50))
        assert masker.smoothing_fwhm is None


def test_surface_maps_masker_transform_clean(surf_maps_img, surf_img_2d):
    """Smoke test for clean arguments."""
    masker = SurfaceMapsMasker(
        surf_maps_img,
        t_r=2.0,
        high_pass=1 / 128,
        clean_args={"filter": "cosine"},
    ).fit()
    masker.transform(surf_img_2d(50))


def test_surface_maps_masker_labels_img_none():
    """Test that an error is raised when maps_img is None."""
    with pytest.raises(
        ValueError,
        match="provide a maps_img during initialization",
    ):
        SurfaceMapsMasker(maps_img=None).fit()


@pytest.mark.parametrize("confounds", [None, np.ones((20, 3)), "str", "Path"])
def test_surface_maps_masker_confounds_to_fit_transform(
    surf_maps_img, surf_img_2d, confounds
):
    """Test fit_transform with confounds."""
    masker = SurfaceMapsMasker(surf_maps_img)
    if isinstance(confounds, str) and confounds == "Path":
        nilearn_dir = Path(__file__).parent.parent.parent
        confounds = nilearn_dir / "tests" / "data" / "spm_confounds.txt"
    elif isinstance(confounds, str) and confounds == "str":
        # we need confound to be a string so using os.path.join
        confounds = join(  # noqa: PTH118
            Path(__file__).parent.parent.parent,
            "tests",
            "data",
            "spm_confounds.txt",
        )
    signals = masker.fit_transform(surf_img_2d(20), confounds=confounds)
    assert signals.shape == (20, masker.n_elements_)


def test_surface_maps_masker_sample_mask_to_fit_transform(
    surf_maps_img, surf_img_2d
):
    """Test transform with sample_mask."""
    masker = SurfaceMapsMasker(surf_maps_img)
    masker = masker.fit()
    signals = masker.transform(
        surf_img_2d(5),
        sample_mask=np.asarray([True, False, True, False, True]),
    )
    # we remove two samples via sample_mask so we should have 3 samples
    assert signals.shape == (3, masker.n_elements_)
