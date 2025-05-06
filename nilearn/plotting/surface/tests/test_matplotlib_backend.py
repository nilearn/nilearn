"""Test nilearn.plotting.surface._matplotlib_backend functions."""

# ruff: noqa: ARG001

import re
import tempfile

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from nilearn.datasets import fetch_surf_fsaverage
from nilearn.plotting import (
    plot_surf,
    plot_surf_contours,
    plot_surf_roi,
    plot_surf_stat_map,
)
from nilearn.plotting.surface._matplotlib_backend import (
    MATPLOTLIB_VIEWS,
    MatplotlibSurfaceBackend,
    _compute_facecolors,
    _get_bounds,
    _get_ticks,
)
from nilearn.surface import (
    load_surf_data,
    load_surf_mesh,
)

ENGINE = "matplotlib"

pytest.importorskip(
    ENGINE,
    reason="Matplotlib is not installed; required to run the tests!",
)

EXPECTED_VIEW_MATPLOTLIB = {
    "left": {
        "anterior": (0, 90),
        "posterior": (0, 270),
        "medial": (0, 0),
        "lateral": (0, 180),
        "dorsal": (90, 0),
        "ventral": (270, 0),
    },
    "right": {
        "anterior": (0, 90),
        "posterior": (0, 270),
        "medial": (0, 180),
        "lateral": (0, 0),
        "dorsal": (90, 0),
        "ventral": (270, 0),
    },
    "both": {
        "right": (0, 0),
        "left": (0, 180),
        "dorsal": (90, 0),
        "ventral": (270, 0),
        "anterior": (0, 90),
        "posterior": (0, 270),
    },
}


@pytest.fixture
def matplotlib_backend():
    return MatplotlibSurfaceBackend()


@pytest.mark.parametrize("hemi, views", MATPLOTLIB_VIEWS.items())
def test_get_view_plot_surf(matplotlib_backend, hemi, views):
    """Test if nilearn.plotting.surface._matplotlib_backend._get_view_plot_surf
    returns expected values.
    """
    for v in views:
        assert (
            matplotlib_backend._get_view_plot_surf(hemi, v)
            == EXPECTED_VIEW_MATPLOTLIB[hemi][v]
        )


@pytest.mark.parametrize("hemi,view", [("foo", "medial"), ("bar", "anterior")])
def test_get_view_plot_surf_hemisphere_errors(matplotlib_backend, hemi, view):
    """Test nilearn.plotting.surface._matplotlib_backend._get_view_plot_surf
    for invalid hemisphere values.
    """
    with pytest.raises(ValueError, match="Invalid hemispheres definition"):
        matplotlib_backend._get_view_plot_surf(hemi, view)


@pytest.mark.parametrize(
    "hemi,view",
    [
        ("left", "foo"),
        ("right", "bar"),
        ("both", "lateral"),
        ("both", "medial"),
        ("both", "foo"),
    ],
)
def test_get_view_plot_surf_view_errors(matplotlib_backend, hemi, view):
    """Test nilearn.plotting.surface._matplotlib_backend._get_view_plot_surf
    for invalid view values.
    """
    with pytest.raises(ValueError, match="Invalid view definition"):
        matplotlib_backend._get_view_plot_surf(hemi, view)


@pytest.mark.parametrize(
    "data,expected",
    [
        (np.linspace(0, 1, 100), (0, 1)),
        (np.linspace(-0.7, -0.01, 40), (-0.7, -0.01)),
    ],
)
def test_get_bounds(data, expected):
    """Test if nilearn.plotting.surface._matplotlib_backend._get_bounds
    returns expected values.
    """
    assert _get_bounds(data) == expected
    assert _get_bounds(data, vmin=0.2) == (0.2, expected[1])
    assert _get_bounds(data, vmax=0.8) == (expected[0], 0.8)
    assert _get_bounds(data, vmin=0.1, vmax=0.8) == (0.1, 0.8)


@pytest.mark.parametrize(
    "vmin,vmax,cbar_tick_format,expected",
    [
        (0, 0, "%i", [0]),
        (0, 3, "%i", [0, 1, 2, 3]),
        (0, 4, "%i", [0, 1, 2, 3, 4]),
        (1, 5, "%i", [1, 2, 3, 4, 5]),
        (0, 5, "%i", [0, 1.25, 2.5, 3.75, 5]),
        (0, 10, "%i", [0, 2.5, 5, 7.5, 10]),
        (0, 0, "%.1f", [0]),
        (0, 1, "%.1f", [0, 0.25, 0.5, 0.75, 1]),
        (1, 2, "%.1f", [1, 1.25, 1.5, 1.75, 2]),
        (1.1, 1.2, "%.1f", [1.1, 1.125, 1.15, 1.175, 1.2]),
        (0, np.nextafter(0, 1), "%.1f", [0.0e000, 5.0e-324]),
    ],
)
def test_get_ticks(vmin, vmax, cbar_tick_format, expected):
    """Test if nilearn.plotting.surface._matplotlib_backend._get_ticks
    returns expected values.
    """
    ticks = _get_ticks(vmin, vmax, cbar_tick_format, threshold=None)
    assert 1 <= len(ticks) <= 5
    assert ticks[0] == vmin and ticks[-1] == vmax
    assert (
        len(np.unique(ticks)) == len(expected)
        and (np.unique(ticks) == expected).all()
    )


def test_compute_facecolors():
    """Test if nilearn.plotting.surface._matplotlib_backend._compute_facecolors
    returns expected values.
    """
    fsaverage = fetch_surf_fsaverage()
    mesh = load_surf_mesh(fsaverage["pial_left"])
    alpha = "auto"
    # Surface map whose value in each vertex is
    # 1 if this vertex's curv > 0
    # 0 if this vertex's curv is 0
    # -1 if this vertex's curv < 0
    bg_map = np.sign(load_surf_data(fsaverage["curv_left"]))
    bg_min, bg_max = np.min(bg_map), np.max(bg_map)
    assert bg_min < 0 or bg_max > 1

    facecolors_auto_normalized = _compute_facecolors(
        bg_map,
        mesh.faces,
        len(mesh.coordinates),
        None,
        alpha,
    )

    assert len(facecolors_auto_normalized) == len(mesh.faces)

    # Manually set values of background map between 0 and 1
    bg_map_normalized = (bg_map - bg_min) / (bg_max - bg_min)
    assert np.min(bg_map_normalized) == 0 and np.max(bg_map_normalized) == 1

    facecolors_manually_normalized = _compute_facecolors(
        bg_map_normalized,
        mesh.faces,
        len(mesh.coordinates),
        None,
        alpha,
    )

    assert len(facecolors_manually_normalized) == len(mesh.faces)
    assert np.allclose(
        facecolors_manually_normalized, facecolors_auto_normalized
    )

    # Scale background map between 0.25 and 0.75
    bg_map_scaled = bg_map_normalized / 2 + 0.25
    assert np.min(bg_map_scaled) == 0.25 and np.max(bg_map_scaled) == 0.75

    facecolors_manually_rescaled = _compute_facecolors(
        bg_map_scaled,
        mesh.faces,
        len(mesh.coordinates),
        None,
        alpha,
    )

    assert len(facecolors_manually_rescaled) == len(mesh.faces)
    assert not np.allclose(
        facecolors_manually_rescaled, facecolors_auto_normalized
    )


def test_compute_facecolors_deprecation():
    """Test warning deprecation."""
    fsaverage = fetch_surf_fsaverage()
    mesh = load_surf_mesh(fsaverage["pial_left"])
    alpha = "auto"
    # Surface map whose value in each vertex is
    # 1 if this vertex's curv > 0
    # 0 if this vertex's curv is 0
    # -1 if this vertex's curv < 0
    bg_map = np.sign(load_surf_data(fsaverage["curv_left"]))
    bg_min, bg_max = np.min(bg_map), np.max(bg_map)
    assert bg_min < 0 or bg_max > 1
    with pytest.warns(
        DeprecationWarning,
        match=(
            "The `darkness` parameter will be deprecated in release 0.13. "
            "We recommend setting `darkness` to None"
        ),
    ):
        _compute_facecolors(
            bg_map,
            mesh.faces,
            len(mesh.coordinates),
            0.5,
            alpha,
        )


def test_plot_surf_with_title(matplotlib_pyplot, in_memory_mesh, bg_map):
    """Test if figure title is set correctly in
    nilearn.plotting.surface.surf_plotting.plot_surf.
    """
    display = plot_surf(
        in_memory_mesh, bg_map=bg_map, title="Test title", engine=ENGINE
    )

    assert len(display.axes) == 1
    assert display.axes[0].title._text == "Test title"


def test_plot_surf_avg_method(matplotlib_pyplot, in_memory_mesh, bg_map):
    """Test nilearn.plotting.surface.surf_plotting.plot_surf for valid
    values of avg_method.
    """
    # Plot with avg_method
    # Test all built-in methods and check
    faces = in_memory_mesh.faces

    for method in ["mean", "median", "min", "max"]:
        display = plot_surf(
            in_memory_mesh,
            surf_map=bg_map,
            avg_method=method,
            engine=ENGINE,
        )
        if method == "mean":
            agg_faces = np.mean(bg_map[faces], axis=1)
        elif method == "median":
            agg_faces = np.median(bg_map[faces], axis=1)
        elif method == "min":
            agg_faces = np.min(bg_map[faces], axis=1)
        elif method == "max":
            agg_faces = np.max(bg_map[faces], axis=1)
        vmin = np.min(agg_faces)
        vmax = np.max(agg_faces)
        agg_faces -= vmin
        agg_faces /= vmax - vmin
        cmap = matplotlib_pyplot.get_cmap(
            matplotlib_pyplot.rcParamsDefault["image.cmap"]
        )
        assert_array_equal(
            cmap(agg_faces),
            display._axstack.as_list()[0].collections[0]._facecolors,
        )

    #  Try custom avg_method
    def custom_avg_function(vertices):
        return vertices[0] * vertices[1] * vertices[2]

    plot_surf(
        in_memory_mesh,
        surf_map=bg_map,
        avg_method=custom_avg_function,
        engine=ENGINE,
    )


def test_plot_surf_avg_method_errors(in_memory_mesh, bg_map):
    """Test nilearn.plotting.surface.surf_plotting.plot_surf for invalid
    values of avg_method.
    """
    with pytest.raises(
        ValueError,
        match=(
            "Array computed with the custom "
            "function from avg_method does "
            "not have the correct shape"
        ),
    ):

        def custom_avg_function(vertices):
            return [vertices[0] * vertices[1], vertices[2]]

        plot_surf(
            in_memory_mesh,
            surf_map=bg_map,
            avg_method=custom_avg_function,
            engine=ENGINE,
        )

    with pytest.raises(
        ValueError,
        match=re.escape(
            "avg_method should be either "
            "['mean', 'median', 'max', 'min'] "
            "or a custom function"
        ),
    ):
        custom_avg_function = {}

        plot_surf(
            in_memory_mesh,
            surf_map=bg_map,
            avg_method=custom_avg_function,
            engine=ENGINE,
        )

        plot_surf(
            in_memory_mesh,
            surf_map=bg_map,
            avg_method="foo",
            engine=ENGINE,
        )

    with pytest.raises(
        ValueError,
        match=re.escape(
            "Array computed with the custom function "
            "from avg_method should be an array of "
            "numbers (int or float)"
        ),
    ):

        def custom_avg_function(vertices):
            return "string"

        plot_surf(
            in_memory_mesh,
            surf_map=bg_map,
            avg_method=custom_avg_function,
            engine=ENGINE,
        )


@pytest.mark.parametrize(
    "kwargs", [{"symmetric_cmap": True}, {"title_font_size": 18}]
)
def test_plot_surf_warnings_not_implemented_in_matplotlib(
    kwargs, in_memory_mesh, bg_map
):
    """Test if nilearn.plotting.surface.surf_plotting.plot_surf raises error
    when a parameter that is not supported by matplotlib is specified with a
    value other than None.
    """
    with pytest.warns(
        UserWarning, match="is not implemented for the matplotlib engine"
    ):
        plot_surf(
            in_memory_mesh,
            surf_map=bg_map,
            engine=ENGINE,
            **kwargs,
        )


def test_surface_plotting_axes_error(matplotlib_pyplot, surf_img_1d):
    """Test error msg for invalid axes."""
    figure, axes = matplotlib_pyplot.subplots()
    with pytest.raises(AttributeError, match="the projection must be '3d'"):
        plot_surf_stat_map(stat_map=surf_img_1d, axes=axes)


def test_plot_surf_stat_map_matplotlib_specific(
    matplotlib_pyplot, in_memory_mesh, bg_map
):
    # Plot to axes
    axes = matplotlib_pyplot.subplots(
        ncols=2, subplot_kw={"projection": "3d"}
    )[1]
    for ax in axes.flatten():
        plot_surf_stat_map(in_memory_mesh, stat_map=bg_map, axes=ax)
    axes = matplotlib_pyplot.subplots(
        ncols=2, subplot_kw={"projection": "3d"}
    )[1]
    for ax in axes.flatten():
        plot_surf_stat_map(in_memory_mesh, stat_map=bg_map, axes=ax)

    fig = plot_surf_stat_map(in_memory_mesh, stat_map=bg_map, colorbar=False)

    assert len(fig.axes) == 1

    # symmetric_cbar
    fig = plot_surf_stat_map(
        in_memory_mesh, stat_map=bg_map, symmetric_cbar=True
    )
    fig.canvas.draw()

    assert len(fig.axes) == 2

    yticklabels = fig.axes[1].get_yticklabels()
    first, last = yticklabels[0].get_text(), yticklabels[-1].get_text()

    assert float(first) == -float(last)

    # no symmetric_cbar
    fig = plot_surf_stat_map(
        in_memory_mesh, stat_map=bg_map, symmetric_cbar=False
    )
    fig.canvas.draw()

    assert len(fig.axes) == 2

    yticklabels = fig.axes[1].get_yticklabels()
    first, last = yticklabels[0].get_text(), yticklabels[-1].get_text()

    assert float(first) != -float(last)

    # Test handling of nan values in texture data
    # Add nan values in the texture
    bg_map[2] = np.nan
    # Plot the surface stat map
    fig = plot_surf_stat_map(in_memory_mesh, stat_map=bg_map)
    # Check that the resulting plot facecolors contain no transparent faces
    # (last column equals zero) even though the texture contains nan values
    tmp = fig._axstack.as_list()[0].collections[0]

    assert (
        in_memory_mesh.faces.shape[0] == ((tmp._facecolors[:, 3]) != 0).sum()
    )


def test_plot_surf_roi_matplotlib_specific(
    matplotlib_pyplot, surface_image_roi
):
    # change vmin, vmax
    img = plot_surf_roi(
        surface_image_roi.mesh,
        roi_map=surface_image_roi,
        vmin=1.2,
        vmax=8.9,
        colorbar=True,
        engine=ENGINE,
    )
    img.canvas.draw()
    cbar = img.axes[-1]
    cbar_vmin = float(cbar.get_yticklabels()[0].get_text())
    cbar_vmax = float(cbar.get_yticklabels()[-1].get_text())

    assert cbar_vmin == 1.0
    assert cbar_vmax == 8.0

    img2 = plot_surf_roi(
        surface_image_roi.mesh,
        roi_map=surface_image_roi,
        vmin=1.2,
        vmax=8.9,
        colorbar=True,
        cbar_tick_format="%.2g",
        engine=ENGINE,
    )
    img2.canvas.draw()
    cbar = img2.axes[-1]
    cbar_vmin = float(cbar.get_yticklabels()[0].get_text())
    cbar_vmax = float(cbar.get_yticklabels()[-1].get_text())

    assert cbar_vmin == 1.2
    assert cbar_vmax == 8.9


def test_plot_surf_roi_matplotlib_specific_nan_handling(
    matplotlib_pyplot,
    surface_image_parcellation,
):
    # Test nans handling
    surface_image_parcellation.data.parts["left"][::2] = np.nan
    img = plot_surf_roi(
        surface_image_parcellation.mesh,
        roi_map=surface_image_parcellation,
        engine=ENGINE,
        hemi="left",
    )
    # Check that the resulting plot facecolors contain no transparent faces
    # (last column equals zero) even though the texture contains nan values
    tmp = img._axstack.as_list()[0].collections[0]
    n_faces = surface_image_parcellation.mesh.parts["left"].faces.shape[0]

    assert n_faces == ((tmp._facecolors[:, 3]) != 0).sum()


def test_plot_surf_roi_matplotlib_specific_plot_to_axes(
    matplotlib_pyplot, surface_image_roi
):
    """Test plotting directly on some axes."""
    plot_surf_roi(
        surface_image_roi.mesh,
        roi_map=surface_image_roi,
        axes=None,
        figure=matplotlib_pyplot.gcf(),
        engine=ENGINE,
    )

    _, ax = matplotlib_pyplot.subplots(subplot_kw={"projection": "3d"})

    with tempfile.NamedTemporaryFile() as tmp_file:
        plot_surf_roi(
            surface_image_roi.mesh,
            roi_map=surface_image_roi,
            axes=ax,
            figure=None,
            output_file=tmp_file.name,
            engine=ENGINE,
        )

    with tempfile.NamedTemporaryFile() as tmp_file:
        plot_surf_roi(
            surface_image_roi.mesh,
            roi_map=surface_image_roi,
            axes=ax,
            figure=None,
            output_file=tmp_file.name,
            colorbar=True,
            engine=ENGINE,
        )


def test_plot_surf_contours_fig_axes(
    matplotlib_pyplot, in_memory_mesh, parcellation
):
    fig, axes = matplotlib_pyplot.subplots(
        1, 1, subplot_kw={"projection": "3d"}
    )
    plot_surf_contours(in_memory_mesh, parcellation, axes=axes)
    plot_surf_contours(in_memory_mesh, parcellation, figure=fig)


def test_plot_surf_contours_error(
    matplotlib_pyplot, rng, in_memory_mesh, parcellation
):
    # we need an invalid parcellation for testing
    invalid_parcellation = rng.uniform(size=(in_memory_mesh.n_vertices))
    with pytest.raises(
        ValueError, match="Vertices in parcellation do not form region."
    ):
        plot_surf_contours(in_memory_mesh, invalid_parcellation)

    _, axes = matplotlib_pyplot.subplots(1, 1)
    with pytest.raises(ValueError, match="Axes must be 3D."):
        plot_surf_contours(in_memory_mesh, parcellation, axes=axes)

    msg = "All elements of colors .* matplotlib .* RGBA"
    with pytest.raises(ValueError, match=msg):
        plot_surf_contours(
            in_memory_mesh, parcellation, levels=[1, 2], colors=[[1, 2], 3]
        )

    msg = "Levels, labels, and colors argument .* same length or None."
    with pytest.raises(ValueError, match=msg):
        plot_surf_contours(
            in_memory_mesh,
            parcellation,
            levels=[1, 2],
            colors=["r"],
            labels=["1", "2"],
        )
