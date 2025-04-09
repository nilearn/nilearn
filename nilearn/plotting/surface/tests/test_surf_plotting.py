# Tests for functions in surf_plotting.py

# ruff: noqa: ARG001

import tempfile

import numpy as np
import pandas as pd
import pytest

from nilearn._utils.exceptions import MeshDimensionError
from nilearn._utils.helpers import is_plotly_installed
from nilearn.datasets import fetch_surf_fsaverage
from nilearn.plotting import (
    plot_img_on_surf,
    plot_surf,
    plot_surf_contours,
    plot_surf_roi,
    plot_surf_stat_map,
)
from nilearn.plotting.displays import PlotlySurfaceFigure, SurfaceFigure


@pytest.mark.parametrize(
    "fn",
    [
        plot_surf,
        plot_surf_stat_map,
        plot_surf_contours,
        plot_surf_roi,
    ],
)
def test_check_surface_plotting_inputs_error_mesh_and_data_none(fn):
    """Fail if no mesh or data is passed."""
    with pytest.raises(TypeError, match="cannot both be None"):
        fn(None, None)


def test_check_surface_plotting_inputs_errors():
    """Fail if mesh is none and data is not a SurfaceImage."""
    with pytest.raises(TypeError, match="must be a SurfaceImage instance"):
        plot_surf(surf_map=1, surf_mesh=None)
    with pytest.raises(TypeError, match="must be a SurfaceImage instance"):
        plot_surf_stat_map(stat_map=1, surf_mesh=None)
    with pytest.raises(TypeError, match="must be a SurfaceImage instance"):
        plot_surf_contours(roi_map=1, surf_mesh=None)
    with pytest.raises(TypeError, match="must be a SurfaceImage instance"):
        plot_surf_roi(roi_map=1, surf_mesh=None)


def test_plot_surf_contours_warning_hemi(in_memory_mesh):
    """Test warning that hemi will be ignored."""
    parcellation = np.zeros((in_memory_mesh.n_vertices,))
    parcellation[in_memory_mesh.faces[3]] = 1
    with pytest.warns(UserWarning, match="This value will be ignored"):
        plot_surf_contours(in_memory_mesh, parcellation, hemi="left")


def test_surface_figure():
    s = SurfaceFigure()
    assert s.output_file is None
    assert s.figure is None
    with pytest.raises(NotImplementedError):
        s.show()
    with pytest.raises(ValueError, match="You must provide an output file"):
        s._check_output_file()
    s._check_output_file("foo.png")
    assert s.output_file == "foo.png"
    s = SurfaceFigure(output_file="bar.png")
    assert s.output_file == "bar.png"


def test_surface_figure_add_contours_raises_not_implemented():
    """Test that calling add_contours method of SurfaceFigure raises a \
    NotImplementedError.
    """
    figure = SurfaceFigure()
    with pytest.raises(NotImplementedError):
        figure.add_contours()


@pytest.mark.skipif(
    is_plotly_installed(),
    reason=("This test only runs if Plotly is not installed."),
)
def test_plotly_surface_figure_import_error():
    """Test that an ImportError is raised when instantiating \
       a PlotlySurfaceFigure without having Plotly installed.
    """
    with pytest.raises(ImportError, match="Plotly is required"):
        PlotlySurfaceFigure()


@pytest.fixture
def surf_roi_data(rng, in_memory_mesh):
    roi_map = np.zeros((in_memory_mesh.n_vertices, 1))
    roi_idx = rng.integers(0, in_memory_mesh.n_vertices, size=10)
    roi_map[roi_idx] = 1
    return roi_map


def test_plot_surf_engine_error(in_memory_mesh):
    with pytest.raises(ValueError, match="Unknown plotting engine"):
        plot_surf(in_memory_mesh, engine="foo")


@pytest.mark.skipif(
    is_plotly_installed(),
    reason="This test is run only if plotly is not installed.",
)
def test_plot_surf_engine_error_plotly_not_installed(in_memory_mesh):
    with pytest.raises(ImportError, match="Using engine"):
        plot_surf(in_memory_mesh, engine="plotly")


@pytest.mark.parametrize("engine", ["matplotlib", "plotly"])
def test_plot_surf(
    matplotlib_pyplot, engine, tmp_path, in_memory_mesh, bg_map
):
    if not is_plotly_installed() and engine == "plotly":
        pytest.skip("Plotly is not installed; required for this test.")

    # to avoid extra warnings
    alpha = None
    cbar_vmin = None
    cbar_vmax = None
    if engine == "matplotlib":
        alpha = 0.5
        cbar_vmin = 0
        cbar_vmax = 150

    # Plot mesh only
    plot_surf(in_memory_mesh, engine=engine)

    # Plot mesh with background
    plot_surf(in_memory_mesh, bg_map=bg_map, engine=engine)
    plot_surf(in_memory_mesh, bg_map=bg_map, darkness=0.5, engine=engine)
    plot_surf(
        in_memory_mesh,
        bg_map=bg_map,
        alpha=alpha,
        output_file=tmp_path / "tmp.png",
        engine=engine,
    )

    # Plot with colorbar
    plot_surf(in_memory_mesh, bg_map=bg_map, colorbar=True, engine=engine)
    plot_surf(
        in_memory_mesh,
        bg_map=bg_map,
        colorbar=True,
        cbar_vmin=cbar_vmin,
        cbar_vmax=cbar_vmax,
        cbar_tick_format="%i",
        engine=engine,
    )


@pytest.mark.parametrize("view", ["anterior", "posterior"])
@pytest.mark.parametrize("hemi", ["left", "right", "both"])
def test_plot_surf_hemi_views_plotly(
    matplotlib_pyplot, plotly, in_memory_mesh, hemi, view, bg_map
):
    """Check plotting view and hemispheres."""
    plot_surf(
        in_memory_mesh, bg_map=bg_map, hemi=hemi, view=view, engine="plotly"
    )


@pytest.mark.parametrize("hemi", ["left", "right", "both"])
def test_plot_surf_swap_hemi(
    matplotlib_pyplot, surf_img_1d, hemi, flip_surf_img
):
    """Check error is raised if background image is incompatible."""
    with pytest.raises(
        MeshDimensionError,
        match="Number of vertices do not match for between meshes.",
    ):
        plot_surf(
            surf_map=surf_img_1d,
            bg_map=flip_surf_img(surf_img_1d),
            hemi=hemi,
            surf_mesh=None,
        )


@pytest.mark.parametrize("engine", ["matplotlib", "plotly"])
def test_plot_surf_error(engine, rng, in_memory_mesh):
    if not is_plotly_installed() and engine == "plotly":
        pytest.skip("Plotly is not installed; required for this test.")
    # Wrong inputs for view or hemi
    with pytest.raises(ValueError, match="Invalid view definition"):
        plot_surf(in_memory_mesh, view="middle", engine=engine)
    with pytest.raises(ValueError, match="Invalid hemispheres definition"):
        plot_surf(in_memory_mesh, hemi="lft", engine=engine)

    # Wrong size of background image
    with pytest.raises(
        ValueError, match="bg_map does not have the same number of vertices"
    ):
        plot_surf(
            in_memory_mesh,
            bg_map=rng.standard_normal(size=in_memory_mesh.n_vertices - 1),
            engine=engine,
        )

    # Wrong size of surface data
    with pytest.raises(
        ValueError, match="surf_map does not have the same number of vertices"
    ):
        plot_surf(
            in_memory_mesh,
            surf_map=rng.standard_normal(size=in_memory_mesh.n_vertices + 1),
            engine=engine,
        )

    with pytest.raises(
        ValueError, match="'surf_map' can only have one dimension"
    ):
        plot_surf(
            in_memory_mesh,
            surf_map=rng.standard_normal(size=(in_memory_mesh.n_vertices, 2)),
            engine=engine,
        )


# @pytest.mark.parametrize()
@pytest.mark.parametrize("engine", ["matplotlib", "plotly"])
def test_plot_surf_stat_map(matplotlib_pyplot, engine, in_memory_mesh, bg_map):
    if not is_plotly_installed() and engine == "plotly":
        pytest.skip("Plotly is not installed; required for this test.")

    alpha = 1 if engine == "matplotlib" else None
    # Plot mesh with stat map
    plot_surf_stat_map(in_memory_mesh, stat_map=bg_map, engine=engine)
    plot_surf_stat_map(
        in_memory_mesh, stat_map=bg_map, alpha=alpha, engine=engine
    )


@pytest.mark.parametrize("engine", ["matplotlib", "plotly"])
def test_plot_surf_stat_map_with_background(
    matplotlib_pyplot, engine, in_memory_mesh, bg_map
):
    """Plot mesh with background and stat map."""
    if not is_plotly_installed() and engine == "plotly":
        pytest.skip("Plotly is not installed; required for this test.")

    plot_surf_stat_map(
        in_memory_mesh, stat_map=bg_map, bg_map=bg_map, engine=engine
    )
    plot_surf_stat_map(
        in_memory_mesh,
        stat_map=bg_map,
        bg_map=bg_map,
        bg_on_data=True,
        darkness=0.5,
        engine=engine,
    )


@pytest.mark.parametrize("engine", ["matplotlib", "plotly"])
def test_plot_surf_stat_map_with_title(
    matplotlib_pyplot, engine, in_memory_mesh, bg_map
):
    """Check title is added."""
    if not is_plotly_installed() and engine == "plotly":
        pytest.skip("Plotly is not installed; required for this test.")

    display = plot_surf_stat_map(
        in_memory_mesh, stat_map=bg_map, title="Stat map title"
    )
    assert display.axes[0].title._text == "Stat map title"


@pytest.mark.parametrize("engine", ["matplotlib", "plotly"])
def test_plot_surf_stat_map_with_threshold(
    matplotlib_pyplot, engine, in_memory_mesh, bg_map
):
    """Check title is added."""
    if not is_plotly_installed() and engine == "plotly":
        pytest.skip("Plotly is not installed; required for this test.")

    plot_surf_stat_map(
        in_memory_mesh,
        stat_map=bg_map,
        threshold=0.3,
        engine=engine,
    )


@pytest.mark.parametrize("engine", ["matplotlib", "plotly"])
def test_plot_surf_stat_map_vmax(
    matplotlib_pyplot, engine, in_memory_mesh, bg_map
):
    """Change vmax."""
    if not is_plotly_installed() and engine == "plotly":
        pytest.skip("Plotly is not installed; required for this test.")

    plot_surf_stat_map(in_memory_mesh, stat_map=bg_map, vmax=5, engine=engine)


@pytest.mark.parametrize("engine", ["matplotlib", "plotly"])
def test_plot_surf_stat_map_colormap(
    matplotlib_pyplot, engine, in_memory_mesh, bg_map
):
    """Change colormap."""
    if not is_plotly_installed() and engine == "plotly":
        pytest.skip("Plotly is not installed; required for this test.")

    plot_surf_stat_map(
        in_memory_mesh, stat_map=bg_map, cmap="cubehelix", engine=engine
    )


def test_plot_surf_stat_map_error(in_memory_mesh, bg_map):
    # Wrong size of stat map data
    with pytest.raises(
        ValueError, match="surf_map does not have the same number of vertices"
    ):
        plot_surf_stat_map(
            in_memory_mesh, stat_map=np.hstack((bg_map, bg_map))
        )

    with pytest.raises(
        ValueError, match="'surf_map' can only have one dimension"
    ):
        plot_surf_stat_map(
            in_memory_mesh, stat_map=np.vstack((bg_map, bg_map)).T
        )


@pytest.mark.parametrize("engine", ["matplotlib", "plotly"])
@pytest.mark.parametrize("colorbar", [True, False])
def test_plot_surf_roi(matplotlib_pyplot, engine, surface_image_roi, colorbar):
    if not is_plotly_installed() and engine == "plotly":
        pytest.skip("Plotly is not installed; required for this test.")
    plot_surf_roi(
        surface_image_roi.mesh,
        roi_map=surface_image_roi,
        colorbar=colorbar,
        engine=engine,
    )


def test_plot_surf_roi_cmap_as_lookup_table(surface_image_roi):
    """Test colormap passed as BIDS lookup table."""
    lut = pd.DataFrame(
        {"index": [0, 1], "name": ["foo", "bar"], "color": ["#000", "#fff"]}
    )
    plot_surf_roi(surface_image_roi.mesh, roi_map=surface_image_roi, cmap=lut)

    lut = pd.DataFrame({"index": [0, 1], "name": ["foo", "bar"]})
    with pytest.warns(
        UserWarning, match="No 'color' column found in the look-up table."
    ):
        plot_surf_roi(
            surface_image_roi.mesh, roi_map=surface_image_roi, cmap=lut
        )


@pytest.mark.parametrize("engine", ["matplotlib", "plotly"])
def test_plot_surf_roi_error(engine, rng, in_memory_mesh, surf_roi_data):
    if not is_plotly_installed() and engine == "plotly":
        pytest.skip("Plotly is not installed; required for this test.")
    # too many axes
    with pytest.raises(
        ValueError, match="roi_map can only have one dimension but has"
    ):
        plot_surf_roi(
            in_memory_mesh,
            roi_map=np.array([surf_roi_data, surf_roi_data]),
            engine=engine,
        )

    # wrong number of vertices
    roi_idx = rng.integers(0, in_memory_mesh.n_vertices, size=5)
    with pytest.raises(
        ValueError, match="roi_map does not have the same number of vertices"
    ):
        plot_surf_roi(in_memory_mesh, roi_map=roi_idx, engine=engine)

    # negative value in roi map
    surf_roi_data[0] = -1
    with pytest.warns(
        DeprecationWarning,
        match="Negative values in roi_map will no longer be allowed",
    ):
        plot_surf_roi(in_memory_mesh, roi_map=surf_roi_data, engine=engine)

    # float value in roi map
    surf_roi_data[0] = 1.2
    with pytest.warns(
        DeprecationWarning,
        match="Non-integer values in roi_map will no longer be allowed",
    ):
        plot_surf_roi(in_memory_mesh, roi_map=surf_roi_data, engine=engine)


@pytest.mark.parametrize(
    "hemispheres, views",
    [
        (["right"], ["lateral"]),
        (["left", "right"], ["anterior"]),
        (["right"], ["medial", "lateral"]),
        (["left", "right"], ["dorsal", "ventral"]),
        # Check that manually set view angles work.
        (["left", "right"], [(210.0, 90.0), (15.0, -45.0)]),
    ],
)
def test_plot_img_on_surf_hemispheres_and_orientations(
    matplotlib_pyplot, img_3d_mni, hemispheres, views
):
    # Check that all combinations of 1D or 2D hemis and orientations work.
    plot_img_on_surf(img_3d_mni, hemispheres=hemispheres, views=views)


def test_plot_img_on_surf_colorbar(matplotlib_pyplot, img_3d_mni):
    plot_img_on_surf(
        img_3d_mni,
        hemispheres=["right"],
        views=["lateral"],
        colorbar=True,
        vmin=-5,
        vmax=5,
        threshold=3,
    )
    plot_img_on_surf(
        img_3d_mni,
        hemispheres=["right"],
        views=["lateral"],
        colorbar=True,
        vmin=-1,
        vmax=5,
        symmetric_cbar=False,
        threshold=3,
    )
    plot_img_on_surf(
        img_3d_mni, hemispheres=["right"], views=["lateral"], colorbar=False
    )
    plot_img_on_surf(
        img_3d_mni,
        hemispheres=["right"],
        views=["lateral"],
        colorbar=False,
        cmap="roy_big_bl",
    )
    plot_img_on_surf(
        img_3d_mni,
        hemispheres=["right"],
        views=["lateral"],
        colorbar=True,
        cmap="roy_big_bl",
        vmax=2,
    )


def test_plot_img_on_surf_inflate(matplotlib_pyplot, img_3d_mni):
    plot_img_on_surf(
        img_3d_mni, hemispheres=["right"], views=["lateral"], inflate=True
    )


@pytest.mark.parametrize("surf_mesh", ["fsaverage5", fetch_surf_fsaverage()])
def test_plot_img_on_surf_surf_mesh(matplotlib_pyplot, img_3d_mni, surf_mesh):
    plot_img_on_surf(
        img_3d_mni,
        hemispheres=["right", "left"],
        views=["anterior"],
        surf_mesh=surf_mesh,
    )


def test_plot_img_on_surf_surf_mesh_low_alpha(matplotlib_pyplot, img_3d_mni):
    """Check that low alpha value do not cause floating point error.

    regression test for: https://github.com/nilearn/nilearn/issues/4900
    """
    plot_img_on_surf(img_3d_mni, threshold=3, alpha=0.1)


def test_plot_img_on_surf_with_invalid_orientation(img_3d_mni):
    kwargs = {"hemisphere": ["right"], "inflate": True}
    with pytest.raises(ValueError):
        plot_img_on_surf(img_3d_mni, views=["latral"], **kwargs)
    with pytest.raises(ValueError):
        plot_img_on_surf(img_3d_mni, views=["dorsal", "post"], **kwargs)
    with pytest.raises(TypeError):
        plot_img_on_surf(img_3d_mni, views=0, **kwargs)
    with pytest.raises(ValueError):
        plot_img_on_surf(img_3d_mni, views=["medial", {"a": "a"}], **kwargs)


def test_plot_img_on_surf_with_invalid_hemisphere(img_3d_mni):
    with pytest.raises(ValueError):
        plot_img_on_surf(
            img_3d_mni, views=["lateral"], inflate=True, hemispheres=["lft]"]
        )
    with pytest.raises(ValueError):
        plot_img_on_surf(
            img_3d_mni, views=["medial"], inflate=True, hemispheres=["lef"]
        )
    with pytest.raises(ValueError):
        plot_img_on_surf(
            img_3d_mni,
            views=["anterior", "posterior"],
            inflate=True,
            hemispheres=["left", "right", "middle"],
        )


def test_plot_img_on_surf_with_figure_kwarg(img_3d_mni):
    with pytest.raises(ValueError):
        plot_img_on_surf(
            img_3d_mni,
            views=["anterior"],
            hemispheres=["right"],
            figure=True,
        )


def test_plot_img_on_surf_with_axes_kwarg(img_3d_mni):
    with pytest.raises(ValueError):
        plot_img_on_surf(
            img_3d_mni,
            views=["anterior"],
            hemispheres=["right"],
            inflat=True,
            axes="something",
        )


def test_plot_img_on_surf_with_engine_kwarg(img_3d_mni):
    with pytest.raises(ValueError):
        plot_img_on_surf(
            img_3d_mni,
            views=["anterior"],
            hemispheres=["right"],
            inflat=True,
            engine="something",
        )


def test_plot_img_on_surf_title(matplotlib_pyplot, img_3d_mni):
    title = "Title"
    fig, _ = plot_img_on_surf(
        img_3d_mni, hemispheres=["right"], views=["lateral"]
    )
    assert fig._suptitle is None, "Created title without title kwarg."
    fig, _ = plot_img_on_surf(
        img_3d_mni, hemispheres=["right"], views=["lateral"], title=title
    )
    assert fig._suptitle is not None, "Title not created."
    assert fig._suptitle.get_text() == title, "Title text not assigned."


def test_plot_img_on_surf_output_file(matplotlib_pyplot, tmp_path, img_3d_mni):
    fname = tmp_path / "tmp.png"
    return_value = plot_img_on_surf(
        img_3d_mni,
        hemispheres=["right"],
        views=["lateral"],
        output_file=str(fname),
    )
    assert return_value is None, "Returned figure and axes on file output."
    assert fname.is_file(), "Saved image file could not be found."


def test_plot_img_on_surf_input_as_file(matplotlib_pyplot, img_3d_mni_as_file):
    """Test nifti is supported when passed as string or path to a file."""
    plot_img_on_surf(stat_map=img_3d_mni_as_file)
    plot_img_on_surf(stat_map=str(img_3d_mni_as_file))


def test_plot_surf_contours(
    matplotlib_pyplot, in_memory_mesh, parcellation, surf_mask_1d
):
    plot_surf_contours(in_memory_mesh, parcellation)
    plot_surf_contours(in_memory_mesh, parcellation, levels=[1, 2])
    plot_surf_contours(
        in_memory_mesh, parcellation, levels=[1, 2], cmap="gist_ncar"
    )


def test_plot_surf_contour_roi_map_as_surface_image(
    matplotlib_pyplot, surf_mesh, surf_mask_1d
):
    """Check that mesh can be PolyMesh and roi_map can be a SurfaceImage."""
    plot_surf_contours(surf_mesh, roi_map=surf_mask_1d, hemi=None)


def test_plot_surf_contours_legend(
    matplotlib_pyplot, in_memory_mesh, parcellation
):
    fig = plot_surf_contours(
        in_memory_mesh,
        parcellation,
        legend=True,
    )
    assert fig.legends is not None


def test_plot_surf_contours_colors(
    matplotlib_pyplot, in_memory_mesh, parcellation
):
    plot_surf_contours(
        in_memory_mesh, parcellation, levels=[1, 2], colors=["r", "g"]
    )
    plot_surf_contours(
        in_memory_mesh,
        parcellation,
        levels=[1, 2],
        labels=["1", "2"],
        colors=["r", "g"],
    )
    plot_surf_contours(
        in_memory_mesh,
        parcellation,
        levels=[1, 2],
        colors=[[0, 0, 0, 1], [1, 1, 1, 1]],
    )


def test_plot_surf_contours_axis_title(
    matplotlib_pyplot, in_memory_mesh, parcellation
):
    fig = plot_surf(in_memory_mesh)
    plot_surf_contours(in_memory_mesh, parcellation, figure=fig)
    display = plot_surf_contours(
        in_memory_mesh,
        parcellation,
        levels=[1, 2],
        labels=["1", "2"],
        colors=["r", "g"],
        legend=True,
        title="title",
        figure=fig,
    )
    # Non-regression assertion: we switched from _suptitle to axis title
    assert display._suptitle is None
    assert display.axes[0].get_title() == "title"

    fig = plot_surf(in_memory_mesh, title="title 2")
    display = plot_surf_contours(
        in_memory_mesh,
        parcellation,
        levels=[1, 2],
        labels=["1", "2"],
        colors=["r", "g"],
        legend=True,
        figure=fig,
    )
    # Non-regression assertion: we switched from _suptitle to axis title
    assert display._suptitle is None
    assert display.axes[0].get_title() == "title 2"
    with tempfile.NamedTemporaryFile() as tmp_file:
        plot_surf_contours(
            in_memory_mesh, parcellation, output_file=tmp_file.name
        )


@pytest.mark.parametrize("avg_method", ["mean", "median"])
@pytest.mark.parametrize("symmetric_cmap", [True, False, None])
@pytest.mark.parametrize("engine", ["matplotlib", "plotly"])
def test_plot_surf_roi_default_arguments(
    plotly, engine, symmetric_cmap, avg_method, surface_image_roi
):
    """Regression test for https://github.com/nilearn/nilearn/issues/3941."""
    # To avoid extra warnings
    if engine == "plotly":
        avg_method = None

    plot_surf_roi(
        surface_image_roi.mesh,
        roi_map=surface_image_roi,
        engine=engine,
        symmetric_cmap=symmetric_cmap,
        darkness=None,  # to avoid deprecation warning
        cmap="RdYlBu_r",
        avg_method=avg_method,
    )


@pytest.mark.parametrize(
    "function",
    [plot_surf_roi, plot_surf_stat_map, plot_surf_contours, plot_surf],
)
def test_error_nifti_not_supported(
    function, img_3d_mni_as_file, in_memory_mesh
):
    """Test nifti file not supported by several surface plotting functions."""
    with pytest.raises(ValueError, match="The input type is not recognized"):
        function(in_memory_mesh, img_3d_mni_as_file)
    with pytest.raises(ValueError, match="The input type is not recognized"):
        function(in_memory_mesh, str(img_3d_mni_as_file))
