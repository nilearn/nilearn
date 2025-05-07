"""Test nilearn.plotting.displays.PlotlySurfaceFigure."""

from unittest import mock

import numpy as np
import pytest
from matplotlib.figure import Figure

from nilearn._utils.helpers import is_kaleido_installed
from nilearn.plotting import plot_surf
from nilearn.plotting.displays import PlotlySurfaceFigure

try:
    import IPython.display  # noqa:F401
except ImportError:
    IPYTHON_INSTALLED = False
else:
    IPYTHON_INSTALLED = True

ENGINE = "plotly"

pytest.importorskip(
    ENGINE,
    reason="Plotly is not installed; required to run the tests!",
)


@pytest.mark.skipif(
    is_kaleido_installed(),
    reason="This test only runs if Plotly is installed, but not kaleido.",
)
def test_plotly_surface_figure_savefig_error():
    """Test that an ImportError is raised when saving \
       a PlotlySurfaceFigure without having kaleido installed.
    """
    with pytest.raises(ImportError, match="`kaleido` is required"):
        PlotlySurfaceFigure().savefig()


@pytest.mark.skipif(
    not is_kaleido_installed(),
    reason="Kaleido is not installed; required for this test.",
)
def test_plotly_surface_figure():
    """Test ValueError when saving a PlotlySurfaceFigure without specifying
    output file.
    """
    ps = PlotlySurfaceFigure()
    assert ps.output_file is None
    assert ps.figure is None
    ps.show()
    with pytest.raises(ValueError, match="You must provide an output file"):
        ps.savefig()
    ps.savefig("foo.png")


@pytest.mark.skipif(
    not IPYTHON_INSTALLED,
    reason="IPython is not installed; required for this test.",
)
@pytest.mark.skipif(
    not is_kaleido_installed(),
    reason="Kaleido is not installed; required for this test.",
)
@pytest.mark.parametrize("renderer", ["png", "jpeg", "svg"])
def test_plotly_show(plotly, renderer):
    """Test PlotlySurfaceFigure.show method."""
    ps = PlotlySurfaceFigure(plotly.graph_objects.Figure())
    assert ps.output_file is None
    assert ps.figure is not None
    with mock.patch("IPython.display.display") as mock_display:
        ps.show(renderer=renderer)
    assert len(mock_display.call_args.args) == 1
    key = "svg+xml" if renderer == "svg" else renderer
    assert f"image/{key}" in mock_display.call_args.args[0]


@pytest.mark.skipif(
    not is_kaleido_installed(),
    reason="Kaleido is not installed; required for this test.",
)
def test_plotly_savefig(plotly, tmp_path):
    """Test PlotlySurfaceFigure.savefig method."""
    figure = plotly.graph_objects.Figure()
    ps = PlotlySurfaceFigure(figure, output_file=tmp_path / "foo.png")
    assert ps.output_file == tmp_path / "foo.png"
    assert ps.figure is not None
    ps.savefig()
    assert (tmp_path / "foo.png").exists()


@pytest.mark.parametrize("input_obj", ["foo", Figure(), ["foo", "bar"]])
def test_instantiation_error_plotly_surface_figure(input_obj):
    """Test if PlotlySurfaceFigure raises TypeError if an object other than
    :obj:`plotly Figure` object is specified.
    """
    with pytest.raises(
        TypeError,
        match=("`PlotlySurfaceFigure` accepts only plotly Figure objects."),
    ):
        PlotlySurfaceFigure(input_obj)


def test_distant_line_segments_detected_as_not_intersecting():
    """Test that distant lines are detected as not intersecting."""
    assert not PlotlySurfaceFigure._do_segs_intersect(0, 0, 1, 1, 5, 5, 6, 6)


def test_plotly_surface_figure_warns_on_isolated_roi(in_memory_mesh):
    """Test that a warning is generated for ROIs with isolated vertices."""
    figure = plot_surf(in_memory_mesh, engine=ENGINE)
    # the method raises an error because the (randomly generated)
    # vertices don't form regions
    try:
        with pytest.raises(UserWarning, match="contains isolated vertices:"):
            figure.add_contours(levels=[0], roi_map=np.array([0, 1] * 10))
    except Exception:
        pass


@pytest.mark.parametrize("levels,labels", [([0], ["a", "b"]), ([0, 1], ["a"])])
def test_value_error_add_contours_levels_labels(
    levels, labels, in_memory_mesh
):
    """Test that add_contours raises a ValueError when called with levels and \
    labels that have incompatible lengths.
    """
    figure = plot_surf(in_memory_mesh, engine=ENGINE)
    with pytest.raises(
        ValueError,
        match=("levels and labels need to be either the same length or None."),
    ):
        figure.add_contours(
            levels=levels, labels=labels, roi_map=np.ones((10,))
        )


@pytest.mark.parametrize(
    "levels,lines",
    [([0], [{}, {}]), ([0, 1], [{}, {}, {}])],
)
def test_value_error_add_contours_levels_lines(levels, lines, in_memory_mesh):
    """Test that add_contours raises a ValueError when called with levels and \
    lines that have incompatible lengths.
    """
    figure = plot_surf(in_memory_mesh, engine=ENGINE)
    with pytest.raises(
        ValueError,
        match=("levels and lines need to be either the same length or None."),
    ):
        figure.add_contours(levels=levels, lines=lines, roi_map=np.ones((10,)))


def test_add_contours(surface_image_roi):
    """Test that add_contours updates data in PlotlySurfaceFigure."""
    figure = plot_surf(surface_image_roi.mesh, engine=ENGINE)
    figure.add_contours(surface_image_roi)
    assert len(figure.figure.to_dict().get("data")) == 4

    figure.add_contours(surface_image_roi, levels=[1])
    assert len(figure.figure.to_dict().get("data")) == 5


@pytest.mark.parametrize("hemi", ["left", "right", "both"])
def test_add_contours_hemi(surface_image_roi, hemi):
    """Test that add_contours works with all hemi inputs."""
    if hemi == "both":
        n_vertices = surface_image_roi.mesh.n_vertices
    else:
        n_vertices = surface_image_roi.data.parts[hemi].shape[0]
    figure = plot_surf(
        surface_image_roi.mesh,
        engine=ENGINE,
        hemi=hemi,
    )
    figure.add_contours(surface_image_roi)
    assert figure._coords.shape[0] == n_vertices


def test_add_contours_plotly_surface_image(surface_image_roi):
    """Test that add_contours works with SurfaceImage."""
    figure = plot_surf(surf_map=surface_image_roi, hemi="left", engine=ENGINE)
    figure.add_contours(roi_map=surface_image_roi)


def test_add_contours_has_name(surface_image_roi):
    """Test that contours added to a PlotlySurfaceFigure can be named."""
    figure = plot_surf(surface_image_roi.mesh, engine=ENGINE)
    figure.add_contours(surface_image_roi, levels=[1], labels=["x"])
    assert figure.figure.to_dict().get("data")[2].get("name") == "x"


def test_add_contours_lines_duplicated(surface_image_roi):
    """Test that the specifications of length 1 line provided to \
     add_contours are duplicated to all requested contours.
    """
    figure = plot_surf(surface_image_roi.mesh, engine=ENGINE)
    figure.add_contours(surface_image_roi, lines=[{"width": 10}])
    newlines = figure.figure.to_dict().get("data")[2:]
    assert all(x.get("line").__contains__("width") for x in newlines)


@pytest.mark.parametrize(
    "key,value",
    [
        ("color", "yellow"),
        ("width", 10),
    ],
)
def test_add_contours_line_properties(key, value, surface_image_roi):
    """Test that the specifications of a line provided to add_contours are \
    stored in the PlotlySurfaceFigure data.
    """
    figure = plot_surf(surface_image_roi.mesh, engine=ENGINE)
    figure.add_contours(surface_image_roi, levels=[1], lines=[{key: value}])
    newline = figure.figure.to_dict().get("data")[2].get("line")
    assert newline.get(key) == value
