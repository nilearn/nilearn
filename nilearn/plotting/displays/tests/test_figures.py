# ruff: noqa: ARG001
from unittest import mock

import pytest
from matplotlib.figure import Figure

from nilearn._utils.helpers import is_kaleido_installed, is_plotly_installed
from nilearn.plotting.displays import PlotlySurfaceFigure, SurfaceFigure

try:
    import IPython.display  # noqa:F401
except ImportError:
    IPYTHON_INSTALLED = False
else:
    IPYTHON_INSTALLED = True


def test_surface_figure():
    """Tests SurfaceFigure class."""
    s = SurfaceFigure()
    assert s.output_file is None
    assert s.figure is None
    assert s.hemi == "left"
    s._check_output_file("foo.png")
    assert s.output_file == "foo.png"
    s = SurfaceFigure(output_file="bar.png")
    assert s.output_file == "bar.png"


def test_surface_figure_errors():
    """Test SurfaceFigure class for errors."""
    figure = SurfaceFigure()
    with pytest.raises(NotImplementedError):
        figure.add_contours()
    with pytest.raises(NotImplementedError):
        figure.show()
    with pytest.raises(ValueError, match="You must provide an output file"):
        figure._check_output_file()


@pytest.mark.skipif(
    is_plotly_installed(),
    reason="This test only runs if Plotly is not installed.",
)
def test_plotly_surface_figure_import_error():
    """Test that an ImportError is raised when instantiating \
       a PlotlySurfaceFigure without having Plotly installed.
    """
    with pytest.raises(ImportError, match="Plotly is required"):
        PlotlySurfaceFigure()


@pytest.mark.skipif(
    is_kaleido_installed(),
    reason="This test only runs if Plotly is installed, but not kaleido.",
)
def test_plotly_surface_figure_savefig_error(plotly):
    """Test that an ImportError is raised when saving \
       a PlotlySurfaceFigure without having kaleido installed.
    """
    with pytest.raises(ImportError, match="`kaleido` is required"):
        PlotlySurfaceFigure().savefig()


@pytest.mark.skipif(
    not is_kaleido_installed(),
    reason="Kaleido is not installed; required for this test.",
)
def test_plotly_surface_figure(plotly):
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
    ps = PlotlySurfaceFigure(plotly.Figure())
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
    ps = PlotlySurfaceFigure(plotly.Figure(), output_file=tmp_path / "foo.png")
    assert ps.output_file == tmp_path / "foo.png"
    assert ps.figure is not None
    ps.savefig()
    assert (tmp_path / "foo.png").exists()


@pytest.mark.parametrize("input_obj", ["foo", Figure(), ["foo", "bar"]])
def test_instantiation_error_plotly_surface_figure(plotly, input_obj):
    """Test if PlotlySurfaceFigure raises TypeError if an object other than
    :obj:`plotly.Figure` object is specified.
    """
    with pytest.raises(
        TypeError,
        match=("`PlotlySurfaceFigure` accepts only plotly.Figure objects."),
    ):
        PlotlySurfaceFigure(input_obj)


def test_distant_line_segments_detected_as_not_intersecting(plotly):
    """Test that distant lines are detected as not intersecting."""
    assert not PlotlySurfaceFigure._do_segs_intersect(0, 0, 1, 1, 5, 5, 6, 6)
