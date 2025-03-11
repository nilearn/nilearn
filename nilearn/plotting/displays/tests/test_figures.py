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


@pytest.mark.skipif(is_plotly_installed(), reason="Plotly is installed.")
def test_plotly_surface_figure_import_error():
    """Test that an ImportError is raised when instantiating \
       a PlotlySurfaceFigure without having Plotly installed.
    """
    with pytest.raises(ImportError, match="Plotly is required"):
        PlotlySurfaceFigure()


@pytest.mark.skipif(
    not is_plotly_installed() or is_kaleido_installed(),
    reason=("This test only runs if Plotly is installed, but not kaleido."),
)
def test_plotly_surface_figure_savefig_error():
    """Test that an ImportError is raised when saving \
       a PlotlySurfaceFigure without having kaleido installed.
    """
    with pytest.raises(ImportError, match="`kaleido` is required"):
        PlotlySurfaceFigure().savefig()


@pytest.mark.skipif(
    not is_plotly_installed() or not is_kaleido_installed(),
    reason=("Plotly and/or kaleido not installed; required for this test."),
)
def test_plotly_surface_figure():
    ps = PlotlySurfaceFigure()
    assert ps.output_file is None
    assert ps.figure is None
    ps.show()
    with pytest.raises(ValueError, match="You must provide an output file"):
        ps.savefig()
    ps.savefig("foo.png")


@pytest.mark.skipif(
    not is_plotly_installed() or not IPYTHON_INSTALLED,
    reason=("Plotly and/or Ipython is not installed; required for this test."),
)
@pytest.mark.parametrize("renderer", ["png", "jpeg", "svg"])
def test_plotly_show(renderer):
    import plotly.graph_objects as go

    ps = PlotlySurfaceFigure(go.Figure())
    assert ps.output_file is None
    assert ps.figure is not None
    with mock.patch("IPython.display.display") as mock_display:
        ps.show(renderer=renderer)
    assert len(mock_display.call_args.args) == 1
    key = "svg+xml" if renderer == "svg" else renderer
    assert f"image/{key}" in mock_display.call_args.args[0]


@pytest.mark.skipif(
    not is_plotly_installed() or not is_kaleido_installed(),
    reason=("Plotly and/or kaleido not installed; required for this test."),
)
def test_plotly_savefig(tmp_path):
    import plotly.graph_objects as go

    ps = PlotlySurfaceFigure(go.Figure(), output_file=tmp_path / "foo.png")
    assert ps.output_file == tmp_path / "foo.png"
    assert ps.figure is not None
    ps.savefig()
    assert (tmp_path / "foo.png").exists()


@pytest.mark.parametrize("input_obj", ["foo", Figure(), ["foo", "bar"]])
def test_instantiation_error_plotly_surface_figure(input_obj):
    with pytest.raises(
        TypeError,
        match=("`PlotlySurfaceFigure` accepts only plotly figure objects."),
    ):
        PlotlySurfaceFigure(input_obj)
