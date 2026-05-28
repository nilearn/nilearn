"""Test nilearn.plotting.displays._figures.SurfaceFigure."""

import pytest

from nilearn.plotting.displays import SurfaceFigure


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
