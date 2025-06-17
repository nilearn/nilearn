import matplotlib.pyplot as plt
import pytest

from nilearn.plotting.matrix._matplotlib_backend import (
    _sanitize_figure_and_axes,
)

##############################################################################
# Some smoke testing for graphics-related code


@pytest.mark.parametrize(
    "fig,axes", [("foo", "bar"), (1, 2), plt.subplots(1, 1, figsize=(7, 5))]
)
def test_sanitize_figure_and_axes_error(fig, axes):
    with pytest.raises(
        ValueError,
        match=("Parameters figure and axes cannot be specified together."),
    ):
        _sanitize_figure_and_axes(fig, axes)


@pytest.mark.parametrize(
    "fig,axes,expected",
    [
        ((6, 4), None, True),
        (plt.figure(figsize=(3, 2)), None, True),
        (None, None, True),
        (None, plt.subplots(1, 1)[1], False),
    ],
)
def test_sanitize_figure_and_axes(fig, axes, expected):
    fig2, axes2, own_fig = _sanitize_figure_and_axes(fig, axes)
    assert isinstance(fig2, plt.Figure)
    assert isinstance(axes2, plt.Axes)
    assert own_fig == expected
