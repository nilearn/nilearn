import re

import matplotlib.pyplot as plt
import numpy as np
import pytest

from nilearn.plotting.matrix._matplotlib_backend import (
    _sanitize_figure_and_axes,
    _sanitize_labels,
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


def test_sanitize_labels():
    labs = ["foo", "bar"]
    with pytest.raises(
        ValueError,
        match=re.escape(
            "Length of labels (2) unequal to length of matrix (6)."
        ),
    ):
        _sanitize_labels((6, 6), labs)
    for lab in [labs, np.array(labs)]:
        assert _sanitize_labels((2, 2), lab) == labs
