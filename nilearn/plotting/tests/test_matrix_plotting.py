import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from nibabel.tmpdirs import InTemporaryDirectory

from nilearn.glm.first_level.design_matrix import (
    make_first_level_design_matrix,
)
from nilearn.plotting.matrix_plotting import (
    plot_contrast_matrix,
    plot_design_matrix,
    plot_event,
    plot_matrix,
)

##############################################################################
# Some smoke testing for graphics-related code


@pytest.mark.parametrize(
    "fig,axes", [("foo", "bar"), (1, 2), plt.subplots(1, 1, figsize=(7, 5))]
)
def test_sanitize_figure_and_axes_error(fig, axes):
    from ..matrix_plotting import _sanitize_figure_and_axes

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
    from ..matrix_plotting import _sanitize_figure_and_axes

    fig2, axes2, own_fig = _sanitize_figure_and_axes(fig, axes)
    assert isinstance(fig2, plt.Figure)
    assert isinstance(axes2, plt.Axes)
    assert own_fig == expected


def test_sanitize_labels():
    from ..matrix_plotting import _sanitize_labels

    labs = ["foo", "bar"]
    with pytest.raises(
        ValueError, match="Length of labels unequal to length of matrix."
    ):
        _sanitize_labels((6, 6), labs)
    for lab in [labs, np.array(labs)]:
        assert _sanitize_labels((2, 2), lab) == labs


VALID_TRI_VALUES = ("full", "lower", "diag")


@pytest.mark.parametrize("tri", VALID_TRI_VALUES)
def test_sanitize_tri(tri):
    from ..matrix_plotting import _sanitize_tri

    _sanitize_tri(tri)


@pytest.mark.parametrize("tri", [None, "foo", 2])
def test_sanitize_tri_error(tri):
    from ..matrix_plotting import _sanitize_tri

    with pytest.raises(
        ValueError,
        match=(
            "Parameter tri needs to be "
            f"one of: {', '.join(VALID_TRI_VALUES)}"
        ),
    ):
        _sanitize_tri(tri)


VALID_REORDER_VALUES = (True, False, "single", "complete", "average")


@pytest.mark.parametrize("reorder", VALID_REORDER_VALUES)
def test_sanitize_reorder(reorder):
    from ..matrix_plotting import _sanitize_reorder

    if reorder is not True:
        assert _sanitize_reorder(reorder) == reorder
    else:
        assert _sanitize_reorder(reorder) == "average"


@pytest.mark.parametrize("reorder", [None, "foo", 2])
def test_sanitize_reorder_error(reorder):
    from ..matrix_plotting import _sanitize_reorder

    with pytest.raises(
        ValueError, match=("Parameter reorder needs to be one of")
    ):
        _sanitize_reorder(reorder)


@pytest.fixture
def mat():
    return np.zeros((10, 10))


@pytest.fixture
def labels():
    return [str(i) for i in range(10)]


@pytest.mark.parametrize(
    "matrix,lab,reorder",
    [
        (np.zeros((10, 10)), [0, 1, 2], False),
        (np.zeros((10, 10)), None, True),
        (np.zeros((10, 10)), [str(i) for i in range(10)], " "),
    ],
)
def test_matrix_plotting_errors(matrix, lab, reorder):
    with pytest.raises(ValueError):
        plot_matrix(matrix, labels=lab, reorder=reorder)
        plt.close()


@pytest.mark.parametrize("tri", VALID_TRI_VALUES)
def test_matrix_plotting_with_labels_and_different_tri(mat, labels, tri):
    ax = plot_matrix(mat, labels=labels, tri=tri)
    assert isinstance(ax, mpl.image.AxesImage)
    ax.axes.set_title("Title")
    assert ax._axes.get_title() == "Title"
    for axis in [ax._axes.xaxis, ax._axes.yaxis]:
        assert len(axis.majorTicks) == len(labels)
        for tick, label in zip(axis.majorTicks, labels):
            assert tick.label1.get_text() == label
    plt.close()


@pytest.mark.parametrize(
    "lab", [[], np.array([str(i) for i in range(10)]), None]
)
def test_matrix_plotting_labels(mat, lab):
    plot_matrix(mat, labels=lab)
    plt.close()


@pytest.mark.parametrize("title", ["foo", "foo bar", " ", None])
def test_matrix_plotting_set_title(mat, labels, title):
    ax = plot_matrix(mat, labels=labels, title=title)
    nb_txt = 0 if title is None else 1
    assert len(ax._axes.texts) == nb_txt
    if title is not None:
        assert ax._axes.texts[0].get_text() == title
    plt.close()


@pytest.mark.parametrize("tri", VALID_TRI_VALUES)
def test_matrix_plotting_grid(mat, labels, tri):
    plot_matrix(mat, labels=labels, grid=True, tri=tri)


def test_matrix_plotting_reorder(mat, labels):
    from itertools import permutations

    # test if reordering with default linkage works
    idx = [2, 3, 5]
    # make symmetric matrix of similarities so we can get a block
    for perm in permutations(idx, 2):
        mat[perm] = 1
    ax = plot_matrix(mat, labels=labels, reorder=True)
    assert len(labels) == len(ax.axes.get_xticklabels())
    reordered_labels = [
        int(lbl.get_text()) for lbl in ax.axes.get_xticklabels()
    ]
    # block order does not matter
    assert (
        reordered_labels[:3] == idx or reordered_labels[-3:] == idx
    ), "Clustering does not find block structure."
    plt.close()
    # test if reordering with specific linkage works
    ax = plot_matrix(mat, labels=labels, reorder="complete")
    plt.close()


def test_show_design_matrix():
    # test that the show code indeed (formally) runs
    frame_times = np.linspace(0, 127 * 1.0, 128)
    dmtx = make_first_level_design_matrix(
        frame_times, drift_model="polynomial", drift_order=3
    )
    ax = plot_design_matrix(dmtx)
    assert ax is not None
    with InTemporaryDirectory():
        ax = plot_design_matrix(dmtx, output_file="dmtx.png")
        assert os.path.exists("dmtx.png")
        assert ax is None
        plot_design_matrix(dmtx, output_file="dmtx.pdf")
        assert os.path.exists("dmtx.pdf")


def test_show_event_plot():
    # test that the show code indeed (formally) runs
    onset = np.linspace(0, 19.0, 20)
    duration = np.full(20, 0.5)
    trial_idx = np.arange(20)
    # This makes 11 events in order to test cmap error
    trial_idx[11:] -= 10
    condition_ids = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k"]

    trial_type = np.array([condition_ids[i] for i in trial_idx])

    model_event = pd.DataFrame(
        {"onset": onset, "duration": duration, "trial_type": trial_type}
    )
    # Test Dataframe
    fig = plot_event(model_event)
    assert fig is not None

    # Test List
    fig = plot_event([model_event, model_event])
    assert fig is not None

    # Test error
    with pytest.raises(ValueError):
        fig = plot_event(model_event, cmap="tab10")

    # Test save
    with InTemporaryDirectory():
        fig = plot_event(model_event, output_file="event.png")
        assert os.path.exists("event.png")
        assert fig is None
        plot_event(model_event, output_file="event.pdf")
        assert os.path.exists("event.pdf")


def test_show_contrast_matrix():
    # test that the show code indeed (formally) runs
    frame_times = np.linspace(0, 127 * 1.0, 128)
    dmtx = make_first_level_design_matrix(
        frame_times, drift_model="polynomial", drift_order=3
    )
    contrast = np.ones(4)
    ax = plot_contrast_matrix(contrast, dmtx)
    assert ax is not None
    with InTemporaryDirectory():
        ax = plot_contrast_matrix(contrast, dmtx, output_file="contrast.png")
        assert os.path.exists("contrast.png")
        assert ax is None
        plot_contrast_matrix(contrast, dmtx, output_file="contrast.pdf")
        assert os.path.exists("contrast.pdf")
