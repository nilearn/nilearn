import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_equal

from nilearn._utils import _constrained_layout_kwargs
from nilearn.glm.first_level.design_matrix import (
    make_first_level_design_matrix,
)
from nilearn.glm.tests._testing import block_paradigm, modulated_event_paradigm
from nilearn.plotting.matrix_plotting import (
    pad_contrast_matrix,
    plot_contrast_matrix,
    plot_design_matrix,
    plot_design_matrix_correlation,
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
    n_txt = 0 if title is None else len(title)
    assert len(ax._axes.title.get_text()) == n_txt
    if title is not None:
        assert ax._axes.title.get_text() == title
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


def test_show_design_matrix(tmp_path):
    # test that the show code indeed (formally) runs
    frame_times = np.linspace(0, 127 * 1.0, 128)
    dmtx = make_first_level_design_matrix(
        frame_times, drift_model="polynomial", drift_order=3
    )
    ax = plot_design_matrix(dmtx)
    assert ax is not None

    ax = plot_design_matrix(dmtx, output_file=tmp_path / "dmtx.png")
    assert (tmp_path / "dmtx.png").exists()
    assert ax is None
    plot_design_matrix(dmtx, output_file=tmp_path / "dmtx.pdf")
    assert (tmp_path / "dmtx.pdf").exists()


@pytest.mark.parametrize("suffix, sep", [(".csv", ","), (".tsv", "\t")])
def test_plot_design_matrix_path_str(tmp_path, suffix, sep):
    # test that the show code indeed (formally) runs
    frame_times = np.linspace(0, 127 * 1.0, 128)
    dmtx = make_first_level_design_matrix(
        frame_times, drift_model="polynomial", drift_order=3
    )
    filename = (tmp_path / "tmp").with_suffix(suffix)
    dmtx.to_csv(filename, sep=sep, index=False)

    ax = plot_design_matrix(filename)

    assert ax is not None

    ax = plot_design_matrix(str(filename))

    assert ax is not None


def test_show_event_plot(tmp_path):
    # test that the show code indeed (formally) runs
    onset = np.linspace(0, 19.0, 20)
    duration = np.full(20, 0.5)
    trial_idx = np.arange(20)
    # This makes 11 events in order to test cmap error
    trial_idx[11:] -= 10
    condition_ids = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k"]

    # add some modulation
    modulation = np.full(20, 1)
    modulation[[1, 5, 15]] = 0.5

    trial_type = np.array([condition_ids[i] for i in trial_idx])

    model_event = pd.DataFrame(
        {
            "onset": onset,
            "duration": duration,
            "trial_type": trial_type,
            "modulation": modulation,
        }
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
    fig = plot_event(model_event, output_file=tmp_path / "event.png")
    assert (tmp_path / "event.png").exists()
    assert fig is None
    plot_event(model_event, output_file=tmp_path / "event.pdf")
    assert (tmp_path / "event.pdf").exists()


@pytest.mark.parametrize("suffix, sep", [(".csv", ","), (".tsv", "\t")])
def test_plot_event_path_tsv_csv(tmp_path, suffix, sep):
    """Test plot_events directly from file."""
    model_event = block_paradigm()
    filename = (tmp_path / "tmp").with_suffix(suffix)
    model_event.to_csv(filename, sep=sep, index=False)

    plot_event(filename)
    plot_event([filename, str(filename)])


def test_show_contrast_matrix(tmp_path):
    # test that the show code indeed (formally) runs
    frame_times = np.linspace(0, 127 * 1.0, 128)
    dmtx = make_first_level_design_matrix(
        frame_times, drift_model="polynomial", drift_order=3
    )
    contrast = np.ones(4)
    ax = plot_contrast_matrix(contrast, dmtx)
    assert ax is not None

    ax = plot_contrast_matrix(
        contrast, dmtx, output_file=tmp_path / "contrast.png"
    )
    assert (tmp_path / "contrast.png").exists()
    assert ax is None

    plot_contrast_matrix(contrast, dmtx, output_file=tmp_path / "contrast.pdf")
    assert (tmp_path / "contrast.pdf").exists()


def test_show_contrast_matrix_axes():
    frame_times = np.linspace(0, 127 * 1.0, 128)
    dmtx = make_first_level_design_matrix(
        frame_times, drift_model="polynomial", drift_order=3
    )
    contrast = np.ones(4)
    fig, ax = plt.subplots(**_constrained_layout_kwargs())
    plot_contrast_matrix(contrast, dmtx, axes=ax)

    # to actually check we need get_layout_engine, but even without it the
    # above allows us to test the kwargs are at least okay
    pytest.importorskip("matplotlib", minversion="3.5.0")
    assert "constrained" in fig.get_layout_engine().__class__.__name__.lower()


def test_pad_contrast_matrix():
    """Test for contrasts padding before plotting.

    See https://github.com/nilearn/nilearn/issues/4211
    """
    frame_times = np.linspace(0, 127 * 1.0, 128)
    dmtx = make_first_level_design_matrix(
        frame_times, drift_model="polynomial", drift_order=3
    )
    contrast = np.array([[1, -1]])
    padded_contrast = pad_contrast_matrix(contrast, dmtx)
    assert_array_equal(padded_contrast, np.array([[1, -1, 0, 0]]))

    contrast = np.eye(3)
    padded_contrast = pad_contrast_matrix(contrast, dmtx)
    assert_array_equal(
        padded_contrast,
        np.array(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
            ]
        ),
    )


def test_show_event_plot_duration_0():
    plot_event(modulated_event_paradigm())


@pytest.mark.parametrize("tri", ["full", "diag"])
@pytest.mark.parametrize("cmap", ["RdBu_r", "bwr", "seismic_r"])
def test_plot_design_matrix_correlation(tri, cmap, tmp_path):
    """Smoke test for the 'happy path'."""
    frame_times = np.linspace(0, 127 * 1.0, 128)
    dmtx = make_first_level_design_matrix(
        frame_times, events=modulated_event_paradigm()
    )

    plot_design_matrix_correlation(
        dmtx, tri=tri, cmap=cmap, output_file=tmp_path / "corr_mat.png"
    )

    assert (tmp_path / "corr_mat.png").exists()


def test_plot_design_matrix_correlation_smoke_path(tmp_path):
    """Check that plot_design_matrix_correlation works with paths."""
    frame_times = np.linspace(0, 127 * 1.0, 128)
    dmtx = make_first_level_design_matrix(
        frame_times, events=modulated_event_paradigm()
    )

    dmtx.to_csv(tmp_path / "tmp.tsv", sep="\t", index=False)

    plot_design_matrix_correlation(tmp_path / "tmp.tsv")
    plot_design_matrix_correlation(str(tmp_path / "tmp.tsv"))


def test_plot_design_matrix_correlation_errors(mat):
    with pytest.raises(
        ValueError, match="Tables to load can only be TSV or CSV."
    ):
        plot_design_matrix_correlation("foo")

    with pytest.raises(ValueError, match="dataframe cannot be empty."):
        plot_design_matrix_correlation(pd.DataFrame())

    with pytest.raises(ValueError, match="cmap must be one of"):
        plot_design_matrix_correlation(pd.DataFrame(mat), cmap="foo")

    dmtx = pd.DataFrame(
        {"event_1": [0, 1], "constant": [1, 1], "drift_1": [0, 1]}
    )
    with pytest.raises(ValueError, match="tri needs to be one of"):
        plot_design_matrix_correlation(dmtx, tri="lower")

    dmtx = pd.DataFrame({"constant": [1, 1], "drift_1": [0, 1]})
    with pytest.raises(ValueError, match="Nothing left to plot after "):
        plot_design_matrix_correlation(dmtx)
