import re

import numpy as np
import pytest
from matplotlib.colors import Normalize

from nilearn.plotting._engine_utils import (
    colorscale,
    create_colorbar_for_fig,
    threshold_cmap,
    to_color_strings,
)

pytest.importorskip(
    "matplotlib",
    reason="Matplotlib is not installed; required to run the tests!",
)


def check_colors(colors):
    """Perform several checks on colors obtained from function colorscale."""
    assert len(colors) == 100
    val, cstring = zip(*colors, strict=False)
    assert np.allclose(np.linspace(0, 1, 100), val, atol=1e-3)
    assert val[0] == 0
    assert val[-1] == 1
    for cs in cstring:
        assert re.match(r"rgb\(\d+, \d+, \d+\)", cs)
    return val, cstring


def test_colorscale_no_threshold():
    """Test colorscale with no thresholding."""
    values = np.linspace(-13, -1.5, 20)
    colors = colorscale("jet", values, None)
    check_colors(colors["colors"])
    assert (colors["vmin"], colors["vmax"]) == (-13, 13)
    assert colors["cmap"].N == 256
    assert (colors["norm"].vmax, colors["norm"].vmin) == (13, -13)
    assert colors["abs_threshold"] is None


@pytest.fixture
def expected_abs_threshold(threshold):
    """Return the expected absolute threshold."""
    expected = {"0%": 1.5, "50%": 7.55, "99%": 13}
    return (
        expected.get(threshold)
        if isinstance(threshold, str)
        else abs(threshold)
    )


@pytest.mark.parametrize(
    "threshold, min_th, max_th",
    [(1, 63, 127), (3, 0, 191), (5, 0, 255), (0, 95, 95)],
)
def test_threshold_cmap(threshold, min_th, max_th):
    """Test nilearn.plotting._engine_utils.threshold_cmap function for valid
    threshold values.
    """
    norm = Normalize(-3, 5)
    cmap = "RdBu"
    thrs_cmap = threshold_cmap(cmap, norm, threshold)

    for i in range(min_th, max_th):
        assert thrs_cmap(i) == (0.5, 0.5, 0.5, 1.0)


def test_threshold_cmap_invalid():
    """Test nilearn.plotting._engine_utils.threshold_cmap function for negative
    threshold.
    """
    threshold = -1
    norm = Normalize(-3, 5)
    cmap = "RdBu"
    with pytest.raises(ValueError, match="Threshold should be a"):
        threshold_cmap(cmap, norm, threshold)


@pytest.mark.parametrize("threshold", ["0%", "50%", "99%", 0.5, 7.25])
def test_colorscale_threshold(threshold, expected_abs_threshold):
    """Test colorscale with different threshold values."""
    colors = colorscale("jet", np.linspace(-13, -1.5, 20), threshold=threshold)
    _, cstring = check_colors(colors["colors"])
    assert cstring[50] == "rgb(127, 127, 127)"
    assert (colors["vmin"], colors["vmax"]) == (-13, 13)
    assert colors["cmap"].N == 256
    assert (colors["norm"].vmax, colors["norm"].vmin) == (13, -13)
    assert np.allclose(colors["abs_threshold"], expected_abs_threshold, 2)


@pytest.mark.parametrize("vmin,vmax", [(None, 7), (-5, 7)])
def test_colorscale_symmetric_cmap(vmin, vmax):
    """Test colorscale with symmetric cmap and positive values."""
    colors = colorscale("jet", np.arange(15), vmin=vmin, vmax=vmax)
    assert (colors["vmin"], colors["vmax"]) == (-7, 7)
    assert colors["cmap"].N == 256
    assert (colors["norm"].vmax, colors["norm"].vmin) == (7, -7)


@pytest.fixture
def expected_vmin_vmax(values, vmax, vmin):
    """Return expected vmin and vmax."""
    if vmax is None:
        return (min(values), max(values))
    return (min(values), vmax) if vmin is None else (vmin, vmax)


@pytest.mark.parametrize(
    "values,vmax,vmin,threshold",
    [
        (np.arange(15), None, None, None),
        (np.arange(15), 7, None, None),
        (np.arange(15), 7, -5, None),
        (np.arange(15) + 3, 7, None, None),
        (np.arange(15) + 3, None, None, None),
        (np.arange(15) + 3, 7, 1, None),
        (np.arange(15) + 3, 10, 6, 5),
        (np.arange(15) + 3, 10, None, 5),
        (np.linspace(-15, 4), 7, None, None),
    ],
)
def test_colorscale_asymmetric_cmap(
    values, vmax, vmin, threshold, expected_vmin_vmax
):
    """Test colorscale with asymmetric cmap."""
    colors = colorscale(
        "jet",
        values,
        vmax=vmax,
        vmin=vmin,
        threshold=threshold,
        symmetric_cmap=False,
    )
    assert colors["cmap"].N == 256
    assert (int(colors["vmin"]), int(colors["vmax"])) == expected_vmin_vmax
    assert (colors["norm"].vmax, colors["norm"].vmin) == expected_vmin_vmax[
        ::-1
    ]


@pytest.mark.parametrize(
    "colors",
    [
        [[0, 0, 1], [1, 0, 0], [0.5, 0.5, 0.5]],
        [[0, 0, 1, 1], [1, 0, 0, 1], [0.5, 0.5, 0.5, 0]],
        ["#0000ff", "#ff0000", "#7f7f7f"],
        [[0, 0, 1, 1], [1, 0, 0, 1], [0.5, 0.5, 0.5, 0]],
        ["r", "green", "black", "white"],
        ["#0000ffff", "#ff0000ab", "#7f7f7f00"],
    ],
)
def test_to_color_strings(colors):
    """Tests for function to_color_strings with different color inputs."""
    if len(colors) == 3:
        expected = ["#0000ff", "#ff0000", "#7f7f7f"]
    else:
        expected = ["#ff0000", "#008000", "#000000", "#ffffff"]
    assert to_color_strings(colors) == expected


@pytest.mark.parametrize(
    "threshold, cbar_vmin, cbar_vmax, vmin, vmax, expected_ticks",
    [
        (0, -10, 10, -5, 5, [-10, -5, 0, 5, 10]),
        (0.1, None, None, -10, 10, [-10, -5, -0.1, 0.1, 5, 10]),
        (0.9, -10, 10, -5, 5, [-10, -5, -0.9, 0.9, 5, 10]),
        (1.3, None, 10, -10, 10, [-10, -5, -1.3, 1.3, 5, 10]),
        (3, -10, None, -10, 10, [-10, -5, -3, 0, 3, 5, 10]),
    ],
)
def test_create_colorbar_for_fig(
    matplotlib_pyplot,
    threshold,
    cbar_vmin,
    cbar_vmax,
    vmin,
    vmax,
    expected_ticks,
):
    """Test nilearn.plotting._engine_utils.create_colorbar_for_fig function for
    valid values.
    """
    fig, ax = matplotlib_pyplot.subplots()
    cmap = matplotlib_pyplot.get_cmap("Greys")
    norm = Normalize(vmin, vmax)

    colorbar = create_colorbar_for_fig(
        fig, ax, cmap, norm, threshold, cbar_vmin, cbar_vmax
    )

    assert colorbar is not None
    assert [
        float(tick.get_text()) for tick in colorbar.ax.get_yticklabels()
    ] == expected_ticks
