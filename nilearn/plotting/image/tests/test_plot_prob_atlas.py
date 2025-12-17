"""Tests for :func:`nilearn.plotting.plot_prob_atlas`."""

# ruff: noqa: ARG001

import pytest

from nilearn.plotting import plot_prob_atlas


@pytest.mark.slow
@pytest.mark.parametrize(
    "params",
    [
        {"view_type": "contours"},
        {"view_type": "filled_contours", "threshold": 0.2},
        {"view_type": "continuous"},
        {"view_type": "filled_contours"},
        {"threshold": None},
    ],
)
@pytest.mark.parametrize("colorbar", [True, False])
@pytest.mark.parametrize("n_regions", [1, 3])
def test_plot_prob_atlas(
    matplotlib_pyplot, params, img_maps, colorbar, n_regions
):
    """Smoke tests for plot_prob_atlas.

    Tests different combinations of parameters `view_type`, `threshold`,
    and `colorbar`.
    """
    if colorbar and n_regions == 1:
        with pytest.warns(
            RuntimeWarning, match="The image maps contains a single image"
        ):
            plot_prob_atlas(img_maps, colorbar=colorbar, **params)
    else:
        plot_prob_atlas(img_maps, colorbar=colorbar, **params)


@pytest.mark.slow
def test_plot_prob_atlas_radiological_view(matplotlib_pyplot, img_4d_rand_eye):
    """Smoke test for radiological view."""
    result = plot_prob_atlas(img_4d_rand_eye, radiological=True)
    assert result.axes.get("y").radiological is True
