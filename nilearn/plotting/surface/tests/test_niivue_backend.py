"""Test nilearn.plotting.surface.html_surface functions."""

import pytest

from nilearn.plotting.surface._niivue_backend import matplotlib_cm_to_niivue_cm


@pytest.mark.thread_unsafe
def test_matplotlib_cm_to_niivue_cm():
    """Make sure _matplotlib_cm_to_niivue_cm raises errors appropriately."""
    with pytest.warns(
        UserWarning, match="'cmap' must be a str or a Colormap. Got"
    ):
        niivue_cmap = matplotlib_cm_to_niivue_cm(None)
        assert niivue_cmap is None

    with pytest.warns(
        UserWarning, match="'cmap' must be a str or a Colormap. Got"
    ):
        niivue_cmap = matplotlib_cm_to_niivue_cm(1)
        assert niivue_cmap is None

    with pytest.raises(
        ValueError, match="'foo' is not a valid value for name"
    ):
        matplotlib_cm_to_niivue_cm("foo")
