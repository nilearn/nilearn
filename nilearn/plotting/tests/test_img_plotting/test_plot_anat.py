"""Tests for :func:`nilearn.plotting.plot_anat`."""

import matplotlib.pyplot as plt
import pytest

from nilearn.plotting import plot_anat


def test_plot_anat_3d_img(img_3d_mni, tmp_path):
    """Smoke test for plot_anat."""
    filename = tmp_path / "test.png"
    slicer = plot_anat(img_3d_mni, dim="auto")
    slicer.savefig(filename)
    plt.close()


def test_plot_img_invalid():
    """Check that we get a meaningful error message \
       when we give a wrong display_mode argument.
    """
    with pytest.raises(ValueError):
        plot_anat(display_mode="zzz")
