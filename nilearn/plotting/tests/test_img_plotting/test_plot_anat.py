"""Tests for :func:`nilearn.plotting.plot_anat`."""

import matplotlib.pyplot as plt
import pytest

from nilearn.plotting import plot_anat
from nilearn.plotting.img_plotting import MNI152TEMPLATE


@pytest.mark.parametrize("anat_img", [False, MNI152TEMPLATE])
@pytest.mark.parametrize("display_mode", ["z", "ortho"])
def test_plot_anat_mni(anat_img, display_mode, tmp_path):
    """Tests for plot_anat with MNI template."""
    slicer = plot_anat(anat_img=anat_img, display_mode=display_mode)
    filename = tmp_path / "test.png"
    slicer.savefig(filename)
    plt.close()


@pytest.mark.parametrize("anat_img", [False, MNI152TEMPLATE])
@pytest.mark.parametrize("display_mode", ["z", "ortho"])
@pytest.mark.parametrize("cbar_tick_format", ["%.2g", "%i"])
def test_plot_anat_colorbar(
    anat_img, display_mode, cbar_tick_format, tmp_path
):
    """Tests for plot_anat with MNI template and colorbar."""
    slicer = plot_anat(
        anat_img=anat_img,
        display_mode=display_mode,
        colorbar=True,
        cbar_tick_format=cbar_tick_format,
    )
    filename = tmp_path / "test.png"
    slicer.savefig(filename)
    plt.close()


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
