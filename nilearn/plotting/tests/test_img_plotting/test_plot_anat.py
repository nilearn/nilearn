"""Tests for :func:`nilearn.plotting.plot_anat`."""

import matplotlib.pyplot as plt
import pytest

from nilearn.plotting import plot_anat
from nilearn.plotting.img_plotting import MNI152TEMPLATE


@pytest.mark.parametrize("anat_img", [False, MNI152TEMPLATE])
@pytest.mark.parametrize("display_mode", ["z", "ortho"])
def test_plot_anat_MNI(anat_img, display_mode, tmpdir):
    """Tests for plot_anat with MNI template."""
    slicer = plot_anat(anat_img=anat_img, display_mode=display_mode)
    filename = str(tmpdir.join("test.png"))
    slicer.savefig(filename)
    plt.close()


@pytest.mark.parametrize("anat_img", [False, MNI152TEMPLATE])
@pytest.mark.parametrize("display_mode", ["z", "ortho"])
@pytest.mark.parametrize("cbar_tick_format", ["%.2g", "%i"])
def test_plot_anat_colorbar(anat_img, display_mode, cbar_tick_format, tmpdir):
    """Tests for plot_anat with MNI template and colorbar."""
    slicer = plot_anat(
        anat_img=anat_img,
        display_mode=display_mode,
        colorbar=True,
        cbar_tick_format=cbar_tick_format,
    )
    filename = str(tmpdir.join("test.png"))
    slicer.savefig(filename)
    plt.close()


def test_plot_anat_3d_img(testdata_3d_for_plotting, tmpdir):
    """Smoke test for plot_anat."""
    filename = str(tmpdir.join("test.png"))
    slicer = plot_anat(testdata_3d_for_plotting["img"], dim="auto")
    slicer.savefig(filename)
    plt.close()


def test_plot_img_invalid():
    """Check that we get a meaningful error message when
    we give a wrong display_mode argument.
    """
    pytest.raises(Exception, plot_anat, display_mode="zzz")
