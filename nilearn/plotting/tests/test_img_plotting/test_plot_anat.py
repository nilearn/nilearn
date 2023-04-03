"""Tests for :func:`nilearn.plotting.plot_anat`."""

import matplotlib.pyplot as plt
import numpy as np
import pytest
from nibabel import Nifti1Image
from nilearn.plotting import plot_anat
from nilearn.plotting.img_plotting import MNI152TEMPLATE

from .testing_utils import MNI_AFFINE


@pytest.fixture()
def testdata_3d():
    """A random 3D image for testing figures."""
    data_positive = np.zeros((7, 7, 3))
    rng = np.random.RandomState(42)
    data_rng = rng.uniform(size=(7, 7, 3))
    data_positive[1:-1, 2:-1, 1:] = data_rng[1:-1, 2:-1, 1:]
    img_3d = Nifti1Image(data_positive, MNI_AFFINE)
    return {'img': img_3d}


@pytest.mark.parametrize("anat_img", [False, MNI152TEMPLATE])
@pytest.mark.parametrize("display_mode", ['z', 'ortho'])
def test_plot_anat_MNI(anat_img, display_mode, tmpdir):
    """Tests for plot_anat with MNI template."""
    slicer = plot_anat(anat_img=anat_img, display_mode=display_mode)
    filename = str(tmpdir.join('test.png'))
    slicer.savefig(filename)
    plt.close()


@pytest.mark.parametrize("anat_img", [False, MNI152TEMPLATE])
@pytest.mark.parametrize("display_mode", ['z', 'ortho'])
@pytest.mark.parametrize("cbar_tick_format", ["%.2g", "%i"])
def test_plot_anat_colorbar(anat_img, display_mode, cbar_tick_format, tmpdir):
    """Tests for plot_anat with MNI template and colorbar."""
    slicer = plot_anat(
        anat_img=anat_img, display_mode=display_mode, colorbar=True,
        cbar_tick_format=cbar_tick_format
    )
    filename = str(tmpdir.join('test.png'))
    slicer.savefig(filename)
    plt.close()


def test_plot_anat_3d_img(testdata_3d, tmpdir):
    """Smoke test for plot_anat."""
    filename = str(tmpdir.join('test.png'))
    slicer = plot_anat(testdata_3d['img'], dim='auto')
    slicer.savefig(filename)
    plt.close()


def test_plot_img_invalid():
    """Check that we get a meaningful error message when
    we give a wrong display_mode argument.
    """
    pytest.raises(Exception, plot_anat, display_mode='zzz')
