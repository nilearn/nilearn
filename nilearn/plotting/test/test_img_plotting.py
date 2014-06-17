# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
import tempfile

import numpy as np

from nose import SkipTest
try:
    import matplotlib as mp
    # Make really sure that we don't try to open an Xserver connection.
    mp.use('svg', warn=False)
    import pylab as pl
    pl.switch_backend('svg')
except ImportError:
    raise SkipTest('Could not import matplotlib')

import nibabel

from ..img_plotting import demo_plot_roi, plot_anat, plot_img

mni_affine = np.array([[  -2.,    0.,    0.,   90.],
                        [   0.,    2.,    0., -126.],
                        [   0.,    0.,    2.,  -72.],
                        [   0.,    0.,    0.,    1.]])


def test_demo_plot_roi():
    # This is only a smoke test
    mp.use('svg', warn=False)
    import pylab as pl
    pl.switch_backend('svg')
    demo_plot_roi()
    # Test the black background code path
    demo_plot_roi(black_bg=True)


def test_plot_anat():
    # This is only a smoke test
    mp.use('svg', warn=False)
    import pylab as pl
    pl.switch_backend('svg')
    data = np.zeros((20, 20, 20))
    data[3:-3, 3:-3, 3:-3] = 1
    img = nibabel.Nifti1Image(data, mni_affine)
    ortho_slicer = plot_anat(img, dim=True)
    ortho_slicer = plot_anat(img, cut_coords=(80, -120, -60))
    # Saving forces a draw, and thus smoke-tests the axes locators
    pl.savefig(tempfile.TemporaryFile())
    ortho_slicer.edge_map(img, color='c')

    # Test saving with empty plot
    z_slicer = plot_anat(anat_img=False, display_mode='z')
    pl.savefig(tempfile.TemporaryFile())
    z_slicer = plot_anat(display_mode='z')
    pl.savefig(tempfile.TemporaryFile())
    z_slicer.edge_map(img, color='c')
    # Smoke test coordinate finder, with and without mask
    masked_img = nibabel.Nifti1Image(np.ma.masked_equal(data, 0),
                                     mni_affine)
    plot_img(masked_img, display_mode='x')
    plot_img(img, display_mode='y')


def test_plot_img_empty():
    # Test that things don't crash when we give a map with nothing above
    # threshold
    # This is only a smoke test
    mp.use('svg', warn=False)
    import pylab as pl
    pl.switch_backend('svg')
    data = np.zeros((20, 20, 20))
    img = nibabel.Nifti1Image(data, mni_affine)
    plot_anat(img)
    plot_img(img, display_mode='y', threshold=1)
    pl.close('all')


def test_plot_img_with_auto_cut_coords():
    import pylab as pl
    pl.switch_backend('svg')
    data = np.zeros((20, 20, 20))
    data[3:-3, 3:-3, 3:-3] = 1
    img = nibabel.Nifti1Image(data, np.eye(4))

    for display_mode in 'xyz':
        plot_img(img, cut_coords=None, display_mode=display_mode,
                 black_bg=True)
