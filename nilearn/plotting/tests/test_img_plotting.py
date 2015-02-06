
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
import tempfile

import numpy as np

from nose import SkipTest
from nose.tools import assert_raises, assert_true
from functools import partial

try:
    import matplotlib as mp
    # Make really sure that we don't try to open an Xserver connection.
    mp.use('svg', warn=False)
    import pylab as pl
    pl.switch_backend('svg')
except ImportError:
    raise SkipTest('Could not import matplotlib')

import nibabel

from ...image.resampling import coord_transform
from ..img_plotting import MNI152TEMPLATE, plot_anat, plot_img, plot_roi,\
    plot_stat_map, plot_epi, plot_glass_brain

mni_affine = np.array([[  -2.,    0.,    0.,   90.],
                        [   0.,    2.,    0., -126.],
                        [   0.,    0.,    2.,  -72.],
                        [   0.,    0.,    0.,    1.]])


def demo_plot_roi(**kwargs):
    """ Demo plotting an ROI
    """
    mni_affine = MNI152TEMPLATE.get_affine()
    data = np.zeros((91, 109, 91))
    # Color a asymetric rectangle around Broca area:
    x, y, z = -52, 10, 22
    x_map, y_map, z_map = coord_transform(x, y, z,
                                          np.linalg.inv(mni_affine))
    data[int(x_map)-5:int(x_map)+5, int(y_map)-3:int(y_map)+3,
         int(z_map)-10:int(z_map)+10] = 1
    img = nibabel.Nifti1Image(data, mni_affine)
    return plot_roi(img, title="Broca's area", **kwargs)


def test_demo_plot_roi():
    # This is only a smoke test
    mp.use('svg', warn=False)
    import pylab as pl
    pl.switch_backend('svg')
    demo_plot_roi()
    # Test the black background code path
    demo_plot_roi(black_bg=True)

    out = demo_plot_roi(output_file=tempfile.TemporaryFile(suffix='.png'))
    assert_true(out is None)


def test_plot_functions():
    # This is only a smoke test
    mp.use('svg', warn=False)
    import pylab as pl
    pl.switch_backend('svg')
    data_positive = np.zeros((7, 7, 3))
    data_positive[1:-1, 2:-1, 1:] = 1
    data_negative = -data_positive
    rng = np.random.RandomState(42)
    data_heterogeneous = data_positive * rng.randn(*data_positive.shape)
    img_positive = nibabel.Nifti1Image(data_positive, mni_affine)
    img_negative = nibabel.Nifti1Image(data_negative, mni_affine)
    img_heterogeneous = nibabel.Nifti1Image(data_heterogeneous, mni_affine)

    # Test saving with empty plot
    z_slicer = plot_anat(anat_img=False, display_mode='z')
    z_slicer = plot_anat(display_mode='z')
    z_slicer.add_edges(img_positive, color='c')
    z_slicer.savefig(tempfile.TemporaryFile())
    for img in [img_positive, img_negative, img_heterogeneous]:
        ortho_slicer = plot_anat(img, dim=True)
        ortho_slicer.savefig(tempfile.TemporaryFile())

        for func in [plot_anat, plot_img, plot_stat_map,
                 plot_epi, plot_glass_brain,
                 partial(plot_stat_map, symmetric_cbar=True),
                 partial(plot_stat_map, symmetric_cbar=False),
                 partial(plot_stat_map, symmetric_cbar=False, vmax=10),
                 partial(plot_stat_map, symmetric_cbar=True, vmax=10),
                 partial(plot_stat_map, colorbar=False)
                 ]:
            ortho_slicer = func(img, cut_coords=(80, -120, -60))
            # Saving forces a draw, and thus smoke-tests the axes locators
            ortho_slicer.savefig(tempfile.TemporaryFile())
            ortho_slicer.add_edges(img, color='c')

            # Smoke test coordinate finder, with and without mask
            masked_img = nibabel.Nifti1Image(
                np.ma.masked_equal(img.get_data(), 0),
                mni_affine)
            func(masked_img, display_mode='x')
            func(img, display_mode='y')

            out = func(img,
                       output_file=tempfile.TemporaryFile(suffix='.png'))
            assert_true(out is None)
        pl.close('all')


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
    slicer = plot_img(img, display_mode='y', threshold=1)
    slicer.close()
    pl.close('all')


def test_plot_img_invalid():
    # Check that we get a meaningful error message when we give a wrong
    # display_mode argument
    assert_raises(Exception, plot_anat, display_mode='zzz')


def test_plot_img_with_auto_cut_coords():
    import pylab as pl
    pl.switch_backend('svg')
    data = np.zeros((20, 20, 20))
    data[3:-3, 3:-3, 3:-3] = 1
    img = nibabel.Nifti1Image(data, np.eye(4))

    for display_mode in 'xyz':
        plot_img(img, cut_coords=None, display_mode=display_mode,
                 black_bg=True)


def test_plot_img_with_resampling():
    import pylab as pl
    pl.switch_backend('svg')
    data = MNI152TEMPLATE.get_data()[:5, :5, :5]
    affine = np.array([[ 1., -1.,  0.,  0.],
                       [ 1.,  1.,  0.,  0.],
                       [ 0.,  0.,  1.,  0.],
                       [ 0.,  0.,  0.,  1.]])
    img = nibabel.Nifti1Image(data, affine)
    display = plot_img(img)
    display.add_overlay(img)
