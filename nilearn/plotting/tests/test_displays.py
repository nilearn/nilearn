# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
import tempfile

import matplotlib.pyplot as plt

from nilearn.plotting.displays import OrthoSlicer, XSlicer, OrthoProjector
from nilearn.datasets import load_mni152_template


##############################################################################
# Some smoke testing for graphics-related code

def test_demo_ortho_slicer():
    # This is only a smoke test
    oslicer = OrthoSlicer(cut_coords=(0, 0, 0))
    img = load_mni152_template()
    oslicer.add_overlay(img, cmap=plt.cm.gray)
    oslicer.close()


def test_stacked_slicer():
    # Test stacked slicers, like the XSlicer
    img = load_mni152_template()
    slicer = XSlicer.init_with_figure(img=img, cut_coords=3)
    slicer.add_overlay(img, cmap=plt.cm.gray)
    # Forcing a layout here, to test the locator code
    with tempfile.TemporaryFile() as fp:
        slicer.savefig(fp)
    slicer.close()


def test_demo_ortho_projector():
    # This is only a smoke test
    img = load_mni152_template()
    oprojector = OrthoProjector.init_with_figure(img=img)
    oprojector.add_overlay(img, cmap=plt.cm.gray)
    with tempfile.TemporaryFile() as fp:
        oprojector.savefig(fp)
    oprojector.close()


def test_contour_fillings_levels_in_add_contours():
    oslicer = OrthoSlicer(cut_coords=(0, 0, 0))
    img = load_mni152_template()
    # levels should be atleast 2
    # If single levels are passed then we force upper level to be inf
    oslicer.add_contours(img, filled=True, colors='r',
                         alpha=0.2, levels=[0.])

    # If two levels are passed, it should be increasing from zero index
    # In this case, we simply omit appending inf
    oslicer.add_contours(img, filled=True, colors='b',
                         alpha=0.1, levels=[0., 0.2])

    # without passing colors and alpha. In this case, default values are
    # chosen from matplotlib
    oslicer.add_contours(img, filled=True, levels=[0., 0.2])

    # levels with only one value
    oslicer.add_contours(img, filled=True, levels=[0.])

    # without passing levels, should work with default levels from
    # matplotlib
    oslicer.add_contours(img, filled=True)


def test_user_given_cmap_with_colorbar():
    img = load_mni152_template()
    oslicer = OrthoSlicer(cut_coords=(0, 0, 0))

    # Test with cmap given as a string
    oslicer.add_overlay(img, cmap='Paired', colorbar=True)
    oslicer.close()
