# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
import tempfile

import numpy as np

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
