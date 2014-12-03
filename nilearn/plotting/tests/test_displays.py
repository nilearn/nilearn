# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
import nose
import tempfile

try:
    import matplotlib as mp
    # Make really sure that we don't try to open an Xserver connection.
    mp.use('svg', warn=False)
    import pylab as pl
    pl.switch_backend('svg')
except ImportError:
    raise nose.SkipTest('Could not import matplotlib')

from ..displays import OrthoSlicer, XSlicer, OrthoProjector
from ...datasets import load_mni152_template

################################################################################
# Some smoke testing for graphics-related code

def test_demo_ortho_slicer():
    # This is only a smoke test
    mp.use('svg', warn=False)
    import pylab as pl
    pl.switch_backend('svg')
    pl.clf()
    oslicer = OrthoSlicer(cut_coords=(0, 0, 0))
    img = load_mni152_template()
    oslicer.add_overlay(img, cmap=pl.cm.gray)
    oslicer.close()


def test_stacked_slicer():
    # Test stacked slicers, like the XSlicer
    mp.use('svg', warn=False)
    import pylab as pl
    pl.switch_backend('svg')
    pl.clf()
    img = load_mni152_template()
    slicer = XSlicer.init_with_figure(img=img, cut_coords=3)
    slicer.add_overlay(img, cmap=pl.cm.gray)
    # Forcing a layout here, to test the locator code
    slicer.savefig(tempfile.TemporaryFile())
    slicer.close()


def test_demo_ortho_projector():
    # This is only a smoke test
    mp.use('svg', warn=False)
    import pylab as pl
    pl.switch_backend('svg')
    pl.clf()
    img = load_mni152_template()
    oprojector = OrthoProjector.init_with_figure(img=img)
    oprojector.add_overlay(img, cmap=pl.cm.gray)
    oprojector.savefig(tempfile.TemporaryFile())
    oprojector.close()
