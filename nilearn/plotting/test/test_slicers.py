# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
import nose


try:
    import matplotlib as mp
    # Make really sure that we don't try to open an Xserver connection.
    mp.use('svg', warn=False)
    import pylab as pl
    pl.switch_backend('svg')
except ImportError:
    raise nose.SkipTest('Could not import matplotlib')

from ..slicers import OrthoSlicer
from ...datasets import load_mni152_template

################################################################################
# Some smoke testing for graphics-related code

def test_demo_ortho_slicer():
    # This is only a smoke test
    # conditioned on presence of MNI templated
    mp.use('svg', warn=False)
    import pylab as pl
    pl.switch_backend('svg')
    pl.clf()
    oslicer = OrthoSlicer(cut_coords=(0, 0, 0))
    img = load_mni152_template()
    oslicer.add_overlay(img, cmap=pl.cm.gray)
    oslicer.close()


