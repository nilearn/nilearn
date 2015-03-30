# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
""" Smoke testing the cm module
"""
from nose import SkipTest

try:
    import matplotlib as mp
    # Make really sure that we don't try to open an Xserver connection.
    mp.use('template', warn=False)
    import matplotlib.pyplot as plt
    plt.switch_backend('template')
except ImportError:
    raise SkipTest('Could not import matplotlib')

from nilearn.plotting.cm import dim_cmap, replace_inside


def test_dim_cmap():
    # This is only a smoke test
    mp.use('template', warn=False)
    import matplotlib.pyplot as plt
    plt.switch_backend('template')
    dim_cmap(plt.cm.jet)


def test_replace_inside():
    # This is only a smoke test
    mp.use('template', warn=False)
    import matplotlib.pyplot as plt
    plt.switch_backend('template')
    replace_inside(plt.cm.jet, plt.cm.hsv, .2, .8)
    # We also test with gnuplot, which is defined using function
    if hasattr(plt.cm, 'gnuplot'):
        # gnuplot is only in recent version of MPL
        replace_inside(plt.cm.gnuplot, plt.cm.gnuplot2, .2, .8)
