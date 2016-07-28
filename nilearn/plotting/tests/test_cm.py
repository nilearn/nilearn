# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
""" Smoke testing the cm module
"""
import matplotlib.pyplot as plt

from nilearn.plotting.cm import dim_cmap, replace_inside


def test_dim_cmap():
    # This is only a smoke test
    dim_cmap(plt.cm.jet)


def test_replace_inside():
    # This is only a smoke test
    replace_inside(plt.cm.jet, plt.cm.hsv, .2, .8)
    # We also test with gnuplot, which is defined using function
    if hasattr(plt.cm, 'gnuplot'):
        # gnuplot is only in recent version of MPL
        replace_inside(plt.cm.gnuplot, plt.cm.gnuplot2, .2, .8)


def test_cm_preload():
    plt.imshow([list(range(10))], cmap="cold_hot")
