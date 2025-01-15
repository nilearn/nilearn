# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Smoke testing the cm module."""

import matplotlib.pyplot as plt
import numpy as np
import pytest

from nilearn.plotting.cm import dim_cmap, mix_colormaps, replace_inside


def test_dim_cmap():
    # This is only a smoke test
    dim_cmap(plt.cm.jet)


def test_replace_inside():
    # This is only a smoke test
    replace_inside(plt.cm.jet, plt.cm.hsv, 0.2, 0.8)
    # We also test with gnuplot, which is defined using function
    if hasattr(plt.cm, "gnuplot"):
        # gnuplot is only in recent version of MPL
        replace_inside(plt.cm.gnuplot, plt.cm.gnuplot2, 0.2, 0.8)


def test_cm_preload():
    plt.imshow([list(range(10))], cmap="cold_hot")


def test_mix_colormaps(rng):
    n = 100

    # Mixin map's shape should be equal to that of
    # the foreground and background maps
    foreground_map = rng.random((n, 4))
    background_map = rng.random((n, 4))
    mix_map = mix_colormaps(foreground_map, background_map)
    assert mix_map.shape == (n, 4)
    # Transparency of mixin map should be higher
    # than that of both the background and the foreground maps
    assert np.all(mix_map[:, 3] >= foreground_map[:, 3])
    assert np.all(mix_map[:, 3] >= background_map[:, 3])

    # If foreground and background maps' shapes are different,
    # an Exception should be raised
    background_map = rng.random((n - 1, 4))
    with pytest.raises(ValueError):
        mix_colormaps(foreground_map, background_map)

    # If foreground map is transparent,
    # mixin should be equal to background map
    foreground_map = rng.random((n, 4))
    background_map = rng.random((n, 4))
    foreground_map[:, 3] = 0
    mix_map = mix_colormaps(foreground_map, background_map)
    assert np.allclose(mix_map, background_map)

    # If background map is transparent,
    # mixin should be equal to foreground map
    foreground_map = rng.random((n, 4))
    background_map = rng.random((n, 4))
    background_map[:, 3] = 0
    mix_map = mix_colormaps(foreground_map, background_map)
    assert np.allclose(mix_map, foreground_map)

    # If foreground and background maps are equal,
    # RBG values of the mixin map should be equal
    # to that of the foreground and background maps
    foreground_map = rng.random((n, 4))
    background_map = foreground_map
    mix_map = mix_colormaps(foreground_map, background_map)
    assert np.allclose(mix_map[:, :3], foreground_map[:, :3])
