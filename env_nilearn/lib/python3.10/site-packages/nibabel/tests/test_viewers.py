# emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the NiBabel package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##

import unittest
from collections import namedtuple as nt

import numpy as np
import pytest
from numpy.testing import assert_array_equal, assert_equal

from ..optpkg import optional_package
from ..viewers import OrthoSlicer3D

# Need at least MPL 1.3 for viewer tests.
# 2020.02.11 - 1.3 wheels are no longer distributed, so the minimum we test with is 1.5
matplotlib, has_mpl, _ = optional_package('matplotlib', min_version='1.5')

needs_mpl = unittest.skipUnless(has_mpl, 'These tests need matplotlib')
if has_mpl:
    matplotlib.use('Agg')


@needs_mpl
def test_viewer():
    # Test viewer
    plt = optional_package('matplotlib.pyplot')[0]
    a = np.sin(np.linspace(0, np.pi, 20))
    b = np.sin(np.linspace(0, np.pi * 5, 30))
    data = (np.outer(a, b)[..., np.newaxis] * a)[:, :, :, np.newaxis]
    data = data * np.array([1.0, 2.0])  # give it a # of volumes > 1
    v = OrthoSlicer3D(data)
    assert_array_equal(v.position, (0, 0, 0))
    assert 'OrthoSlicer3D' in repr(v)

    # fake some events, inside and outside axes
    v._on_scroll(nt('event', 'button inaxes key')('up', None, None))
    for ax in (v._axes[0], v._axes[3]):
        v._on_scroll(nt('event', 'button inaxes key')('up', ax, None))
    v._on_scroll(nt('event', 'button inaxes key')('up', ax, 'shift'))
    # "click" outside axes, then once in each axis, then move without click
    v._on_mouse(nt('event', 'xdata ydata inaxes button')(0.5, 0.5, None, 1))
    for ax in v._axes:
        v._on_mouse(nt('event', 'xdata ydata inaxes button')(0.5, 0.5, ax, 1))
    v._on_mouse(nt('event', 'xdata ydata inaxes button')(0.5, 0.5, None, None))
    v.set_volume_idx(1)
    v.cmap = 'hot'
    v.clim = (0, 3)
    with pytest.raises(ValueError):
        OrthoSlicer3D.clim.fset(v, (0.0,))  # bad limits
    with pytest.raises(
        (
            ValueError,  # MPL3.5 and lower
            KeyError,  # MPL3.6 and higher
        )
    ):
        OrthoSlicer3D.cmap.fset(v, 'foo')  # wrong cmap

    # decrement/increment volume numbers via keypress
    v.set_volume_idx(1)  # should just pass
    v._on_keypress(nt('event', 'key')('-'))  # decrement
    assert_equal(v._data_idx[3], 0)
    v._on_keypress(nt('event', 'key')('+'))  # increment
    assert_equal(v._data_idx[3], 1)
    v._on_keypress(nt('event', 'key')('-'))
    v._on_keypress(nt('event', 'key')('='))  # alternative increment key
    assert_equal(v._data_idx[3], 1)

    v.close()
    v._draw()  # should be safe

    # non-multi-volume
    v = OrthoSlicer3D(data[:, :, :, 0])
    v._on_scroll(nt('event', 'button inaxes key')('up', v._axes[0], 'shift'))
    v._on_keypress(nt('event', 'key')('escape'))
    v.close()

    # complex input should raise a TypeError prior to figure creation
    with pytest.raises(TypeError):
        OrthoSlicer3D(data[:, :, :, 0].astype(np.complex64))

    # other cases
    fig, axes = plt.subplots(1, 4)
    plt.close(fig)
    v1 = OrthoSlicer3D(data, axes=axes)
    aff = np.array([[0, 1, 0, 3], [-1, 0, 0, 2], [0, 0, 2, 1], [0, 0, 0, 1]], float)
    v2 = OrthoSlicer3D(data, affine=aff, axes=axes[:3])
    # bad data (not 3+ dim)
    with pytest.raises(ValueError):
        OrthoSlicer3D(data[:, :, 0, 0])
    # bad affine (not 4x4)
    with pytest.raises(ValueError):
        OrthoSlicer3D(data, affine=np.eye(3))
    with pytest.raises(TypeError):
        v2.link_to(1)
    v2.link_to(v1)
    v2.link_to(v1)  # shouldn't do anything
    v1.close()
    v2.close()


@needs_mpl
def test_viewer_nonRAS():
    data1 = np.random.rand(10, 20, 40)
    data1[5, 10, :] = 0
    data1[5, :, 30] = 0
    data1[:, 10, 30] = 0
    # RSA affine
    aff1 = np.array([[1, 0, 0, -5], [0, 0, 1, -30], [0, 1, 0, -10], [0, 0, 0, 1]])
    o1 = OrthoSlicer3D(data1, aff1)
    sag = o1._ims[0].get_array()
    cor = o1._ims[1].get_array()
    axi = o1._ims[2].get_array()

    # Sagittal view: [0, I->S, P->A], so data is transposed, matching plot array
    assert_array_equal(sag, data1[5, :, :])
    # Coronal view: [L->R, I->S, 0]. Data is not transposed, transpose to match plot array
    assert_array_equal(cor, data1[:, :, 30].T)
    # Axial view: [L->R, 0, P->A]. Data is not transposed, transpose to match plot array
    assert_array_equal(axi, data1[:, 10, :].T)

    o1.set_position(1, 2, 3)  # R, A, S coordinates

    sag = o1._ims[0].get_array()
    cor = o1._ims[1].get_array()
    axi = o1._ims[2].get_array()

    # Shift 1 right, 2 anterior, 3 superior
    assert_array_equal(sag, data1[6, :, :])
    assert_array_equal(cor, data1[:, :, 32].T)
    assert_array_equal(axi, data1[:, 13, :].T)


@needs_mpl
def test_viewer_nonRAS_on_mouse():
    """
    test on_mouse selection on non RAS matrices

    """
    # This affine simulates an acquisition on a quadruped subject that is in a prone position.
    # This corresponds to an acquisition with:
    # - LR inverted on scanner x (i)
    # - IS on scanner y (j)
    # - PA on scanner z (k)
    # This example enables to test also OrthoSlicer3D properties `_flips` and `_order`.

    (I, J, K) = (10, 20, 40)
    data1 = np.random.rand(I, J, K)
    (i_target, j_target, k_target) = (2, 14, 12)
    i1 = i_target - 2
    i2 = i_target + 2
    j1 = j_target - 3
    j2 = j_target + 3
    k1 = k_target - 4
    k2 = k_target + 4
    data1[i1 : i2 + 1, j1 : j2 + 1, k1 : k2 + 1] = 0
    data1[i_target, j_target, k_target] = 1
    valp1 = 1.5
    valm1 = 0.5
    data1[i_target - 1, j_target, k_target] = valp1  # x flipped
    data1[i_target + 1, j_target, k_target] = valm1  # x flipped
    data1[i_target, j_target - 1, k_target] = valm1
    data1[i_target, j_target + 1, k_target] = valp1
    data1[i_target, j_target, k_target - 1] = valm1
    data1[i_target, j_target, k_target + 1] = valp1

    aff1 = np.array([[-1, 0, 0, 5], [0, 0, 1, -10], [0, 1, 0, -30], [0, 0, 0, 1]])

    o1 = OrthoSlicer3D(data1, aff1)

    class Event:
        def __init__(self):
            self.name = 'simulated mouse event'
            self.button = 1

    event = Event()
    event.xdata = k_target
    event.ydata = j_target
    event.inaxes = o1._ims[0].axes
    o1._on_mouse(event)

    event.inaxes = o1._ims[1].axes
    event.xdata = (I - 1) - i_target  # x flipped
    event.ydata = j_target
    o1._on_mouse(event)

    event.inaxes = o1._ims[2].axes
    event.xdata = (I - 1) - i_target  # x flipped
    event.ydata = k_target
    o1._on_mouse(event)

    sag = o1._ims[0].get_array()
    cor = o1._ims[1].get_array()
    axi = o1._ims[2].get_array()

    assert_array_equal(sag, data1[i_target, :, :])  #
    assert_array_equal(cor, data1[::-1, :, k_target].T)  # x flipped
    assert_array_equal(axi, data1[::-1, j_target, :].T)  # x flipped
    return None


@needs_mpl
def test_viewer_nonRAS_on_scroll():
    """
    test scrolling on non RAS matrices

    """
    # This affine simulates an acquisition on a quadruped subject that is in a prone position.
    # This corresponds to an acquisition with:
    # - LR inverted on scanner x (i)
    # - IS on scanner y (j)
    # - PA on scanner z (k)
    # This example enables to test also OrthoSlicer3D properties `_flips` and `_order`.

    (I, J, K) = (10, 20, 40)
    data1 = np.random.rand(I, J, K)
    (i_target, j_target, k_target) = (2, 14, 12)
    i1 = i_target - 2
    i2 = i_target + 2
    j1 = j_target - 3
    j2 = j_target + 3
    k1 = k_target - 4
    k2 = k_target + 4
    data1[i1 : i2 + 1, j1 : j2 + 1, k1 : k2 + 1] = 0
    data1[i_target, j_target, k_target] = 1
    valp1 = 1.5
    valm1 = 0.5
    data1[i_target - 1, j_target, k_target] = valp1  # x flipped
    data1[i_target + 1, j_target, k_target] = valm1  # x flipped
    data1[i_target, j_target - 1, k_target] = valm1
    data1[i_target, j_target + 1, k_target] = valp1
    data1[i_target, j_target, k_target - 1] = valm1
    data1[i_target, j_target, k_target + 1] = valp1

    aff1 = np.array([[-1, 0, 0, 5], [0, 0, 1, -10], [0, 1, 0, -30], [0, 0, 0, 1]])

    o1 = OrthoSlicer3D(data1, aff1)

    class Event:
        def __init__(self):
            self.name = 'simulated mouse event'
            self.button = None
            self.key = None

    [x_t, y_t, z_t] = list(aff1.dot(np.array([i_target, j_target, k_target, 1]))[:3])
    # print(x_t, y_t, z_t)
    # scanner positions are x_t=3, y_t=2, z_t=16

    event = Event()

    # Sagittal plane - one scroll up
    # x coordinate is flipped so index decrease by 1
    o1.set_position(x_t, y_t, z_t)
    event.inaxes = o1._ims[0].axes
    event.button = 'up'
    o1._on_scroll(event)
    sag = o1._ims[0].get_array()
    cor = o1._ims[1].get_array()
    axi = o1._ims[2].get_array()
    assert_array_equal(sag, data1[i_target - 1, :, :])
    assert_array_equal(cor, data1[::-1, :, k_target].T)  # ::-1 because the array is flipped in x
    assert_array_equal(axi, data1[::-1, j_target, :].T)  # ::-1 because the array is flipped in x

    # Sagittal plane - one scrolled down
    o1.set_position(x_t, y_t, z_t)
    event.button = 'down'
    o1._on_scroll(event)
    sag = o1._ims[0].get_array()
    cor = o1._ims[1].get_array()
    axi = o1._ims[2].get_array()
    assert_array_equal(sag, data1[i_target + 1, :, :])
    assert_array_equal(cor, data1[::-1, :, k_target].T)
    assert_array_equal(axi, data1[::-1, j_target, :].T)

    # Coronal plane - one scroll up
    # y coordinate is increase by 1
    o1.set_position(x_t, y_t, z_t)
    event.inaxes = o1._ims[1].axes
    event.button = 'up'
    o1._on_scroll(event)
    sag = o1._ims[0].get_array()
    cor = o1._ims[1].get_array()
    axi = o1._ims[2].get_array()
    assert_array_equal(sag, data1[i_target, :, :])
    assert_array_equal(
        cor, data1[::-1, :, k_target + 1].T
    )  # ::-1 because the array is flipped in x
    assert_array_equal(axi, data1[::-1, j_target, :].T)  # ::-1 because the array is flipped in x

    # Coronal plane - one scrolled down
    o1.set_position(x_t, y_t, z_t)
    event.button = 'down'
    o1._on_scroll(event)
    sag = o1._ims[0].get_array()
    cor = o1._ims[1].get_array()
    axi = o1._ims[2].get_array()
    assert_array_equal(sag, data1[i_target, :, :])
    assert_array_equal(cor, data1[::-1, :, k_target - 1].T)
    assert_array_equal(axi, data1[::-1, j_target, :].T)

    # Axial plane - one scroll up
    # y is increase by 1
    o1.set_position(x_t, y_t, z_t)
    event.inaxes = o1._ims[2].axes
    event.button = 'up'
    o1._on_scroll(event)
    sag = o1._ims[0].get_array()
    cor = o1._ims[1].get_array()
    axi = o1._ims[2].get_array()
    assert_array_equal(sag, data1[i_target, :, :])
    assert_array_equal(cor, data1[::-1, :, k_target].T)  # ::-1 because the array is flipped in x
    assert_array_equal(
        axi, data1[::-1, j_target + 1, :].T
    )  # ::-1 because the array is flipped in x

    # Axial plane - one scrolled down
    o1.set_position(x_t, y_t, z_t)
    event.button = 'down'
    o1._on_scroll(event)
    sag = o1._ims[0].get_array()
    cor = o1._ims[1].get_array()
    axi = o1._ims[2].get_array()
    assert_array_equal(sag, data1[i_target, :, :])
    assert_array_equal(cor, data1[::-1, :, k_target].T)
    assert_array_equal(axi, data1[::-1, j_target - 1, :].T)
    return None
