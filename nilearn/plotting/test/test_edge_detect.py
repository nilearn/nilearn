# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
import nose

import numpy as np

from ..edge_detect import _edge_detect, _fast_abs_percentile



################################################################################
def test_fast_abs_percentile():
    data = np.arange(1, 100)
    for p in range(10, 100, 10):
        yield nose.tools.assert_equal, _fast_abs_percentile(data, p-1), p


def test_edge_detect():
    img = np.zeros((10, 10))
    img[:5] = 1
    _, edge_mask = _edge_detect(img)
    np.testing.assert_almost_equal(img[4], 1)
