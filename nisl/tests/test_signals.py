"""
Test the signals module
"""
# Author: Gael Varoquaux
# License: simplified BSD

import numpy as np

from .. import signals

def test_standardize():
    a = np.random.random((10, 10))
    b = signals.standardize(a)
    np.testing.assert_allclose((b**2).sum(axis=-1), np.ones(10))

