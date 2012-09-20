"""
Test the signals module
"""
# Author: Gael Varoquaux, Alexandre Abraham
# License: simplified BSD

import numpy as np

from nose.tools import assert_true, assert_false

from .. import signals
from ..signals import clean

def test_standardize():
    a = np.random.random((10, 10))
    b = signals.standardize(a)
    np.testing.assert_allclose((b**2).sum(axis=-1), np.ones(10))

# The test is inspired from scipy docstring of detrend function
def test_clean_detrending():
    randgen = np.random.RandomState(0)
    npoints = 1e3
    noise = randgen.randn(npoints)
    x = 2 * np.linspace(0, 1, npoints) + noise
    x_detrended = signals.clean([x], normalize=False,
            detrend=True, low_pass=False)[0]
    x_undetrended = signals.clean([x], normalize=False, low_pass=False)[0]
    assert_false((x_undetrended - noise).max() < 0.06)
    assert_true((x_detrended - noise).max() < 0.06)

def test_clean_frequencies():
    sx = np.sin(np.linspace(0, 100, 2000))
    assert_true(clean([sx], normalize=False, high_pass = 0.01).max() > 0.1)
    assert_true(clean([sx], normalize=False, high_pass = 0.2).max() < 0.01)
    assert_true(clean([sx], normalize=False, low_pass = 0.01).max() > 0.9)
    assert_true(clean([sx], normalize=False, low_pass = 0.0005).max() < 0.01)
