"""
Test functions for models.regression
"""

import numpy as np

from nistats.regression import OLSModel, ARModel


RNG = np.random.RandomState(20110902)
X = RNG.standard_normal((40, 10))
Y = RNG.standard_normal((40,))


def test_OLS():
    model = OLSModel(design=X)
    results = model.fit(Y)
    assert results.df_resid == 30


def test_AR():
    model = ARModel(design=X, rho=0.4)
    results = model.fit(Y)
    assert results.df_resid == 30


def test_OLS_degenerate():
    Xd = X.copy()
    Xd[:, 0] = Xd[:, 1] + Xd[:, 2]
    model = OLSModel(design=Xd)
    results = model.fit(Y)
    assert results.df_resid == 31


def test_AR_degenerate():
    Xd = X.copy()
    Xd[:, 0] = Xd[:, 1] + Xd[:, 2]
    model = ARModel(design=Xd, rho=0.9)
    results = model.fit(Y)
    assert results.df_resid == 31
