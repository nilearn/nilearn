"""Define a Massively Univariate Linear Model estimated \
with OLS and permutation test.
"""

from .permuted_least_squares import permuted_ols

__all__ = ["permuted_ols"]
