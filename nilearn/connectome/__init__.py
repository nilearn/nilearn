"""
Tools for computing functional connectivity matrices and also
implementation of algorithm for sparse multi subjects learning
of Gaussian graphical models.
"""

from .connectivity_matrices import sym_to_vec, ConnectivityMeasure

from .group_sparse_cov import (GroupSparseCovariance,
    GroupSparseCovarianceCV, group_sparse_covariance)

__all__ = ['sym_to_vec', 'ConnectivityMeasure',
           'GroupSparseCovariance', 'GroupSparseCovarianceCV',
           'group_sparse_covariance']
