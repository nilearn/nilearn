"""
Tools for computing functional connectivity matrices and also
implementation of algorithm for sparse multi subjects learning
of Gaussian graphical models.
"""

from .connectivity_matrices import (sym_matrix_to_vec, vec_to_sym_matrix,
                                    sym_to_vec, ConnectivityMeasure,
                                    cov_to_corr, prec_to_partial)

from .group_sparse_cov import (GroupSparseCovariance,
    GroupSparseCovarianceCV, group_sparse_covariance)

__all__ = ['sym_matrix_to_vec', 'vec_to_sym_matrix', 'sym_to_vec',
           'ConnectivityMeasure', 'cov_to_corr', 'prec_to_partial',
           'GroupSparseCovariance', 'GroupSparseCovarianceCV',
           'group_sparse_covariance']
