"""

"""
import unittest
import numpy as np
from scipy import sparse

from numpy.testing import assert_almost_equal, assert_equal

from nilearn.nilearn.group_analysis import permuted_OLS


class TestPermutedLeastSquares(unittest.TestCase):
    """

    """
    data = np.load('./testing_data.npz')
    n_perm = data['n_perm']

    tested_vars = data['x']
    tested_vars_intercept = data['x_intercept']
    imaging_vars = np.vstack((data['y_1'], data['y_2']))
    confounding_vars = data['z']

    def test_MULM_OLS(self):
        pvals, h1, h0, params = permuted_OLS(
            self.tested_vars, self.imaging_vars, self.confounding_vars,
            self.n_perm)
        ar = np.load('./res_gstat_test_MULM_OLS.npz')
        assert_almost_equal(ar['h0'], h0)
        h1_mat = sparse.coo_matrix(
            (h1['score'], (h1['testvar_id'], h1['imgvar_id']))).todense()
        h1_mat_ar = ar['h1']
        h1_mat_ar = sparse.coo_matrix(
            (h1_mat_ar['data'],
             (h1_mat_ar['snp'], h1_mat_ar['vox']))).todense()
        assert_almost_equal(h1_mat, h1_mat_ar)
        for param_name, param_value in params.iteritems():
            assert_equal(param_value, ar['param'].tolist()[param_name])

    def test_MULM_OLS_intercept(self):
        pvals, h1, h0, params = permuted_OLS(
            self.tested_vars_intercept, self.imaging_vars,
            self.confounding_vars, self.n_perm)
        ar = np.load('./res_gstat_test_MULM_OLS_intercept.npz')
        assert_almost_equal(ar['h0'], h0)
        h1_mat = sparse.coo_matrix(
            (h1['score'], (h1['testvar_id'], h1['imgvar_id']))).todense()
        h1_mat_ar = ar['h1']
        h1_mat_ar = sparse.coo_matrix(
            (h1_mat_ar['data'],
             (h1_mat_ar['snp'], h1_mat_ar['vox']))).todense()
        assert_almost_equal(h1_mat, h1_mat_ar)
        for param_name, param_value in params.iteritems():
            assert_equal(param_value, ar['param'].tolist()[param_name])


if __name__ == '__main__':
    import nose
    nose.run(argv=['', __file__])
