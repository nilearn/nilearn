"""
Tests for the permuted_ols function.

"""

import os
import numpy as np
from scipy import sparse
from sklearn.utils import check_random_state

from numpy.testing import (assert_almost_equal, assert_array_almost_equal,
                           assert_equal)

from nilearn.mass_univariate import permuted_ols


### Tests for labels swapping permutation scheme ##############################
def test_permuted_ols_gstat():
    """Compare the results to a former implementation.

    """
    # Load input data
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    data = np.load(os.path.join(cur_dir, 'testing_data.npz'))
    n_perm = data['n_perm']
    tested_vars = data['x']
    imaging_vars = np.vstack((data['y_1'], data['y_2']))
    confounding_vars = data['z']

    # Run permuted OLS
    pvals, h1, h0, params = permuted_ols(
        tested_vars, imaging_vars, confounding_vars, model_intercept=False,
        n_perm=n_perm, sparsity_threshold=0.5, n_jobs=1)

    # Load data to compare to
    ar = np.load(os.path.join(cur_dir, 'res_gstat_test_MULM_OLS.npz'))
    # comparison
    assert_almost_equal(ar['h0'], h0)
    h1_mat = sparse.coo_matrix(
        (h1['score'], (h1['x_id'], h1['y_id']))).todense()
    h1_mat_ar = ar['h1']
    h1_mat_ar = sparse.coo_matrix(
        (h1_mat_ar['data'],
         (h1_mat_ar['snp'], h1_mat_ar['vox']))).todense()
    assert_array_almost_equal(h1_mat, h1_mat_ar)
    for param_name, param_value in params.iteritems():
        assert_equal(param_value, ar['param'].tolist()[param_name])


def test_permuted_ols_sklearn_nocovar(random_state=0):
    rng = check_random_state(random_state)
    # design parameters
    n_samples = 50
    # create design
    target_var = rng.randn(n_samples, 1)
    tested_var = rng.randn(n_samples, 1)
    # scikit-learn F-score
    from sklearn.feature_selection import f_regression
    fvals, _ = f_regression(target_var, tested_var, center=False)
    # permuted OLS (sparsity_threshold=1. to get all values)
    _, all_scores, _, _ = permuted_ols(
        tested_var, target_var.T, model_intercept=False,
        n_perm=0, sparsity_threshold=1., n_jobs=1)
    assert_array_almost_equal(fvals, all_scores['score'], decimal=6)

    # test with ravelized tested_var
    _, all_scores, _, _ = permuted_ols(
        np.ravel(tested_var), target_var.T, model_intercept=False,
        n_perm=0, sparsity_threshold=1., n_jobs=1)
    assert_array_almost_equal(fvals, all_scores['score'], decimal=6)

    ### Adds intercept (should be equivalent to centering variates)
    # permuted OLS (sparsity_threshold=1. to get all values)
    _, all_scores_addintercept, _, _ = permuted_ols(
        tested_var, target_var.T, model_intercept=True,
        n_perm=0, sparsity_threshold=1., n_jobs=1)
    target_var -= target_var.mean(0)
    tested_var -= tested_var.mean(0)
    # scikit-learn F-score
    fvals_addintercept, _ = f_regression(target_var, tested_var, center=True)
    assert_array_almost_equal(
        fvals_addintercept, all_scores_addintercept['score'], decimal=6)


def test_permuted_ols_statsmodels_withcovar(random_state=0):
    """

    This test has a statsmodels dependance. There seems to be no simple,
    alternative way to perform a F-test on a linear model including
    covariates.

    """
    try:
        from statsmodels.regression.linear_model import OLS
    except:
        return

    rng = check_random_state(random_state)
    # design parameters
    n_samples = 50
    # create design
    target_var = rng.randn(n_samples, 1)
    tested_var = rng.randn(n_samples, 1)
    confounding_vars = rng.randn(n_samples, 2)
    # statsmodels OLS
    ols = OLS(target_var, np.hstack((tested_var, confounding_vars))).fit()
    fvals = ols.f_test([[1., 0., 0.]]).fvalue[0]
    # permuted OLS (sparsity_threshold=1. to get all values)
    _, all_scores, _, _ = permuted_ols(
        tested_var, target_var.T, confounding_vars, model_intercept=False,
        n_perm=0, sparsity_threshold=1., n_jobs=1)
    assert_array_almost_equal(fvals, all_scores['score'], decimal=6)

    ### Adds intercept
    # permuted OLS (sparsity_threshold=1. to get all values)
    _, all_scores_addintercept, _, _ = permuted_ols(
        tested_var, target_var.T, confounding_vars, model_intercept=True,
        n_perm=0, sparsity_threshold=1., n_jobs=1)
    # statsmodels OLS
    confounding_vars = np.hstack((confounding_vars, np.ones((n_samples, 1))))
    ols = OLS(target_var, np.hstack((tested_var, confounding_vars))).fit()
    fvals_addintercept = ols.f_test([[1., 0., 0., 0.]]).fvalue[0]
    assert_array_almost_equal(
        fvals_addintercept, all_scores_addintercept['score'], decimal=6)


def test_permuted_ols_sklearn_nocovar_multivariate(random_state=0):
    rng = check_random_state(random_state)
    # design parameters
    n_samples = 50
    n_targets = 10
    # create design
    target_vars = rng.randn(n_samples, n_targets)
    tested_var = rng.randn(n_samples, 1)
    # scikit-learn F-scores
    from sklearn.feature_selection import f_regression
    fvals = np.empty(n_targets)
    for i in range(n_targets):
        fvals[i], _ = f_regression(target_vars[:, i], tested_var, center=False)
    # permuted OLS (sparsity_threshold=1. to get all values)
    _, all_scores, _, _ = permuted_ols(
        tested_var, target_vars.T, model_intercept=False,
        n_perm=0, sparsity_threshold=1., n_jobs=1)
    assert_array_almost_equal(fvals, all_scores['score'], decimal=6)

    ### Adds intercept (should be equivalent to centering variates)
    # permuted OLS (sparsity_threshold=1. to get all values)
    _, all_scores_addintercept, _, _ = permuted_ols(
        tested_var, target_vars.T, model_intercept=True,
        n_perm=0, sparsity_threshold=1., n_jobs=1)
    target_vars -= target_vars.mean(0)
    tested_var -= tested_var.mean(0)
    # scikit-learn F-score
    fvals_addintercept = np.empty(n_targets)
    for i in range(n_targets):
        fvals_addintercept[i], _ = f_regression(
            target_vars[:, i], tested_var, center=True)
    assert_array_almost_equal(
        fvals_addintercept, all_scores_addintercept['score'], decimal=6)


def test_permuted_ols_statsmodels_withcovar_multivariate(random_state=0):
    """

    This test has a statsmodels dependance. There seems to be no simple,
    alternative way to perform a F-test on a linear model including
    covariates.

    """
    try:
        from statsmodels.regression.linear_model import OLS
    except:
        return

    rng = check_random_state(random_state)
    # design parameters
    n_samples = 50
    n_targets = 10
    n_covars = 2
    # create design
    target_vars = rng.randn(n_samples, n_targets)
    tested_var = rng.randn(n_samples, 1)
    confounding_vars = rng.randn(n_samples, n_covars)
    # statsmodels OLS
    fvals = np.empty(n_targets)
    test_matrix = np.array([[1.] + [0.] * n_covars])
    for i in range(n_targets):
        ols = OLS(
            target_vars[:, i], np.hstack((tested_var, confounding_vars)))
        fvals[i] = ols.fit().f_test(test_matrix).fvalue[0][0]
    # permuted OLS (sparsity_threshold=1. to get all values)
    _, all_scores, _, _ = permuted_ols(
        tested_var, target_vars.T, confounding_vars, model_intercept=False,
        n_perm=0, sparsity_threshold=1., n_jobs=1)
    assert_almost_equal(fvals, all_scores['score'], decimal=6)

    ### Adds intercept
    # permuted OLS (sparsity_threshold=1. to get all values)
    _, all_scores_addintercept, _, _ = permuted_ols(
        tested_var, target_vars.T, confounding_vars, model_intercept=True,
        n_perm=0, sparsity_threshold=1., n_jobs=1)
    # statsmodels OLS
    confounding_vars = np.hstack((confounding_vars, np.ones((n_samples, 1))))
    fvals_addintercept = np.empty(n_targets)
    test_matrix = np.array([[1.] + [0.] * (n_covars + 1)])
    for i in range(n_targets):
        ols = OLS(
            target_vars[:, i], np.hstack((tested_var, confounding_vars)))
        fvals_addintercept[i] = ols.fit().f_test(test_matrix).fvalue[0][0]
    assert_array_almost_equal(
        fvals_addintercept, all_scores_addintercept['score'], decimal=6)


### Tests for sign swapping permutation scheme ##############################
def test_permuted_ols_intercept_gstat():
    """Compare the results to a former implementation.

    """
    # Load input data
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    data = np.load(os.path.join(cur_dir, 'testing_data.npz'))
    n_perm = data['n_perm']
    tested_vars_intercept = data['x_intercept']
    imaging_vars = np.vstack((data['y_1'], data['y_2']))
    confounding_vars = data['z']

    # Run permuted OLS (intercept version)
    # (intercept version means we randomly swap the sign of the targets
    #  since it would be useless to randomize the tested --constant-- column)
    pvals, h1, h0, params = permuted_ols(
        tested_vars_intercept, imaging_vars, confounding_vars,
        model_intercept=False, n_perm=n_perm, sparsity_threshold=0.5, n_jobs=1)

    # Load data to compare to
    ar = np.load(os.path.join(
            cur_dir, 'res_gstat_test_MULM_OLS_intercept.npz'))
    # comparison
    assert_almost_equal(ar['h0'], h0)
    h1_mat = sparse.coo_matrix(
        (h1['score'], (h1['x_id'], h1['y_id']))).todense()
    h1_mat_ar = ar['h1']
    h1_mat_ar = sparse.coo_matrix(
        (h1_mat_ar['data'],
         (h1_mat_ar['snp'], h1_mat_ar['vox']))).todense()
    assert_almost_equal(h1_mat, h1_mat_ar)
    for param_name, param_value in params.iteritems():
        assert_equal(param_value, ar['param'].tolist()[param_name])


def test_permuted_ols_intercept_sklearn_nocovar(random_state=0):
    rng = check_random_state(random_state)
    # design parameters
    n_samples = 50
    # create design
    target_var = rng.randn(n_samples, 1)
    tested_var = np.ones((n_samples, 1))
    # scikit-learn F-score
    from sklearn.feature_selection import f_regression
    fvals, _ = f_regression(target_var, tested_var, center=False)
    # permuted OLS (sparsity_threshold=1. to get all values)
    _, all_scores, _, _ = permuted_ols(
        tested_var, target_var.T, confounding_vars=None, n_perm=0,
        sparsity_threshold=1., n_jobs=1)
    # same thing but with model_intercept=True to check it has no effect
    _, all_scores_addintercept, _, _ = permuted_ols(
        tested_var, target_var.T, confounding_vars=None, model_intercept=True,
        n_perm=0, sparsity_threshold=1., n_jobs=1)
    assert_array_almost_equal(fvals, all_scores['score'], decimal=6)
    assert_array_almost_equal(all_scores['score'],
                              all_scores_addintercept['score'], decimal=6)


def test_permuted_ols_intercept_statsmodels_withcovar(random_state=0):
    """

    This test has a statsmodels dependance. There seems to be no simple,
    alternative way to perform a F-test on a linear model including
    covariates.

    """
    try:
        from statsmodels.regression.linear_model import OLS
    except:
        return

    rng = check_random_state(random_state)
    # design parameters
    n_samples = 50
    # create design
    target_var = rng.randn(n_samples, 1)
    tested_var = np.ones((n_samples, 1))
    confounding_vars = rng.randn(n_samples, 2)
    # statsmodels OLS
    ols = OLS(target_var, np.hstack((tested_var, confounding_vars))).fit()
    fvals = ols.f_test([[1., 0., 0.]]).fvalue[0]
    # permuted OLS (sparsity_threshold=1. to get all values)
    _, all_scores, _, _ = permuted_ols(
        tested_var, target_var.T, confounding_vars, n_perm=0,
        sparsity_threshold=1., n_jobs=1)
    # same thing but with model_intercept=True to check it has no effect
    _, all_scores_addintercept, _, _ = permuted_ols(
        tested_var, target_var.T, confounding_vars, model_intercept=True,
        n_perm=0, sparsity_threshold=1., n_jobs=1)
    assert_array_almost_equal(fvals, all_scores['score'], decimal=6)
    assert_array_almost_equal(all_scores['score'],
                              all_scores_addintercept['score'], decimal=6)


def test_permuted_ols_intercept_sklearn_nocovar_multivariate(random_state=0):
    rng = check_random_state(random_state)
    # design parameters
    n_samples = 50
    n_targets = 10
    # create design
    target_vars = rng.randn(n_samples, n_targets)
    tested_var = np.ones((n_samples, 1))
    # scikit-learn F-scores
    from sklearn.feature_selection import f_regression
    fvals = np.empty(n_targets)
    for i in range(n_targets):
        fvals[i], _ = f_regression(target_vars[:, i], tested_var, center=False)
    # permuted OLS (sparsity_threshold=1. to get all values)
    _, all_scores, _, _ = permuted_ols(
        tested_var, target_vars.T, confounding_vars=None, n_perm=0,
        sparsity_threshold=1., n_jobs=1)
    # same thing but with model_intercept=True to check it has no effect
    _, all_scores_addintercept, _, _ = permuted_ols(
        tested_var, target_vars.T, confounding_vars=None, model_intercept=True,
        n_perm=0, sparsity_threshold=1., n_jobs=1)
    assert_array_almost_equal(fvals, all_scores['score'], decimal=6)
    assert_array_almost_equal(all_scores['score'],
                              all_scores_addintercept['score'], decimal=6)


def test_permuted_ols_intercept_statsmodels_withcovar_multivariate(
    random_state=0):
    """

    This test has a statsmodels dependance. There seems to be no simple,
    alternative way to perform a F-test on a linear model including
    covariates.

    """
    try:
        from statsmodels.regression.linear_model import OLS
    except:
        return

    rng = check_random_state(random_state)
    # design parameters
    n_samples = 50
    n_targets = 10
    n_covars = 2
    # create design
    target_vars = rng.randn(n_samples, n_targets)
    tested_var = np.ones((n_samples, 1))
    confounding_vars = rng.randn(n_samples, n_covars)
    # statsmodels OLS
    fvals = np.empty(n_targets)
    test_matrix = np.array([[1.] + [0.] * n_covars])
    for i in range(n_targets):
        ols = OLS(
            target_vars[:, i], np.hstack((tested_var, confounding_vars)))
        fvals[i] = ols.fit().f_test(test_matrix).fvalue[0][0]
    # permuted OLS (sparsity_threshold=1. to get all values)
    _, all_scores, _, _ = permuted_ols(
        tested_var, target_vars.T, confounding_vars, n_perm=0,
        sparsity_threshold=1., n_jobs=1)
    # same thing but with model_intercept=True to check it has no effect
    _, all_scores_addintercept, _, _ = permuted_ols(
        tested_var, target_vars.T, confounding_vars, model_intercept=True,
        n_perm=0, sparsity_threshold=1., n_jobs=1)
    assert_almost_equal(fvals, all_scores['score'], decimal=6)
    assert_array_almost_equal(all_scores['score'],
                              all_scores_addintercept['score'], decimal=6)


if __name__ == '__main__':
    import nose
    nose.run(argv=['', __file__])
