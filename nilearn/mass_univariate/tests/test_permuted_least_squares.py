"""
Tests for the permuted_ols function.

"""
# Author: Virgile Fritsch, <virgile.fritsch@inria.fr>, Feb. 2014
import numpy as np
from scipy import stats
from sklearn.utils import check_random_state

from numpy.testing import (assert_almost_equal, assert_array_almost_equal,
                           assert_array_less, assert_equal)

from nilearn.mass_univariate import permuted_ols
from nilearn.mass_univariate.permuted_least_squares import (
    _f_score, orthonormalize_matrix)

from nilearn._utils.fixes import f_regression


### Tests F-scores computation ################################################
def test_f_score_nocovar(random_state=0):
    rng = check_random_state(random_state)

    ### Basic test
    # design parameters
    n_samples = 50
    # generate data
    var1 = rng.randn(n_samples, 1)
    var2 = rng.randn(n_samples, 1)
    f_val_own = _f_score(var1, var2, normalized_design=False)[0]
    f_val_sklearn, _ = f_regression(var2, np.ravel(var1),
                                    center=False)
    assert_array_almost_equal(f_val_own, f_val_sklearn)

    ### Normalized data
    # generate data
    var1 = np.ones((n_samples, 1)) / np.sqrt(n_samples)
    var2 = rng.randn(n_samples, 1)
    var2 = var2 / np.sqrt(np.sum(var2 ** 2, 0))  # normalize
    f_val_own = _f_score(var1, var2, normalized_design=True)[0]
    f_val_own2 = _f_score(var1, var2, normalized_design=False)[0]
    f_val_sklearn, _ = f_regression(var2, np.ravel(var1),
                                    center=False)
    assert_array_almost_equal(f_val_own, f_val_sklearn)
    assert_array_almost_equal(f_val_own, f_val_own2)


def test_f_score_withcovar(random_state=0):
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

    ### Basic test
    # design parameters
    n_samples = 50
    # generate data
    var1 = rng.randn(n_samples, 1)
    var2 = rng.randn(n_samples, 1)
    covars = rng.randn(n_samples, 3)
    # own f_score
    f_val_own = _f_score(var1, var2, covars,
                         normalized_design=False)[0]
    # statsmodels f_score
    test_matrix = np.array([[1., 0., 0., 0.]])
    statsmodels_ols = OLS(var2, np.hstack((var1, covars))).fit()
    f_val_statsmodels = statsmodels_ols.f_test(test_matrix).fvalue[0]
    assert_array_almost_equal(f_val_own, f_val_statsmodels)
    # Same thing with an intercept
    # generate data
    var1 = rng.randn(n_samples, 1)
    var2 = rng.randn(n_samples, 1)
    covars = np.hstack((rng.randn(n_samples, 3), np.ones((n_samples, 1))))
    # own f_score
    f_val_own = _f_score(var1, var2, covars,
                         normalized_design=False)[0]
    # statsmodels f_score
    test_matrix = np.array([[1., 0., 0., 0., 0.]])
    statsmodels_ols = OLS(var2, np.hstack((var1, covars))).fit()
    f_val_statsmodels = statsmodels_ols.f_test(test_matrix).fvalue[0]
    assert_array_almost_equal(f_val_own, f_val_statsmodels)

    ### Normalized data
    # generate data
    var1 = np.ones((n_samples, 1)) / np.sqrt(n_samples)  # normalized
    var2 = rng.randn(n_samples, 1)
    var2 = var2 / np.sqrt(np.sum(var2 ** 2, 0))  # normalize
    covars = np.eye(n_samples, 3)  # covars is orthogonal
    covars[3] = -1  # covars is orthogonal to var1
    covars = orthonormalize_matrix(covars)
    f_val_own = _f_score(var1, var2, covars,
                         normalized_design=True, lost_dof=3)[0]
    f_val_own2 = _f_score(var1, var2, covars,
                          normalized_design=False)[0]
    test_matrix = np.array([[1., 0., 0., 0.]])
    statsmodels_ols = OLS(var2, np.hstack((var1, covars))).fit()
    f_val_statsmodels = statsmodels_ols.f_test(test_matrix).fvalue[0]
    assert_array_almost_equal(f_val_own, f_val_statsmodels)
    assert_array_almost_equal(f_val_own, f_val_own2)


### General tests for permuted_ols function ###################################
def test_permuted_ols_check_h0_noeffect(random_state=0):
    rng = check_random_state(random_state)
    # design parameters
    n_samples = 100
    # create dummy design with no effect
    target_var = rng.randn(n_samples, 1)
    tested_var = np.arange(n_samples).reshape((-1, 1))
    tested_var_not_centered = tested_var.copy()
    tested_var -= tested_var.mean(0)  # centered
    # permuted OLS (sparsity_threshold=1. to get all values)
    # We check that h0 is close to the theoretical distribution, which is
    # known for this simple design (= F(1, 1 - n_samples)).
    perm_ranges = [10, 100, 1000]  # test various number of permutations
    # we use two models (with and without intercept modelling)
    all_kstest_pvals = []
    all_kstest_pvals_intercept = []
    all_kstest_pvals_intercept2 = []
    # we compute the Mean Squared Error between cumulative Density Function
    # as a proof of consistency of the permutation algorithm
    all_mse = []
    all_mse_intercept = []
    all_mse_intercept2 = []
    for i, n_perm in enumerate(np.repeat(perm_ranges, 10)):
        ### Case no. 1: no intercept in the model
        pval, orig_scores, h0 = permuted_ols(
            tested_var, target_var, model_intercept=False,
            n_perm=n_perm, random_state=i)
        assert_equal(h0.size, n_perm)
        # Kolmogorov-Smirnov test
        kstest_pval = stats.kstest(h0, stats.f(1, n_samples - 1).cdf)[1]
        all_kstest_pvals.append(kstest_pval)
        mse = np.mean(
            (stats.f(1, n_samples - 1).cdf(np.sort(h0))
             - np.linspace(0, 1, h0.size + 1)[1:]) ** 2)
        all_mse.append(mse)
        ### Case no. 2: intercept in the model
        pval, orig_scores, h0 = permuted_ols(
            tested_var, target_var, model_intercept=True,
            n_perm=n_perm, random_state=i)
        assert_array_less(pval, 1.)  # pval should not be significant
        # Kolmogorov-Smirnov test
        kstest_pval = stats.kstest(h0, stats.f(1, n_samples - 2).cdf)[1]
        all_kstest_pvals_intercept.append(kstest_pval)
        mse = np.mean(
            (stats.f(1, n_samples - 2).cdf(np.sort(h0))
             - np.linspace(0, 1, h0.size + 1)[1:]) ** 2)
        all_mse_intercept.append(mse)
        ### Case no. 3: intercept in the model, no centering of tested vars
        pval, orig_scores, h0 = permuted_ols(
            tested_var_not_centered, target_var, model_intercept=True,
            n_perm=n_perm, random_state=i)
        assert_array_less(pval, 1.)  # pval should not be significant
        # Kolmogorov-Smirnov test
        kstest_pval = stats.kstest(h0, stats.f(1, n_samples - 2).cdf)[1]
        all_kstest_pvals_intercept2.append(kstest_pval)
        mse = np.mean(
            (stats.f(1, n_samples - 2).cdf(np.sort(h0))
             - np.linspace(0, 1, h0.size + 1)[1:]) ** 2)
        all_mse_intercept2.append(mse)
    all_kstest_pvals = np.array(all_kstest_pvals).reshape(
        (len(perm_ranges), -1))
    all_kstest_pvals_intercept = np.array(all_kstest_pvals_intercept).reshape(
        (len(perm_ranges), -1))
    all_mse = np.array(all_mse).reshape((len(perm_ranges), -1))
    all_mse_intercept = np.array(all_mse_intercept).reshape(
        (len(perm_ranges), -1))
    all_mse_intercept2 = np.array(all_mse_intercept2).reshape(
        (len(perm_ranges), -1))
    # check that a difference between distributions is not rejected by KS test
    assert_array_less(0.01, all_kstest_pvals)
    assert_array_less(0.01, all_kstest_pvals_intercept)
    assert_array_less(0.01, all_kstest_pvals_intercept2)
    # consistency of the algorithm: the more permutations, the less the MSE
    assert_array_less(np.diff(all_mse.mean(1)), 0)
    assert_array_less(np.diff(all_mse_intercept.mean(1)), 0)
    assert_array_less(np.diff(all_mse_intercept2.mean(1)), 0)


### Tests for labels swapping permutation scheme ##############################
def test_permuted_ols_sklearn_nocovar(random_state=0):
    rng = check_random_state(random_state)
    # design parameters
    n_samples = 50
    # create design
    target_var = rng.randn(n_samples, 1)
    tested_var = rng.randn(n_samples, 1)
    # scikit-learn F-score
    fvals, _ = f_regression(target_var, tested_var, center=False)
    # permuted OLS (sparsity_threshold=1. to get all values)
    _, orig_scores, _ = permuted_ols(
        tested_var, target_var, model_intercept=False,
        n_perm=0, random_state=random_state)
    assert_array_almost_equal([fvals], orig_scores, decimal=6)

    # test with ravelized tested_var
    _, orig_scores, _ = permuted_ols(
        np.ravel(tested_var), target_var, model_intercept=False,
        n_perm=0, random_state=random_state)
    assert_array_almost_equal([fvals], orig_scores, decimal=6)

    ### Adds intercept (should be equivalent to centering variates)
    # permuted OLS (sparsity_threshold=1. to get all values)
    _, orig_scores_addintercept, _ = permuted_ols(
        tested_var, target_var, model_intercept=True,
        n_perm=0, random_state=random_state)
    target_var -= target_var.mean(0)
    tested_var -= tested_var.mean(0)
    # scikit-learn F-score
    fvals_addintercept, _ = f_regression(target_var, tested_var, center=True)
    assert_array_almost_equal([fvals_addintercept],
                              orig_scores_addintercept, decimal=6)


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
    fvals = ols.f_test([[1., 0., 0.]]).fvalue
    # permuted OLS (sparsity_threshold=1. to get all values)
    _, orig_scores, _ = permuted_ols(
        tested_var, target_var, confounding_vars, model_intercept=False,
        n_perm=0, random_state=random_state)
    assert_array_almost_equal(fvals, orig_scores, decimal=6)

    ### Adds intercept
    # permuted OLS (sparsity_threshold=1. to get all values)
    _, orig_scores_addintercept, _ = permuted_ols(
        tested_var, target_var, confounding_vars, model_intercept=True,
        n_perm=0, random_state=random_state)
    # statsmodels OLS
    confounding_vars = np.hstack((confounding_vars, np.ones((n_samples, 1))))
    ols = OLS(target_var, np.hstack((tested_var, confounding_vars))).fit()
    fvals_addintercept = ols.f_test([[1., 0., 0., 0.]]).fvalue
    assert_array_almost_equal(fvals_addintercept,
                              orig_scores_addintercept, decimal=6)


def test_permuted_ols_sklearn_nocovar_multivariate(random_state=0):
    rng = check_random_state(random_state)
    # design parameters
    n_samples = 50
    n_targets = 10
    n_regressors = 2
    # create design
    target_vars = rng.randn(n_samples, n_targets)
    tested_var = rng.randn(n_samples, n_regressors)
    # scikit-learn F-scores
    fvals = np.empty((n_targets, n_regressors))
    for i in range(n_targets):
        fvals[i], _ = f_regression(tested_var, target_vars[:, i], center=False)
    # permuted OLS (sparsity_threshold=1. to get all values)
    _, orig_scores, _ = permuted_ols(
        tested_var, target_vars, model_intercept=False,
        n_perm=0, random_state=random_state)
    assert_array_almost_equal(fvals, orig_scores, decimal=6)

    ### Adds intercept (should be equivalent to centering variates)
    # permuted OLS (sparsity_threshold=1. to get all values)
    _, orig_scores_addintercept, _ = permuted_ols(
        tested_var, target_vars, model_intercept=True,
        n_perm=0, random_state=random_state)
    target_vars -= target_vars.mean(0)
    tested_var -= tested_var.mean(0)
    # scikit-learn F-score
    fvals_addintercept = np.empty((n_targets, n_regressors))
    for i in range(n_targets):
        fvals_addintercept[i], _ = f_regression(tested_var, target_vars[:, i],
                                                center=True)
    assert_array_almost_equal(fvals_addintercept,
                              orig_scores_addintercept, decimal=6)


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
    fvals = np.empty((n_targets, 1))
    test_matrix = np.array([[1.] + [0.] * n_covars])
    for i in range(n_targets):
        ols = OLS(
            target_vars[:, i], np.hstack((tested_var, confounding_vars)))
        fvals[i] = ols.fit().f_test(test_matrix).fvalue[0][0]
    # permuted OLS (sparsity_threshold=1. to get all values)
    _, orig_scores, _ = permuted_ols(
        tested_var, target_vars, confounding_vars, model_intercept=False,
        n_perm=0, random_state=random_state)
    assert_almost_equal(fvals, orig_scores, decimal=6)

    ### Adds intercept
    # permuted OLS (sparsity_threshold=1. to get all values)
    _, orig_scores_addintercept, _ = permuted_ols(
        tested_var, target_vars, confounding_vars, model_intercept=True,
        n_perm=0, random_state=random_state)
    # statsmodels OLS
    confounding_vars = np.hstack((confounding_vars, np.ones((n_samples, 1))))
    fvals_addintercept = np.empty((n_targets, 1))
    test_matrix = np.array([[1.] + [0.] * (n_covars + 1)])
    for i in range(n_targets):
        ols = OLS(
            target_vars[:, i], np.hstack((tested_var, confounding_vars)))
        fvals_addintercept[i] = ols.fit().f_test(test_matrix).fvalue[0][0]
    assert_array_almost_equal(fvals_addintercept,
                              orig_scores_addintercept, decimal=6)


### Tests for sign swapping permutation scheme ##############################
def test_permuted_ols_intercept_sklearn_nocovar(random_state=0):
    rng = check_random_state(random_state)
    # design parameters
    n_samples = 50
    # create design
    target_var = rng.randn(n_samples, 1)
    tested_var = np.ones((n_samples, 1))
    # scikit-learn F-score
    fvals, _ = f_regression(target_var, tested_var, center=False)
    # permuted OLS (sparsity_threshold=1. to get all values)
    _, orig_scores, _ = permuted_ols(
        tested_var, target_var, confounding_vars=None, n_perm=0,
        random_state=random_state)
    # same thing but with model_intercept=True to check it has no effect
    _, orig_scores_addintercept, _ = permuted_ols(
        tested_var, target_var, confounding_vars=None, model_intercept=True,
        n_perm=0, random_state=random_state)
    assert_array_almost_equal([fvals], orig_scores, decimal=6)
    assert_array_almost_equal(orig_scores, orig_scores_addintercept, decimal=6)


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
    fvals = ols.f_test([[1., 0., 0.]]).fvalue
    # permuted OLS (sparsity_threshold=1. to get all values)
    _, orig_scores, _ = permuted_ols(
        tested_var, target_var, confounding_vars, n_perm=0,
        random_state=random_state)
    # same thing but with model_intercept=True to check it has no effect
    _, orig_scores_addintercept, _ = permuted_ols(
        tested_var, target_var, confounding_vars, model_intercept=True,
        n_perm=0, random_state=random_state)
    assert_array_almost_equal(fvals, orig_scores, decimal=6)
    assert_array_almost_equal(orig_scores, orig_scores_addintercept, decimal=6)


def test_permuted_ols_intercept_sklearn_nocovar_multivariate(random_state=0):
    rng = check_random_state(random_state)
    # design parameters
    n_samples = 50
    n_targets = 10
    # create design
    target_vars = rng.randn(n_samples, n_targets)
    tested_var = np.ones((n_samples, 1))
    # scikit-learn F-scores
    fvals = np.empty((n_targets, 1))
    for i in range(n_targets):
        fvals[i], _ = f_regression(target_vars[:, i], tested_var, center=False)
    # permuted OLS (sparsity_threshold=1. to get all values)
    _, orig_scores, _ = permuted_ols(
        tested_var, target_vars, confounding_vars=None, n_perm=0,
        random_state=random_state)
    # same thing but with model_intercept=True to check it has no effect
    _, orig_scores_addintercept, _ = permuted_ols(
        tested_var, target_vars, confounding_vars=None, model_intercept=True,
        n_perm=0, random_state=random_state)
    assert_array_almost_equal(fvals, orig_scores, decimal=6)
    assert_array_almost_equal(orig_scores, orig_scores_addintercept, decimal=6)


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
    fvals = np.empty((n_targets, 1))
    test_matrix = np.array([[1.] + [0.] * n_covars])
    for i in range(n_targets):
        ols = OLS(
            target_vars[:, i], np.hstack((tested_var, confounding_vars)))
        fvals[i] = ols.fit().f_test(test_matrix).fvalue[0][0]
    # permuted OLS (sparsity_threshold=1. to get all values)
    _, orig_scores, _ = permuted_ols(
        tested_var, target_vars, confounding_vars, n_perm=0,
        random_state=random_state)
    # same thing but with model_intercept=True to check it has no effect
    _, orig_scores_addintercept, _ = permuted_ols(
        tested_var, target_vars, confounding_vars, model_intercept=True,
        n_perm=0, random_state=random_state)
    assert_almost_equal(fvals, orig_scores, decimal=6)
    assert_array_almost_equal(orig_scores, orig_scores_addintercept, decimal=6)


if __name__ == '__main__':
    import nose
    nose.run(argv=['', __file__])
