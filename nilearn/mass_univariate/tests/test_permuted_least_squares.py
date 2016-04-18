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
    _t_score_with_covars_and_normalized_design, orthonormalize_matrix)


def get_tvalue_with_alternative_library(tested_vars, target_vars, covars=None):
    """Utility function to compute tvalues with linalg or statsmodels

    Massively univariate linear model (= each target is considered
    independently).

    Parameters
    ----------
    tested_vars: array-like, shape=(n_samples, n_regressors)
      Tested variates, the associated coefficient of which are to be tested
      independently with a t-test, resulting in as many t-values.

    target_vars: array-like, shape=(n_samples, n_targets)
      Target variates, to be approximated with a linear combination of
      the tested variates and the confounding variates.

    covars: array-like, shape=(n_samples, n_confounds)
      Confounding variates, to be fitted but not to be tested

    Returns
    -------
    t-values: np.ndarray, shape=(n_regressors, n_targets)

    """
    ### set up design
    n_samples, n_regressors = tested_vars.shape
    n_targets = target_vars.shape[1]
    if covars is not None:
        n_covars = covars.shape[1]
        design_matrix = np.hstack((tested_vars, covars))
    else:
        n_covars = 0
        design_matrix = tested_vars
    mask_covars = np.ones(n_regressors + n_covars, dtype=bool)
    mask_covars[:n_regressors] = False
    test_matrix = np.array([[1.] + [0.] * n_covars])

    ### t-values computation
    try:  # try with statsmodels if available (more concise)
        from statsmodels.regression.linear_model import OLS
        t_values = np.empty((n_targets, n_regressors))
        for i in range(n_targets):
            current_target = target_vars[:, i].reshape((-1, 1))
            for j in range(n_regressors):
                current_tested_mask = mask_covars.copy()
                current_tested_mask[j] = True
                current_design_matrix = design_matrix[:, current_tested_mask]
                ols_fit = OLS(current_target, current_design_matrix).fit()
                t_values[i, j] = np.ravel(ols_fit.t_test(test_matrix).tvalue)
    except:  # use linalg if statsmodels is not available
        from numpy import linalg
        lost_dof = n_covars + 1  # fit all tested variates independently
        t_values = np.empty((n_targets, n_regressors))
        for i in range(n_regressors):
            current_tested_mask = mask_covars.copy()
            current_tested_mask[i] = True
            current_design_matrix = design_matrix[:, current_tested_mask]
            invcov = linalg.pinv(current_design_matrix)
            normalized_cov = np.dot(invcov, invcov.T)
            t_val_denom_aux = np.diag(
                np.dot(test_matrix, np.dot(normalized_cov, test_matrix.T)))
            t_val_denom_aux = t_val_denom_aux.reshape((-1, 1))
            for j in range(n_targets):
                current_target = target_vars[:, j].reshape((-1, 1))
                res_lstsq = linalg.lstsq(current_design_matrix, current_target)
                residuals = (current_target
                             - np.dot(current_design_matrix, res_lstsq[0]))
                t_val_num = np.dot(test_matrix, res_lstsq[0])
                t_val_denom = np.sqrt(
                    np.sum(residuals ** 2, 0) / float(n_samples - lost_dof)
                    * t_val_denom_aux)
                t_values[j, i] = np.ravel(t_val_num / t_val_denom)
    return t_values


### Tests t-scores computation ################################################
def test_t_score_with_covars_and_normalized_design_nocovar(random_state=0):
    rng = check_random_state(random_state)

    ### Normalized data
    n_samples = 50
    # generate data
    var1 = np.ones((n_samples, 1)) / np.sqrt(n_samples)
    var2 = rng.randn(n_samples, 1)
    var2 = var2 / np.sqrt(np.sum(var2 ** 2, 0))  # normalize
    # compute t-scores with nilearn routine
    t_val_own = _t_score_with_covars_and_normalized_design(var1, var2)
    # compute t-scores with linalg or statsmodels
    t_val_alt = get_tvalue_with_alternative_library(var1, var2)
    assert_array_almost_equal(t_val_own, t_val_alt)


def test_t_score_with_covars_and_normalized_design_withcovar(random_state=0):
    """

    """
    rng = check_random_state(random_state)

    ### Normalized data
    n_samples = 50
    # generate data
    var1 = np.ones((n_samples, 1)) / np.sqrt(n_samples)  # normalized
    var2 = rng.randn(n_samples, 1)
    var2 = var2 / np.sqrt(np.sum(var2 ** 2, 0))  # normalize
    covars = np.eye(n_samples, 3)  # covars is orthogonal
    covars[3] = -1  # covars is orthogonal to var1
    covars = orthonormalize_matrix(covars)
    # nilearn t-score
    own_score = _t_score_with_covars_and_normalized_design(var1, var2, covars)
    # compute t-scores with linalg or statmodels
    ref_score = get_tvalue_with_alternative_library(var1, var2, covars)
    assert_array_almost_equal(own_score, ref_score)


### General tests for permuted_ols function ###################################
def test_permuted_ols_check_h0_noeffect_labelswap(random_state=0):
    rng = check_random_state(random_state)
    # design parameters
    n_samples = 100
    # create dummy design with no effect
    target_var = rng.randn(n_samples, 1)
    tested_var = np.arange(n_samples, dtype='f8').reshape((-1, 1))
    tested_var_not_centered = tested_var.copy()
    tested_var -= tested_var.mean(0)  # centered
    # permuted OLS
    # We check that h0 is close to the theoretical distribution, which is
    # known for this simple design (= t(n_samples - dof)).
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
            n_perm=n_perm, two_sided_test=False, random_state=i)
        assert_equal(h0.size, n_perm)
        # Kolmogorov-Smirnov test
        kstest_pval = stats.kstest(h0, stats.t(n_samples - 1).cdf)[1]
        all_kstest_pvals.append(kstest_pval)
        mse = np.mean(
            (stats.t(n_samples - 1).cdf(np.sort(h0))
             - np.linspace(0, 1, h0.size + 1)[1:]) ** 2)
        all_mse.append(mse)
        ### Case no. 2: intercept in the model
        pval, orig_scores, h0 = permuted_ols(
            tested_var, target_var, model_intercept=True,
            n_perm=n_perm, two_sided_test=False, random_state=i)
        assert_array_less(pval, 1.)  # pval should not be significant
        # Kolmogorov-Smirnov test
        kstest_pval = stats.kstest(h0, stats.t(n_samples - 2).cdf)[1]
        all_kstest_pvals_intercept.append(kstest_pval)
        mse = np.mean(
            (stats.t(n_samples - 2).cdf(np.sort(h0))
             - np.linspace(0, 1, h0.size + 1)[1:]) ** 2)
        all_mse_intercept.append(mse)
        ### Case no. 3: intercept in the model, no centering of tested vars
        pval, orig_scores, h0 = permuted_ols(
            tested_var_not_centered, target_var, model_intercept=True,
            n_perm=n_perm, two_sided_test=False, random_state=i)
        assert_array_less(pval, 1.)  # pval should not be significant
        # Kolmogorov-Smirnov test
        kstest_pval = stats.kstest(h0, stats.t(n_samples - 2).cdf)[1]
        all_kstest_pvals_intercept2.append(kstest_pval)
        mse = np.mean(
            (stats.t(n_samples - 2).cdf(np.sort(h0))
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


def test_permuted_ols_check_h0_noeffect_signswap(random_state=0):
    rng = check_random_state(random_state)
    # design parameters
    n_samples = 100
    # create dummy design with no effect
    target_var = rng.randn(n_samples, 1)
    tested_var = np.ones((n_samples, 1))
    # permuted OLS
    # We check that h0 is close to the theoretical distribution, which is
    # known for this simple design (= t(n_samples - dof)).
    perm_ranges = [10, 100, 1000]  # test various number of permutations
    all_kstest_pvals = []
    # we compute the Mean Squared Error between cumulative Density Function
    # as a proof of consistency of the permutation algorithm
    all_mse = []
    for i, n_perm in enumerate(np.repeat(perm_ranges, 10)):
        pval, orig_scores, h0 = permuted_ols(
            tested_var, target_var, model_intercept=False,
            n_perm=n_perm, two_sided_test=False, random_state=i)
        assert_equal(h0.size, n_perm)
        # Kolmogorov-Smirnov test
        kstest_pval = stats.kstest(h0, stats.t(n_samples).cdf)[1]
        all_kstest_pvals.append(kstest_pval)
        mse = np.mean(
            (stats.t(n_samples).cdf(np.sort(h0))
             - np.linspace(0, 1, h0.size + 1)[1:]) ** 2)
        all_mse.append(mse)
    all_kstest_pvals = np.array(all_kstest_pvals).reshape(
        (len(perm_ranges), -1))
    all_mse = np.array(all_mse).reshape((len(perm_ranges), -1))
    # check that a difference between distributions is not rejected by KS test
    assert_array_less(0.01 / (len(perm_ranges) * 10.), all_kstest_pvals)
    # consistency of the algorithm: the more permutations, the less the MSE
    assert_array_less(np.diff(all_mse.mean(1)), 0)


### Tests for labels swapping permutation scheme ##############################
def test_permuted_ols_nocovar(random_state=0):
    rng = check_random_state(random_state)
    # design parameters
    n_samples = 50
    # create design
    target_var = rng.randn(n_samples, 1)
    tested_var = rng.randn(n_samples, 1)
    # compute t-scores with linalg or statsmodels
    ref_score = get_tvalue_with_alternative_library(tested_var, target_var)
    # permuted OLS
    _, own_score, _ = permuted_ols(
        tested_var, target_var, model_intercept=False,
        n_perm=0, random_state=random_state)
    assert_array_almost_equal(ref_score, own_score, decimal=6)

    # test with ravelized tested_var
    _, own_score, _ = permuted_ols(
        np.ravel(tested_var), target_var, model_intercept=False,
        n_perm=0, random_state=random_state)
    assert_array_almost_equal(ref_score, own_score, decimal=6)

    ### Adds intercept (should be equivalent to centering variates)
    # permuted OLS
    _, own_score_intercept, _ = permuted_ols(
        tested_var, target_var, model_intercept=True,
        n_perm=0, random_state=random_state)
    target_var -= target_var.mean(0)
    tested_var -= tested_var.mean(0)
    # compute t-scores with linalg or statsmodels
    ref_score_intercept = get_tvalue_with_alternative_library(
        tested_var, target_var, np.ones((n_samples, 1)))
    assert_array_almost_equal(ref_score_intercept, own_score_intercept,
                              decimal=6)


def test_permuted_ols_withcovar(random_state=0):
    """

    """
    rng = check_random_state(random_state)
    # design parameters
    n_samples = 50
    # create design
    target_var = rng.randn(n_samples, 1)
    tested_var = rng.randn(n_samples, 1)
    confounding_vars = rng.randn(n_samples, 2)
    # compute t-scores with linalg or statsmodels
    ref_score = get_tvalue_with_alternative_library(tested_var, target_var,
                                                    confounding_vars)
    # permuted OLS
    _, own_score, _ = permuted_ols(
        tested_var, target_var, confounding_vars, model_intercept=False,
        n_perm=0, random_state=random_state)
    assert_array_almost_equal(ref_score, own_score, decimal=6)

    ### Adds intercept
    # permuted OLS
    _, own_scores_intercept, _ = permuted_ols(
        tested_var, target_var, confounding_vars, model_intercept=True,
        n_perm=0, random_state=random_state)
    # compute t-scores with linalg or statsmodels
    confounding_vars = np.hstack((confounding_vars, np.ones((n_samples, 1))))
    alt_score_intercept = get_tvalue_with_alternative_library(
        tested_var, target_var, confounding_vars)
    assert_array_almost_equal(alt_score_intercept, own_scores_intercept,
                              decimal=6)


def test_permuted_ols_nocovar_multivariate(random_state=0):
    """Test permuted_ols with multiple tested variates and no covariate.

    It is equivalent to fitting several models with only one tested variate.

    """
    rng = check_random_state(random_state)
    # design parameters
    n_samples = 50
    n_targets = 10
    n_regressors = 2
    # create design
    target_vars = rng.randn(n_samples, n_targets)
    tested_var = rng.randn(n_samples, n_regressors)
    # compute t-scores with linalg or statsmodels
    ref_scores = get_tvalue_with_alternative_library(tested_var, target_vars)
    # permuted OLS
    _, own_scores, _ = permuted_ols(
        tested_var, target_vars, model_intercept=False,
        n_perm=0, random_state=random_state)
    assert_array_almost_equal(ref_scores, own_scores, decimal=6)

    ### Adds intercept (should be equivalent to centering variates)
    # permuted OLS
    _, own_scores_intercept, _ = permuted_ols(
        tested_var, target_vars, model_intercept=True,
        n_perm=0, random_state=random_state)
    target_vars -= target_vars.mean(0)
    tested_var -= tested_var.mean(0)
    # compute t-scores with linalg or statsmodels
    ref_scores_intercept = get_tvalue_with_alternative_library(
            tested_var, target_vars, np.ones((n_samples, 1)))
    assert_array_almost_equal(ref_scores_intercept, own_scores_intercept,
                              decimal=6)


def test_permuted_ols_withcovar_multivariate(random_state=0):
    """Test permuted_ols with multiple tested variates and covariates.

    It is equivalent to fitting several models with only one tested variate.

    """
    rng = check_random_state(random_state)
    # design parameters
    n_samples = 50
    n_targets = 10
    n_covars = 2
    # create design
    target_vars = rng.randn(n_samples, n_targets)
    tested_var = rng.randn(n_samples, 1)
    confounding_vars = rng.randn(n_samples, n_covars)
    # compute t-scores with linalg or statmodels
    ref_scores = get_tvalue_with_alternative_library(tested_var, target_vars,
                                                     confounding_vars)
    # permuted OLS
    _, own_scores, _ = permuted_ols(
        tested_var, target_vars, confounding_vars, model_intercept=False,
        n_perm=0, random_state=random_state)
    assert_almost_equal(ref_scores, own_scores, decimal=6)

    ### Adds intercept
    # permuted OLS
    _, own_scores_intercept, _ = permuted_ols(
        tested_var, target_vars, confounding_vars, model_intercept=True,
        n_perm=0, random_state=random_state)
    # compute t-scores with linalg or statmodels
    confounding_vars = np.hstack((confounding_vars, np.ones((n_samples, 1))))
    ref_scores_intercept = get_tvalue_with_alternative_library(
        tested_var, target_vars, confounding_vars)
    assert_array_almost_equal(ref_scores_intercept,
                              own_scores_intercept, decimal=6)


### Tests for sign swapping permutation scheme ##############################
def test_permuted_ols_intercept_nocovar(random_state=0):
    rng = check_random_state(random_state)
    # design parameters
    n_samples = 50
    # create design
    target_var = rng.randn(n_samples, 1)
    tested_var = np.ones((n_samples, 1))
    # compute t-scores with linalg or statmodels
    t_val_ref = get_tvalue_with_alternative_library(tested_var, target_var)
    # permuted OLS
    neg_log_pvals, orig_scores, _ = permuted_ols(
        tested_var, target_var, confounding_vars=None, n_perm=10,
        random_state=random_state)
    assert_array_less(neg_log_pvals, 1.)  # ensure sign swap is correctly done
    # same thing but with model_intercept=True to check it has no effect
    _, orig_scores_addintercept, _ = permuted_ols(
        tested_var, target_var, confounding_vars=None, model_intercept=True,
        n_perm=0, random_state=random_state)
    assert_array_almost_equal(t_val_ref, orig_scores, decimal=6)
    assert_array_almost_equal(orig_scores, orig_scores_addintercept, decimal=6)


def test_permuted_ols_intercept_statsmodels_withcovar(random_state=0):
    rng = check_random_state(random_state)
    # design parameters
    n_samples = 50
    # create design
    target_var = rng.randn(n_samples, 1)
    tested_var = np.ones((n_samples, 1))
    confounding_vars = rng.randn(n_samples, 2)
    # compute t-scores with linalg or statmodels
    ref_scores = get_tvalue_with_alternative_library(tested_var, target_var,
                                                     confounding_vars)
    # permuted OLS
    _, own_scores, _ = permuted_ols(
        tested_var, target_var, confounding_vars, n_perm=0,
        random_state=random_state)
    # same thing but with model_intercept=True to check it has no effect
    _, own_scores_intercept, _ = permuted_ols(
        tested_var, target_var, confounding_vars, model_intercept=True,
        n_perm=0, random_state=random_state)
    assert_array_almost_equal(ref_scores, own_scores, decimal=6)
    assert_array_almost_equal(ref_scores, own_scores_intercept, decimal=6)


def test_permuted_ols_intercept_nocovar_multivariate(random_state=0):
    rng = check_random_state(random_state)
    # design parameters
    n_samples = 50
    n_targets = 10
    # create design
    target_vars = rng.randn(n_samples, n_targets)
    tested_vars = np.ones((n_samples, 1))
    # compute t-scores with nilearn routine
    ref_scores = get_tvalue_with_alternative_library(tested_vars, target_vars)
    # permuted OLS
    _, own_scores, _ = permuted_ols(
        tested_vars, target_vars, confounding_vars=None, n_perm=0,
        random_state=random_state)
    # same thing but with model_intercept=True to check it has no effect
    _, own_scores_intercept, _ = permuted_ols(
        tested_vars, target_vars, confounding_vars=None, model_intercept=True,
        n_perm=0, random_state=random_state)
    assert_array_almost_equal(ref_scores, own_scores, decimal=6)
    assert_array_almost_equal(own_scores, own_scores_intercept, decimal=6)


def test_permuted_ols_intercept_withcovar_multivariate(random_state=0):
    rng = check_random_state(random_state)
    # design parameters
    n_samples = 50
    n_targets = 10
    n_covars = 2
    # create design
    target_vars = rng.randn(n_samples, n_targets)
    tested_var = np.ones((n_samples, 1))
    confounding_vars = rng.randn(n_samples, n_covars)
    # compute t-scores with linalg or statsmodels
    ref_scores = get_tvalue_with_alternative_library(tested_var, target_vars,
                                                     confounding_vars)
    # permuted OLS
    _, own_scores, _ = permuted_ols(
        tested_var, target_vars, confounding_vars, n_perm=0,
        random_state=random_state)
    # same thing but with model_intercept=True to check it has no effect
    _, own_scores_intercept, _ = permuted_ols(
        tested_var, target_vars, confounding_vars, model_intercept=True,
        n_perm=0, random_state=random_state)
    assert_almost_equal(ref_scores, own_scores, decimal=6)
    assert_array_almost_equal(own_scores, own_scores_intercept, decimal=6)


### Test one-sided versus two-sided ###########################################
def test_sided_test(random_state=0):
    """Check that a positive effect is always better recovered with one-sided.
    """
    rng = check_random_state(random_state)
    # design parameters
    n_samples = 50
    # create design
    target_var = rng.randn(n_samples, 100)
    tested_var = rng.randn(n_samples, 1)
    # permuted OLS
    # one-sided
    neg_log_pvals_onesided, _, _ = permuted_ols(
        tested_var, target_var, model_intercept=False,
        two_sided_test=False, n_perm=100, random_state=random_state)
    # two-sided
    neg_log_pvals_twosided, _, _ = permuted_ols(
        tested_var, target_var, model_intercept=False,
        two_sided_test=True, n_perm=100, random_state=random_state)

    positive_effect_location = neg_log_pvals_onesided > 1
    assert_equal(
        np.sum(neg_log_pvals_twosided[positive_effect_location]
               - neg_log_pvals_onesided[positive_effect_location] > 0),
        0)


def test_sided_test2(random_state=0):
    """Check that two-sided can actually recover positive and negative effects.
    """
    # create design
    target_var1 = np.arange(0, 10).reshape((-1, 1))  # positive effect
    target_var = np.hstack((target_var1, - target_var1))

    tested_var = np.arange(0, 20, 2)
    # permuted OLS
    # one-sided
    neg_log_pvals_onesided, _, _ = permuted_ols(
        tested_var, target_var, model_intercept=False,
        two_sided_test=False, n_perm=100, random_state=random_state)
    # one-sided (other side)
    neg_log_pvals_onesided2, _, _ = permuted_ols(
        tested_var, -target_var, model_intercept=False,
        two_sided_test=False, n_perm=100, random_state=random_state)
    # two-sdided
    neg_log_pvals_twosided, _, _ = permuted_ols(
        tested_var, target_var, model_intercept=False,
        two_sided_test=True, n_perm=100, random_state=random_state)

    assert_array_almost_equal(neg_log_pvals_onesided[0],
                              neg_log_pvals_onesided2[0][::-1])
    assert_array_almost_equal(neg_log_pvals_onesided + neg_log_pvals_onesided2,
                              neg_log_pvals_twosided)
