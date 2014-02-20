"""
Tests for the permuted_ols function.

"""
# Author: Virgile Fritsch, <virgile.fritsch@inria.fr>, Feb. 2014
import numpy as np
from scipy import stats
from sklearn.utils import check_random_state

from numpy.testing import (assert_almost_equal, assert_raises,
                           assert_array_equal, assert_array_almost_equal,
                           assert_array_less, assert_warns)

from nilearn.mass_univariate import permuted_ols
from nilearn.mass_univariate.permuted_least_squares import (
    GrowableSparseArray, _f_score, orthonormalize_matrix)

from nilearn._utils.fixes import f_regression


## Tests for the GrowableSparseArray class used in permuted_ols function. #####
def test_gsarray_append_data():
    """This function tests GrowableSparseArray creation and filling.

    It is especially important to check that the threshold is respected
    and that the structure is robust to threshold choice.

    """
    # Simplest example
    gsarray = GrowableSparseArray(n_iter=1, threshold=0)
    gsarray.append(0, np.ones((5, 1)))
    assert_array_equal(gsarray.get_data()['iter_id'], np.zeros(5))
    assert_array_equal(gsarray.get_data()['x_id'], np.zeros(5))
    assert_array_equal(gsarray.get_data()['y_id'], np.arange(5))
    assert_array_equal(gsarray.get_data()['score'], np.ones(5))

    # Empty array
    gsarray = GrowableSparseArray(n_iter=1, threshold=10)
    gsarray.append(0, np.ones((5, 1)))
    assert_array_equal(gsarray.get_data()['iter_id'], [])
    assert_array_equal(gsarray.get_data()['x_id'], [])
    assert_array_equal(gsarray.get_data()['y_id'], [])
    assert_array_equal(gsarray.get_data()['score'], [])

    # Toy example
    gsarray = GrowableSparseArray(n_iter=10, threshold=8)
    for i in range(10):
        gsarray.append(i, (np.arange(10) - i).reshape((-1, 1)))
    assert_array_equal(gsarray.get_data()['iter_id'], np.array([0., 0., 1.]))
    assert_array_equal(gsarray.get_data()['x_id'], np.zeros(3))
    assert_array_equal(gsarray.get_data()['y_id'], [8, 9, 9])
    assert_array_equal(gsarray.get_data()['score'], [8., 9., 8.])


def test_gsarray_merge():
    """This function tests GrowableSparseArray merging.

    Because of the specific usage of GrowableSparseArrays, only a reduced
    number of manipulations has been implemented.

    """
    # Basic merge
    gsarray = GrowableSparseArray(n_iter=1, threshold=0)
    gsarray.append(0, np.ones((5, 1)))
    gsarray2 = GrowableSparseArray(n_iter=1, threshold=0)
    gsarray2.merge(gsarray)
    assert_array_equal(gsarray.get_data()['iter_id'],
                       gsarray2.get_data()['iter_id'])
    assert_array_equal(gsarray.get_data()['x_id'],
                       gsarray2.get_data()['x_id'])
    assert_array_equal(gsarray.get_data()['y_id'],
                       gsarray2.get_data()['y_id'])
    assert_array_equal(gsarray.get_data()['score'],
                       gsarray2.get_data()['score'])

    # Merge list
    gsarray = GrowableSparseArray(n_iter=2, threshold=0)
    gsarray.append(0, np.ones((5, 1)))
    gsarray2 = GrowableSparseArray(n_iter=2, threshold=0)
    gsarray2.append(1, 2 * np.ones((5, 1)), y_offset=5)
    gsarray3 = GrowableSparseArray(n_iter=2, threshold=0)
    gsarray3.merge([gsarray, gsarray2])
    assert_array_equal(gsarray3.get_data()['iter_id'],
                       np.array([0.] * 5 + [1.] * 5))
    assert_array_equal(gsarray3.get_data()['x_id'], np.zeros(10))
    assert_array_equal(gsarray3.get_data()['y_id'], np.arange(10))
    assert_array_equal(gsarray3.get_data()['score'],
                       np.array([1.] * 5 + [2.] * 5))

    # Test failure case (merging arrays with different n_iter)
    gsarray_wrong = GrowableSparseArray(n_iter=1)
    gsarray_wrong.append(0, np.ones((5, 1)))
    gsarray = GrowableSparseArray(n_iter=2)
    assert_raises(Exception, gsarray.merge, gsarray_wrong)

    # Test failure case (merge a numpy array)
    gsarray = GrowableSparseArray(n_iter=1)
    assert_raises(Exception, gsarray.merge, np.ones(5))

    # Check the threshold is respected when merging
    # merging a gsarray into another one that has a higher threhold
    # (nothing should be left in the parent array)
    gsarray = GrowableSparseArray(n_iter=1, threshold=0)
    gsarray.append(0, np.ones((5, 1)))
    gsarray2 = GrowableSparseArray(n_iter=1, threshold=2)  # higher threshold
    gsarray2.merge(gsarray)
    assert_array_equal(gsarray2.get_data()['score'], [])

    # merging a gsarray into another one that has a higher threhold
    # (should raises a warning on potential information loss)
    gsarray = GrowableSparseArray(n_iter=1, threshold=1)
    gsarray.append(0, np.ones((5, 1)))
    gsarray2 = GrowableSparseArray(n_iter=1, threshold=0)  # lower threshold
    assert_warns(UserWarning, gsarray2.merge, gsarray)
    assert_array_equal(gsarray.get_data()['iter_id'],
                       gsarray2.get_data()['iter_id'])
    assert_array_equal(gsarray.get_data()['x_id'],
                       gsarray2.get_data()['x_id'])
    assert_array_equal(gsarray.get_data()['y_id'],
                       gsarray2.get_data()['y_id'])
    assert_array_equal(gsarray.get_data()['score'],
                       gsarray2.get_data()['score'])


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
def test_permuted_ols_check_h0(random_state=0):
    rng = check_random_state(random_state)
    # design parameters
    n_samples = 50
    # create dummy design with no effect
    target_var = rng.randn(n_samples, 1)
    tested_var = np.arange(n_samples).reshape((-1, 1))
    # permuted OLS (sparsity_threshold=1. to get all values)
    # We check that h0 is close to the theoretical distribution, which is
    # known for this simple design (= F(1, 1 - n_samples)).
    # We use the Mean Squared Error (MSE) between cdf for that purpose.
    perm_ranges = [10, 100, 1000]  # test various number of permutations
    all_mse = []
    for i, n_perm in enumerate(np.repeat(perm_ranges, 10)):
        pval, orig_scores, h0, _ = permuted_ols(
            tested_var, target_var, model_intercept=False,
            n_perm=n_perm, sparsity_threshold=1., random_state=i)
        assert_array_less(pval, 1.)  # pval should not be significant
        # comparing h0 cumulative density function to F(1, n_samples - 1).cdf()
        # (we consider only one target so in each permutation, the max is equal
        #  to the single available value)
        mse = np.mean(
            (stats.f(1, n_samples - 1).cdf(np.sort(h0))
             - np.linspace(0, 1, h0.size)) ** 2)
        all_mse.append(mse)
    all_mse = np.array(all_mse).reshape((len(perm_ranges), -1))
    # for a given n_perm, check that we have a mse below a specific threshold
    assert_array_less(
        all_mse - np.array([0.1, 0.05, 0.005]).reshape((-1, 1)), 0)
    # consistency of the algorithm: the more permutations, the less the mse
    assert_array_less(np.diff(all_mse.mean(1)), 0)

    # create design with strong effect
    target_var = np.arange(n_samples, dtype=float).reshape((-1, 1))
    target_var += rng.randn(n_samples, 1)
    tested_var = np.arange(n_samples).reshape((-1, 1))
    # permuted OLS (sparsity_threshold=1. to get all values)
    n_perm = 1000
    pval, orig_scores, h0, _ = permuted_ols(
        tested_var, target_var, model_intercept=False,
        n_perm=n_perm, sparsity_threshold=1., random_state=random_state)
    assert_array_equal(pval, np.log10(n_perm + 1))  # pval should be large


def test_permuted_ols_intercept_check_h0(random_state=0):
    rng = check_random_state(random_state)
    # design parameters
    n_samples = 50
    # create dummy design with no effect
    target_var = rng.randn(n_samples, 1)
    tested_var = np.ones((n_samples, 1))
    # permuted OLS (sparsity_threshold=1. to get all values)
    # We check that h0 is close to the theoretical distribution, which is
    # known for this simple design (= F(1, 1 - n_samples)).
    # We use the Mean Squared Error (MSE) between cdf for that purpose.
    perm_ranges = [10, 100, 1000]  # test various number of permutations
    all_mse = []
    for i, n_perm in enumerate(np.repeat(perm_ranges, 10)):
        pval, orig_scores, h0, _ = permuted_ols(
            tested_var, target_var, model_intercept=False,
            n_perm=n_perm, sparsity_threshold=1., random_state=i)
        assert_array_less(pval, 1.)  # pval should not be significant
        # comparing h0 cumulative density function to F(1, n_samples - 1).cdf()
        # (we consider only one target so in each permutation, the max is equal
        #  to the single available value)
        mse = np.mean(
            (stats.f(1, n_samples - 1).cdf(np.sort(h0))
             - np.linspace(0, 1, h0.size)) ** 2)
        all_mse.append(mse)
    all_mse = np.array(all_mse).reshape((len(perm_ranges), -1))
    # for a given n_perm, check that we have a mse below a specific threshold
    assert_array_less(
        all_mse - np.array([0.1, 0.01, 0.001]).reshape((-1, 1)), 0)
    # consistency of the algorithm: the more permutations, the less the mse
    assert_array_less(np.diff(all_mse.mean(1)), 0)

    # create design with strong effect
    target_var = np.ones((n_samples, 1)) + rng.randn(n_samples, 1)
    tested_var = np.ones((n_samples, 1))
    # permuted OLS (sparsity_threshold=1. to get all values)
    n_perm = 1000
    pval, orig_scores, h0, _ = permuted_ols(
        tested_var, target_var, model_intercept=False,
        n_perm=n_perm, sparsity_threshold=1., random_state=random_state)
    assert_array_equal(pval, np.log10(n_perm + 1))  # pval should be large


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
    _, orig_scores, _, _ = permuted_ols(
        tested_var, target_var, model_intercept=False,
        n_perm=0, sparsity_threshold=1., random_state=random_state)
    assert_array_almost_equal(fvals, orig_scores['score'], decimal=6)

    # test with ravelized tested_var
    _, orig_scores, _, _ = permuted_ols(
        np.ravel(tested_var), target_var, model_intercept=False,
        n_perm=0, sparsity_threshold=1., random_state=random_state)
    assert_array_almost_equal(fvals, orig_scores['score'], decimal=6)

    ### Adds intercept (should be equivalent to centering variates)
    # permuted OLS (sparsity_threshold=1. to get all values)
    _, orig_scores_addintercept, _, _ = permuted_ols(
        tested_var, target_var, model_intercept=True,
        n_perm=0, sparsity_threshold=1., random_state=random_state)
    target_var -= target_var.mean(0)
    tested_var -= tested_var.mean(0)
    # scikit-learn F-score
    fvals_addintercept, _ = f_regression(target_var, tested_var, center=True)
    assert_array_almost_equal(
        fvals_addintercept, orig_scores_addintercept['score'], decimal=6)


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
    _, orig_scores, _, _ = permuted_ols(
        tested_var, target_var, confounding_vars, model_intercept=False,
        n_perm=0, sparsity_threshold=1., random_state=random_state)
    assert_array_almost_equal(fvals, orig_scores['score'], decimal=6)

    ### Adds intercept
    # permuted OLS (sparsity_threshold=1. to get all values)
    _, orig_scores_addintercept, _, _ = permuted_ols(
        tested_var, target_var, confounding_vars, model_intercept=True,
        n_perm=0, sparsity_threshold=1., random_state=random_state)
    # statsmodels OLS
    confounding_vars = np.hstack((confounding_vars, np.ones((n_samples, 1))))
    ols = OLS(target_var, np.hstack((tested_var, confounding_vars))).fit()
    fvals_addintercept = ols.f_test([[1., 0., 0., 0.]]).fvalue[0]
    assert_array_almost_equal(
        fvals_addintercept, orig_scores_addintercept['score'], decimal=6)


def test_permuted_ols_sklearn_nocovar_multivariate(random_state=0):
    rng = check_random_state(random_state)
    # design parameters
    n_samples = 50
    n_targets = 10
    # create design
    target_vars = rng.randn(n_samples, n_targets)
    tested_var = rng.randn(n_samples, 1)
    # scikit-learn F-scores
    fvals = np.empty(n_targets)
    for i in range(n_targets):
        fvals[i], _ = f_regression(target_vars[:, i], tested_var, center=False)
    # permuted OLS (sparsity_threshold=1. to get all values)
    _, orig_scores, _, _ = permuted_ols(
        tested_var, target_vars, model_intercept=False,
        n_perm=0, sparsity_threshold=1., random_state=random_state)
    assert_array_almost_equal(fvals, orig_scores['score'], decimal=6)

    ### Adds intercept (should be equivalent to centering variates)
    # permuted OLS (sparsity_threshold=1. to get all values)
    _, orig_scores_addintercept, _, _ = permuted_ols(
        tested_var, target_vars, model_intercept=True,
        n_perm=0, sparsity_threshold=1., random_state=random_state)
    target_vars -= target_vars.mean(0)
    tested_var -= tested_var.mean(0)
    # scikit-learn F-score
    fvals_addintercept = np.empty(n_targets)
    for i in range(n_targets):
        fvals_addintercept[i], _ = f_regression(
            target_vars[:, i], tested_var, center=True)
    assert_array_almost_equal(
        fvals_addintercept, orig_scores_addintercept['score'], decimal=6)


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
    _, orig_scores, _, _ = permuted_ols(
        tested_var, target_vars, confounding_vars, model_intercept=False,
        n_perm=0, sparsity_threshold=1., random_state=random_state)
    assert_almost_equal(fvals, orig_scores['score'], decimal=6)

    ### Adds intercept
    # permuted OLS (sparsity_threshold=1. to get all values)
    _, orig_scores_addintercept, _, _ = permuted_ols(
        tested_var, target_vars, confounding_vars, model_intercept=True,
        n_perm=0, sparsity_threshold=1., random_state=random_state)
    # statsmodels OLS
    confounding_vars = np.hstack((confounding_vars, np.ones((n_samples, 1))))
    fvals_addintercept = np.empty(n_targets)
    test_matrix = np.array([[1.] + [0.] * (n_covars + 1)])
    for i in range(n_targets):
        ols = OLS(
            target_vars[:, i], np.hstack((tested_var, confounding_vars)))
        fvals_addintercept[i] = ols.fit().f_test(test_matrix).fvalue[0][0]
    assert_array_almost_equal(
        fvals_addintercept, orig_scores_addintercept['score'], decimal=6)


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
    _, orig_scores, _, _ = permuted_ols(
        tested_var, target_var, confounding_vars=None, n_perm=0,
        sparsity_threshold=1., random_state=random_state)
    # same thing but with model_intercept=True to check it has no effect
    _, orig_scores_addintercept, _, _ = permuted_ols(
        tested_var, target_var, confounding_vars=None, model_intercept=True,
        n_perm=0, sparsity_threshold=1., random_state=random_state)
    assert_array_almost_equal(fvals, orig_scores['score'], decimal=6)
    assert_array_almost_equal(orig_scores['score'],
                              orig_scores_addintercept['score'], decimal=6)


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
    _, orig_scores, _, _ = permuted_ols(
        tested_var, target_var, confounding_vars, n_perm=0,
        sparsity_threshold=1., random_state=random_state)
    # same thing but with model_intercept=True to check it has no effect
    _, orig_scores_addintercept, _, _ = permuted_ols(
        tested_var, target_var, confounding_vars, model_intercept=True,
        n_perm=0, sparsity_threshold=1., random_state=random_state)
    assert_array_almost_equal(fvals, orig_scores['score'], decimal=6)
    assert_array_almost_equal(orig_scores['score'],
                              orig_scores_addintercept['score'], decimal=6)


def test_permuted_ols_intercept_sklearn_nocovar_multivariate(random_state=0):
    rng = check_random_state(random_state)
    # design parameters
    n_samples = 50
    n_targets = 10
    # create design
    target_vars = rng.randn(n_samples, n_targets)
    tested_var = np.ones((n_samples, 1))
    # scikit-learn F-scores
    fvals = np.empty(n_targets)
    for i in range(n_targets):
        fvals[i], _ = f_regression(target_vars[:, i], tested_var, center=False)
    # permuted OLS (sparsity_threshold=1. to get all values)
    _, orig_scores, _, _ = permuted_ols(
        tested_var, target_vars, confounding_vars=None, n_perm=0,
        sparsity_threshold=1., random_state=random_state)
    # same thing but with model_intercept=True to check it has no effect
    _, orig_scores_addintercept, _, _ = permuted_ols(
        tested_var, target_vars, confounding_vars=None, model_intercept=True,
        n_perm=0, sparsity_threshold=1., random_state=random_state)
    assert_array_almost_equal(fvals, orig_scores['score'], decimal=6)
    assert_array_almost_equal(orig_scores['score'],
                              orig_scores_addintercept['score'], decimal=6)


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
    _, orig_scores, _, _ = permuted_ols(
        tested_var, target_vars, confounding_vars, n_perm=0,
        sparsity_threshold=1., random_state=random_state)
    # same thing but with model_intercept=True to check it has no effect
    _, orig_scores_addintercept, _, _ = permuted_ols(
        tested_var, target_vars, confounding_vars, model_intercept=True,
        n_perm=0, sparsity_threshold=1., random_state=random_state)
    assert_almost_equal(fvals, orig_scores['score'], decimal=6)
    assert_array_almost_equal(orig_scores['score'],
                              orig_scores_addintercept['score'], decimal=6)


if __name__ == '__main__':
    import nose
    nose.run(argv=['', __file__])
