import numpy as np
from sklearn.utils import check_random_state
from numpy.testing import assert_array_almost_equal

from nilearn.mass_univariate.utils import f_score, orthonormalize_matrix

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
    f_val_own = f_score(var1, var2, normalized_design=False)[0]
    f_val_sklearn, _ = f_regression(var2, np.ravel(var1),
                                    center=False)
    assert_array_almost_equal(f_val_own, f_val_sklearn)

    ### Normalized data
    # generate data
    var1 = np.ones((n_samples, 1)) / np.sqrt(n_samples)
    var2 = rng.randn(n_samples, 1)
    var2 = var2 / np.sqrt(np.sum(var2 ** 2, 0))  # normalize
    f_val_own = f_score(var1, var2, normalized_design=True)[0]
    f_val_own2 = f_score(var1, var2, normalized_design=False)[0]
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
    f_val_own = f_score(var1, var2, covars,
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
    f_val_own = f_score(var1, var2, covars,
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
    f_val_own = f_score(var1, var2, covars,
                         normalized_design=True, lost_dof=3)[0]
    f_val_own2 = f_score(var1, var2, covars,
                          normalized_design=False)[0]
    test_matrix = np.array([[1., 0., 0., 0.]])
    statsmodels_ols = OLS(var2, np.hstack((var1, covars))).fit()
    f_val_statsmodels = statsmodels_ols.f_test(test_matrix).fvalue[0]
    assert_array_almost_equal(f_val_own, f_val_statsmodels)
    assert_array_almost_equal(f_val_own, f_val_own2)
