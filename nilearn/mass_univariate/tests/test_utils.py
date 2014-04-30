import numpy as np
from sklearn.utils import check_random_state
from numpy.testing import assert_array_almost_equal

from nilearn.mass_univariate.utils import (
    orthonormalize_matrix, t_score_with_covars_and_normalized_design,)


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
    try:  # try with statsmodels is available (more concise)
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
    # compute t-scores with nilearn's routine
    t_val_own = t_score_with_covars_and_normalized_design(var1, var2)
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
    # nilearn's t-score
    own_score = t_score_with_covars_and_normalized_design(var1, var2, covars)
    # compute t-scores with linalg or statmodels
    ref_score = get_tvalue_with_alternative_library(var1, var2, covars)
    assert_array_almost_equal(own_score, ref_score)
