"""Utility functions for the mass_univariate module's test suite."""

import numpy as np


def get_tvalue_with_alternative_library(tested_vars, target_vars, covars=None):
    """Compute t values with linalg or statsmodels.

    Massively univariate linear model (= each target is considered
    independently).

    Parameters
    ----------
    tested_vars : array-like, shape=(n_samples, n_regressors)
      Tested variates, the associated coefficient of which are to be tested
      independently with a t-test, resulting in as many t-values.

    target_vars : array-like, shape=(n_samples, n_descriptors)
      Target variates, to be approximated with a linear combination of
      the tested variates and the confounding variates.

    covars : array-like, shape=(n_samples, n_confounds)
      Confounding variates, to be fitted but not to be tested

    Returns
    -------
    t-values: np.ndarray, shape=(n_regressors, n_descriptors)

    """
    # set up design
    n_samples, n_regressors = tested_vars.shape
    n_descriptors = target_vars.shape[1]
    if covars is not None:
        n_covars = covars.shape[1]
        design_matrix = np.hstack((tested_vars, covars))
    else:
        n_covars = 0
        design_matrix = tested_vars
    mask_covars = np.ones(n_regressors + n_covars, dtype=bool)
    mask_covars[:n_regressors] = False
    test_matrix = np.array([[1.0] + [0.0] * n_covars])

    # t-values computation
    try:  # try with statsmodels if available (more concise)
        from statsmodels.regression.linear_model import OLS

        t_values = np.empty((n_descriptors, n_regressors))
        for i in range(n_descriptors):
            current_target = target_vars[:, i].reshape((-1, 1))
            for j in range(n_regressors):
                current_tested_mask = mask_covars.copy()
                current_tested_mask[j] = True
                current_design_matrix = design_matrix[:, current_tested_mask]
                ols_fit = OLS(current_target, current_design_matrix).fit()
                t_values[i, j] = np.squeeze(ols_fit.t_test(test_matrix).tvalue)

    except ImportError:  # use linalg if statsmodels is not available
        from numpy import linalg

        lost_dof = n_covars + 1  # fit all tested variates independently
        t_values = np.empty((n_descriptors, n_regressors))
        for i in range(n_regressors):
            current_tested_mask = mask_covars.copy()
            current_tested_mask[i] = True
            current_design_matrix = design_matrix[:, current_tested_mask]
            invcov = linalg.pinv(current_design_matrix)
            normalized_cov = np.dot(invcov, invcov.T)
            t_val_denom_aux = np.diag(
                np.dot(test_matrix, np.dot(normalized_cov, test_matrix.T))
            )
            t_val_denom_aux = t_val_denom_aux.reshape((-1, 1))

            for j in range(n_descriptors):
                current_target = target_vars[:, j].reshape((-1, 1))
                res_lstsq = linalg.lstsq(
                    current_design_matrix, current_target, rcond=-1
                )
                residuals = current_target - np.dot(
                    current_design_matrix, res_lstsq[0]
                )
                t_val_num = np.dot(test_matrix, res_lstsq[0])
                t_val_denom = np.sqrt(
                    np.sum(residuals**2, 0)
                    / float(n_samples - lost_dof)
                    * t_val_denom_aux
                )
                t_values[j, i] = np.squeeze(t_val_num / t_val_denom)

    t_values = t_values.T
    assert t_values.shape == (n_regressors, n_descriptors)
    return t_values
