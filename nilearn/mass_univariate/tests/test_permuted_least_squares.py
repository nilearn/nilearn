"""Tests for the permuted_ols function."""
# Author: Virgile Fritsch, <virgile.fritsch@inria.fr>, Feb. 2014
import numpy as np
import pytest
from nilearn.mass_univariate import permuted_ols
from nilearn.mass_univariate.tests.utils import (
    get_tvalue_with_alternative_library,
)
from numpy.testing import (
    assert_array_almost_equal,
    assert_array_less,
    assert_equal,
)
from scipy import stats
from sklearn.utils import check_random_state


# General tests for permuted_ols function
def test_permuted_ols_check_h0_noeffect_labelswap(random_state=0):
    rng = check_random_state(random_state)

    # design parameters
    n_samples = 100
    n_regressors = 1

    # create dummy design with no effect
    target_var = rng.randn(n_samples, 1)
    tested_var = np.arange(n_samples, dtype="f8").reshape((-1, 1))
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
        # Case no. 1: no intercept in the model
        pval, orig_scores, h0 = permuted_ols(
            tested_var,
            target_var,
            model_intercept=False,
            n_perm=n_perm,
            two_sided_test=False,
            random_state=i,
        )
        assert_equal(h0.shape, (n_regressors, n_perm))
        h0_intercept = h0[0, :]

        # Kolmogorov-Smirnov test
        kstest_pval = stats.kstest(h0_intercept, stats.t(n_samples - 1).cdf)[1]
        all_kstest_pvals.append(kstest_pval)
        mse = np.mean(
            (
                stats.t(n_samples - 1).cdf(np.sort(h0_intercept))
                - np.linspace(0, 1, h0_intercept.size + 1)[1:]
            )
            ** 2
        )
        all_mse.append(mse)

        # Case no. 2: intercept in the model
        pval, orig_scores, h0 = permuted_ols(
            tested_var,
            target_var,
            model_intercept=True,
            n_perm=n_perm,
            two_sided_test=False,
            random_state=i,
        )
        assert_array_less(pval, 1.0)  # pval should not be significant
        h0_intercept = h0[0, :]

        # Kolmogorov-Smirnov test
        kstest_pval = stats.kstest(h0_intercept, stats.t(n_samples - 2).cdf)[1]
        all_kstest_pvals_intercept.append(kstest_pval)
        mse = np.mean(
            (
                stats.t(n_samples - 2).cdf(np.sort(h0_intercept))
                - np.linspace(0, 1, h0_intercept.size + 1)[1:]
            )
            ** 2
        )
        all_mse_intercept.append(mse)

        # Case no. 3: intercept in the model, no centering of tested vars
        pval, orig_scores, h0 = permuted_ols(
            tested_var_not_centered,
            target_var,
            model_intercept=True,
            n_perm=n_perm,
            two_sided_test=False,
            random_state=i,
        )
        assert_array_less(pval, 1.0)  # pval should not be significant
        h0_intercept = h0[0, :]

        # Kolmogorov-Smirnov test
        kstest_pval = stats.kstest(h0_intercept, stats.t(n_samples - 2).cdf)[1]
        all_kstest_pvals_intercept2.append(kstest_pval)
        mse = np.mean(
            (
                stats.t(n_samples - 2).cdf(np.sort(h0_intercept))
                - np.linspace(0, 1, h0_intercept.size + 1)[1:]
            )
            ** 2
        )
        all_mse_intercept2.append(mse)

    all_kstest_pvals = np.array(all_kstest_pvals).reshape(
        (len(perm_ranges), -1)
    )
    all_kstest_pvals_intercept = np.array(all_kstest_pvals_intercept).reshape(
        (len(perm_ranges), -1)
    )
    all_mse = np.array(all_mse).reshape((len(perm_ranges), -1))
    all_mse_intercept = np.array(all_mse_intercept).reshape(
        (len(perm_ranges), -1)
    )
    all_mse_intercept2 = np.array(all_mse_intercept2).reshape(
        (len(perm_ranges), -1)
    )

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
    n_regressors = 1

    # create dummy design with no effect
    target_var = rng.randn(n_samples, 1)
    tested_var = np.ones((n_samples, n_regressors))

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
            tested_var,
            target_var,
            model_intercept=False,
            n_perm=n_perm,
            two_sided_test=False,
            random_state=i,
        )
        assert h0.shape == (n_regressors, n_perm)
        h0_intercept = h0[0, :]

        # Kolmogorov-Smirnov test
        kstest_pval = stats.kstest(h0_intercept, stats.t(n_samples).cdf)[1]
        all_kstest_pvals.append(kstest_pval)
        mse = np.mean(
            (
                stats.t(n_samples).cdf(np.sort(h0_intercept))
                - np.linspace(0, 1, h0_intercept.size + 1)[1:]
            )
            ** 2
        )
        all_mse.append(mse)

    all_kstest_pvals = np.array(all_kstest_pvals).reshape(
        (len(perm_ranges), -1)
    )
    all_mse = np.array(all_mse).reshape((len(perm_ranges), -1))

    # check that a difference between distributions is not rejected by KS test
    assert_array_less(0.01 / (len(perm_ranges) * 10.0), all_kstest_pvals)

    # consistency of the algorithm: the more permutations, the less the MSE
    assert_array_less(np.diff(all_mse.mean(1)), 0)


# Tests for labels swapping permutation scheme
def test_permuted_ols_nocovar(random_state=0):
    rng = check_random_state(random_state)

    # design parameters
    n_samples = 50
    n_descriptors = 1
    n_regressors = 1

    # create design
    target_var = rng.randn(n_samples, n_descriptors)
    tested_var = rng.randn(n_samples, n_regressors)

    # compute t-scores with linalg or statsmodels
    ref_score = get_tvalue_with_alternative_library(tested_var, target_var)

    # permuted OLS
    _, own_score, _ = permuted_ols(
        tested_var,
        target_var,
        model_intercept=False,
        n_perm=0,
        random_state=random_state,
    )
    assert_array_almost_equal(ref_score, own_score, decimal=6)

    # test with ravelized tested_var
    _, own_score, _ = permuted_ols(
        np.ravel(tested_var),
        target_var,
        model_intercept=False,
        n_perm=0,
        random_state=random_state,
    )
    assert_array_almost_equal(ref_score, own_score, decimal=6)

    # Adds intercept (should be equivalent to centering variates)
    # permuted OLS
    _, own_score_intercept, _ = permuted_ols(
        tested_var,
        target_var,
        model_intercept=True,
        n_perm=0,
        random_state=random_state,
    )
    target_var -= target_var.mean(0)
    tested_var -= tested_var.mean(0)

    # compute t-scores with linalg or statsmodels
    ref_score_intercept = get_tvalue_with_alternative_library(
        tested_var, target_var, np.ones((n_samples, 1))
    )
    assert_array_almost_equal(
        ref_score_intercept, own_score_intercept, decimal=6
    )


def test_permuted_ols_nocovar_warning(random_state=0):
    """Ensure that a warning is raised when a given voxel has all zeros.

    This test also checks that an invalid n_jobs value will raise a ValueError.
    """
    rng = check_random_state(random_state)

    # design parameters
    n_samples = 50
    n_descriptors = 10
    n_regressors = 1

    # create design
    target_var = rng.randn(n_samples, n_descriptors)
    tested_var = rng.randn(n_samples, n_regressors)

    # permuted OLS
    _, own_score, _ = permuted_ols(
        tested_var,
        target_var,
        model_intercept=False,
        n_perm=100,
        random_state=random_state,
    )

    # test with ravelized tested_var
    target_var[:, 0] = 0

    with pytest.warns(UserWarning):
        _, own_score2, _ = permuted_ols(
            np.ravel(tested_var),
            target_var,
            model_intercept=False,
            n_perm=100,
            random_state=random_state,
        )

    assert np.array_equal(own_score[1:], own_score2[1:])

    # Ensure that passing an unacceptable n_jobs value will raise a ValueError
    with pytest.raises(ValueError):
        permuted_ols(
            tested_var,
            target_var,
            model_intercept=False,
            n_perm=100,
            n_jobs=0,  # not allowed
            random_state=random_state,
        )


def test_permuted_ols_withcovar(random_state=0):
    rng = check_random_state(random_state)

    # design parameters
    n_samples = 50
    n_descriptors = 1
    n_regressors = 1
    n_covars = 2

    # create design
    target_var = rng.randn(n_samples, n_descriptors)
    tested_var = rng.randn(n_samples, n_regressors)
    confounding_vars = rng.randn(n_samples, n_covars)

    # compute t-scores with linalg or statsmodels
    ref_score = get_tvalue_with_alternative_library(
        tested_var, target_var, confounding_vars
    )
    assert ref_score.shape == (n_regressors, n_descriptors)

    # permuted OLS
    _, own_score, _ = permuted_ols(
        tested_var,
        target_var,
        confounding_vars,
        model_intercept=False,
        n_perm=0,
        random_state=random_state,
    )
    assert own_score.shape == (n_regressors, n_descriptors)
    assert_array_almost_equal(ref_score, own_score, decimal=6)

    # Adds intercept
    # permuted OLS
    _, own_score_intercept, _ = permuted_ols(
        tested_var,
        target_var,
        confounding_vars,
        model_intercept=True,
        n_perm=0,
        random_state=random_state,
    )
    assert own_score_intercept.shape == (n_regressors, n_descriptors)

    # compute t-scores with linalg or statsmodels
    confounding_vars = np.hstack((confounding_vars, np.ones((n_samples, 1))))
    alt_score_intercept = get_tvalue_with_alternative_library(
        tested_var, target_var, confounding_vars
    )
    assert alt_score_intercept.shape == (n_regressors, n_descriptors)
    assert_array_almost_equal(
        alt_score_intercept, own_score_intercept, decimal=6
    )

    # Intercept in confounding vars
    # permuted OLS with constant in confounding_vars, model_intercept=True
    confounding_vars = np.ones([n_samples, 1])
    _, own_score, _ = permuted_ols(
        tested_var,
        target_var,
        confounding_vars,
        model_intercept=True,
        n_perm=0,
        random_state=random_state,
    )
    assert own_score.shape == (n_regressors, n_descriptors)

    # permuted OLS with constant in confounding_vars, model_intercept=False
    confounding_vars = np.ones([n_samples, 1])
    _, own_score, _ = permuted_ols(
        tested_var,
        target_var,
        confounding_vars,
        model_intercept=False,
        n_perm=0,
        random_state=random_state,
    )
    assert own_score.shape == (n_regressors, n_descriptors)

    # permuted OLS, multiple constants and covars, model_intercept=False
    confounding_vars = np.hstack(
        (rng.randn(n_samples, n_covars), np.ones([n_samples, 2]))
    )
    _, own_score, _ = permuted_ols(
        tested_var,
        target_var,
        confounding_vars,
        model_intercept=False,
        n_perm=0,
        random_state=random_state,
    )
    assert own_score.shape == (n_regressors, n_descriptors)

    # Multiple intercepts should raise a warning
    # In confounding vars
    with pytest.warns(UserWarning):
        confounding_vars = np.ones([n_samples, 2])
        _, own_score, _ = permuted_ols(
            tested_var,
            target_var,
            confounding_vars,
            n_perm=0,
            random_state=random_state,
        )

    # Across tested vars and confounding vars
    with pytest.warns(UserWarning):
        confounding_vars = np.ones([n_samples, 1])
        tested_var = np.ones([n_samples, 1])
        _, own_score, _ = permuted_ols(
            tested_var,
            target_var,
            confounding_vars,
            n_perm=0,
            random_state=random_state,
        )


def test_permuted_ols_nocovar_multivariate(random_state=0):
    """Test permuted_ols with multiple tested variates and no covariate.

    It is equivalent to fitting several models with only one tested variate.

    """
    rng = check_random_state(random_state)

    # design parameters
    n_samples = 50
    n_descriptors = 10
    n_regressors = 2
    n_perm = 10

    # create design
    target_vars = rng.randn(n_samples, n_descriptors)
    tested_var = rng.randn(n_samples, n_regressors)

    # compute t-scores with linalg or statsmodels
    ref_scores = get_tvalue_with_alternative_library(tested_var, target_vars)

    # permuted OLS
    neg_log10_pvals, own_scores, h0_fmax = permuted_ols(
        tested_var,
        target_vars,
        model_intercept=False,
        n_perm=n_perm,
        random_state=random_state,
    )
    assert_array_almost_equal(ref_scores, own_scores, decimal=6)
    assert neg_log10_pvals.shape == (n_regressors, n_descriptors)
    assert h0_fmax.shape == (n_regressors, n_perm)

    # Adds intercept (should be equivalent to centering variates)
    # permuted OLS
    _, own_scores_intercept, _ = permuted_ols(
        tested_var,
        target_vars,
        model_intercept=True,
        n_perm=0,
        random_state=random_state,
    )
    target_vars -= target_vars.mean(0)
    tested_var -= tested_var.mean(0)

    # compute t-scores with linalg or statsmodels
    ref_scores_intercept = get_tvalue_with_alternative_library(
        tested_var, target_vars, np.ones((n_samples, 1))
    )
    assert_array_almost_equal(
        ref_scores_intercept, own_scores_intercept, decimal=6
    )


# Tests for sign swapping permutation scheme


def test_permuted_ols_intercept_nocovar(random_state=0):
    rng = check_random_state(random_state)

    # design parameters
    n_samples = 50
    n_descriptors = 10
    n_regressors = 1

    # create design
    tested_var = np.ones((n_samples, n_regressors))
    target_var = rng.randn(n_samples, n_descriptors)

    # compute t-scores with linalg or statmodels
    t_val_ref = get_tvalue_with_alternative_library(tested_var, target_var)
    assert t_val_ref.shape == (n_regressors, n_descriptors)

    # permuted OLS
    neg_log_pvals, orig_scores, _ = permuted_ols(
        tested_var,
        target_var,
        confounding_vars=None,
        n_perm=10,
        random_state=random_state,
    )
    assert neg_log_pvals.shape == (n_regressors, n_descriptors)
    assert orig_scores.shape == (n_regressors, n_descriptors)
    assert_array_less(neg_log_pvals, 1.0)  # ensure sign swap is correctly done
    assert_array_almost_equal(t_val_ref, orig_scores, decimal=6)

    # same thing but with model_intercept=True to check it has no effect
    _, orig_scores_addintercept, _ = permuted_ols(
        tested_var,
        target_var,
        confounding_vars=None,
        model_intercept=False,
        n_perm=0,
        random_state=random_state,
    )
    assert orig_scores_addintercept.shape == (n_regressors, n_descriptors)
    assert_array_almost_equal(t_val_ref, orig_scores_addintercept, decimal=6)


def test_permuted_ols_intercept_statsmodels_withcovar(random_state=0):
    rng = check_random_state(random_state)

    # design parameters
    n_samples = 50
    n_descriptors = 10
    n_regressors = 1
    n_covars = 2

    # create design
    tested_var = np.ones((n_samples, n_regressors))
    target_var = rng.randn(n_samples, n_descriptors)
    confounding_vars = rng.randn(n_samples, n_covars)

    # compute t-scores with linalg or statmodels
    ref_scores = get_tvalue_with_alternative_library(
        tested_var, target_var, confounding_vars
    )
    assert ref_scores.shape == (n_regressors, n_descriptors)

    # permuted OLS
    _, own_scores, _ = permuted_ols(
        tested_var,
        target_var,
        confounding_vars,
        n_perm=0,
        random_state=random_state,
    )
    assert own_scores.shape == (n_regressors, n_descriptors)
    assert_array_almost_equal(ref_scores, own_scores, decimal=6)

    # same thing but with model_intercept=True to check it has no effect
    _, own_scores_intercept, _ = permuted_ols(
        tested_var,
        target_var,
        confounding_vars,
        model_intercept=True,
        n_perm=0,
        random_state=random_state,
    )
    assert own_scores_intercept.shape == (n_regressors, n_descriptors)
    assert_array_almost_equal(ref_scores, own_scores_intercept, decimal=6)


# Test one-sided versus two-sided
def test_sided_test(random_state=0):
    """Check that a positive effect is always better \
    recovered with one-sided."""
    rng = check_random_state(random_state)

    # design parameters
    n_samples = 50
    n_descriptors = 100
    n_regressors = 1

    # create design
    target_var = rng.randn(n_samples, n_descriptors)
    tested_var = rng.randn(n_samples, n_regressors)

    # permuted OLS
    # one-sided
    neg_log_pvals_onesided, _, _ = permuted_ols(
        tested_var,
        target_var,
        model_intercept=False,
        two_sided_test=False,
        n_perm=100,
        random_state=random_state,
    )
    assert neg_log_pvals_onesided.shape == (n_regressors, n_descriptors)

    # two-sided
    neg_log_pvals_twosided, _, _ = permuted_ols(
        tested_var,
        target_var,
        model_intercept=False,
        two_sided_test=True,
        n_perm=100,
        random_state=random_state,
        verbose=1,
    )
    assert neg_log_pvals_twosided.shape == (n_regressors, n_descriptors)

    positive_effect_location = neg_log_pvals_onesided > 1
    assert_equal(
        np.sum(
            neg_log_pvals_twosided[positive_effect_location]
            - neg_log_pvals_onesided[positive_effect_location]
            > 0
        ),
        0,
    )


def test_sided_test2(random_state=0):
    """Check that two-sided can actually recover \
    positive and negative effects."""
    # create design
    target_var1 = np.arange(0, 10).reshape((-1, 1))  # positive effect
    target_var = np.hstack((target_var1, -target_var1))
    tested_var = np.arange(0, 20, 2)

    # permuted OLS
    # one-sided
    neg_log_pvals_onesided, _, _ = permuted_ols(
        tested_var,
        target_var,
        model_intercept=False,
        two_sided_test=False,
        n_perm=100,
        random_state=random_state,
    )

    # one-sided (other side)
    neg_log_pvals_onesided2, _, _ = permuted_ols(
        tested_var,
        -target_var,
        model_intercept=False,
        two_sided_test=False,
        n_perm=100,
        random_state=random_state,
    )

    # two-sided
    neg_log_pvals_twosided, _, _ = permuted_ols(
        tested_var,
        target_var,
        model_intercept=False,
        two_sided_test=True,
        n_perm=100,
        random_state=random_state,
    )

    assert_array_almost_equal(
        neg_log_pvals_onesided[0], neg_log_pvals_onesided2[0][::-1]
    )
    assert_array_almost_equal(
        neg_log_pvals_onesided + neg_log_pvals_onesided2,
        neg_log_pvals_twosided,
    )


def test_tfce_smoke(random_state=0):
    """Test combinations of parameters related to TFCE inference."""
    import nibabel as nib
    from nilearn.maskers import NiftiMasker

    # create design
    target_var1 = np.arange(0, 10).reshape((-1, 1))  # positive effect
    target_var = np.hstack(
        (  # corresponds to 3 x 3 x 3 x 10 niimg
            target_var1,  # voxel 1 has positive effect
            -target_var1,  # voxel 2 has negative effect
            np.random.random((10, 25)),  # 25 remaining voxels
        )
    )
    tested_var = np.arange(0, 20, 2)

    mask_img = nib.Nifti1Image(np.ones((3, 3, 3)), np.eye(4))
    masker = NiftiMasker(mask_img)
    masker.fit(mask_img)
    n_descriptors = np.prod(mask_img.shape)
    n_regressors = 1  # tested_var is 1D

    # tfce is True, indicating TFCE inference should be done,
    # but masker is not defined.
    with pytest.raises(ValueError):
        permuted_ols(
            tested_var,
            target_var,
            model_intercept=False,
            two_sided_test=False,
            n_perm=100,
            random_state=random_state,
            threshold=None,
            masker=None,
            tfce=True,
        )

    # tfce is True, but output_type is "legacy".
    # raise a warning, and get a dictionary.
    with pytest.warns(UserWarning, match="Overriding."):
        out = permuted_ols(
            tested_var,
            target_var,
            model_intercept=False,
            two_sided_test=False,
            n_perm=0,
            random_state=random_state,
            threshold=None,
            masker=masker,
            tfce=True,
            output_type="legacy",
        )

    assert isinstance(out, dict)

    # output_type is "legacy".
    # raise a deprecation warning, but get the standard output.
    with pytest.warns(DeprecationWarning):
        out = permuted_ols(
            tested_var,
            target_var,
            model_intercept=False,
            two_sided_test=False,
            n_perm=100,
            random_state=random_state,
            threshold=None,
            masker=None,
            tfce=False,
            output_type="legacy",
        )

    assert isinstance(out, tuple)

    # no permutations and output_type is "dict", so check for "t" and
    # "tfce" maps
    out = permuted_ols(
        tested_var,
        target_var,
        model_intercept=False,
        two_sided_test=False,
        n_perm=0,
        random_state=random_state,
        threshold=None,
        masker=masker,
        tfce=True,
        output_type="dict",
    )

    assert isinstance(out, dict)
    assert "t" in out.keys()
    assert "tfce" in out.keys()
    assert out["t"].shape == (n_regressors, n_descriptors)
    assert out["tfce"].shape == (n_regressors, n_descriptors)

    # permutations, TFCE, and masker are defined,
    # so check for TFCE maps
    n_perm = 10
    out = permuted_ols(
        tested_var,
        target_var,
        model_intercept=False,
        two_sided_test=False,
        n_perm=n_perm,
        random_state=random_state,
        threshold=None,
        masker=masker,
        tfce=True,
        output_type="dict",
    )

    assert isinstance(out, dict)
    assert "t" in out.keys()
    assert "tfce" in out.keys()
    assert "logp_max_t" in out.keys()
    assert "logp_max_tfce" in out.keys()
    assert "h0_max_t" in out.keys()
    assert "h0_max_tfce" in out.keys()
    assert out["t"].shape == (n_regressors, n_descriptors)
    assert out["tfce"].shape == (n_regressors, n_descriptors)
    assert out["logp_max_t"].shape == (n_regressors, n_descriptors)
    assert out["logp_max_tfce"].shape == (n_regressors, n_descriptors)
    assert out["h0_max_t"].size == n_perm
    assert out["h0_max_tfce"].size == n_perm


def test_cluster_level_parameters_smoke(random_state=0):
    """Test combinations of parameters related to cluster-level inference."""
    import nibabel as nib
    from nilearn.maskers import NiftiMasker

    rng = np.random.RandomState(random_state)

    # create design
    target_var1 = np.arange(0, 10).reshape((-1, 1))  # positive effect
    voxel_vars = np.hstack(
        (
            -target_var1,  # negative effect
            target_var1,  # positive effect
            rng.random((10, 1)),  # random voxel
        )
    )

    columns = np.arange(0, voxel_vars.shape[1])
    # create 125 voxels
    chosen_columns = rng.choice(columns, size=125, p=[0.1, 0.1, 0.8])
    # corresponds to 5 x 5 x 5 x 10 niimg
    target_var = voxel_vars[:, chosen_columns]
    tested_var = np.arange(0, 20, 2)

    mask_img = nib.Nifti1Image(np.ones((5, 5, 5)), np.eye(4))
    masker = NiftiMasker(mask_img)
    masker.fit(mask_img)

    # threshold is defined, indicating cluster-level inference should be done,
    # but masker is not defined.
    with pytest.raises(ValueError):
        permuted_ols(
            tested_var,
            target_var,
            model_intercept=False,
            two_sided_test=False,
            n_perm=100,
            random_state=random_state,
            threshold=0.001,
            masker=None,
            tfce=False,
        )

    # masker is defined, but threshold is not.
    # no cluster-level inference is performed, but there's a warning.
    with pytest.warns(Warning):
        out = permuted_ols(
            tested_var,
            target_var,
            model_intercept=False,
            two_sided_test=False,
            n_perm=100,
            random_state=random_state,
            threshold=None,
            masker=masker,
            tfce=False,
            output_type="legacy",
        )

    assert isinstance(out, tuple)

    # threshold is defined, but output_type is "legacy".
    # raise a warning, and get a dictionary.
    with pytest.warns(Warning):
        out = permuted_ols(
            tested_var,
            target_var,
            model_intercept=False,
            two_sided_test=False,
            n_perm=0,
            random_state=random_state,
            threshold=0.001,
            masker=masker,
            tfce=False,
            output_type="legacy",
        )

    assert isinstance(out, dict)

    # output_type is "legacy".
    # raise a deprecation warning, but get the standard output.
    with pytest.warns(DeprecationWarning):
        out = permuted_ols(
            tested_var,
            target_var,
            model_intercept=False,
            two_sided_test=False,
            n_perm=100,
            random_state=random_state,
            threshold=None,
            masker=None,
            tfce=False,
            output_type="legacy",
        )

    assert isinstance(out, tuple)

    # no permutations and output_type is "dict", so check for "t" map
    out = permuted_ols(
        tested_var,
        target_var,
        model_intercept=False,
        two_sided_test=False,
        n_perm=0,
        random_state=random_state,
        threshold=None,
        masker=None,
        tfce=False,
        output_type="dict",
    )

    assert isinstance(out, dict)
    assert "t" in out.keys()

    # permutations, threshold, and masker are defined,
    # so check for cluster-level maps
    n_perm = 10
    out = permuted_ols(
        tested_var,
        target_var,
        model_intercept=False,
        two_sided_test=True,
        n_perm=n_perm,
        random_state=random_state,
        threshold=0.001,
        masker=masker,
        output_type="dict",
    )

    assert isinstance(out, dict)
    assert "t" in out.keys()
    assert "logp_max_t" in out.keys()
    assert "logp_max_size" in out.keys()
    assert "logp_max_mass" in out.keys()
    assert "h0_max_t" in out.keys()
    assert "h0_max_size" in out.keys()
    assert "h0_max_mass" in out.keys()
    assert out["h0_max_t"].size == n_perm
    assert out["h0_max_size"].size == n_perm
    assert out["h0_max_mass"].size == n_perm
