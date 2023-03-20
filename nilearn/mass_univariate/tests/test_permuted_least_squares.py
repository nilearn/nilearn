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

RANDOM_STATE = 0

N_SAMPLES = 50

N_COVARS = 2


def mean_squared_error(df, h0_intercept):
    mse = np.mean(
        (
            stats.t(df).cdf(np.sort(h0_intercept))
            - np.linspace(0, 1, h0_intercept.size + 1)[1:]
        )
        ** 2
    )
    return mse


def ks_stat_and_mse(df, h0_intercept, kstest_pvals_list, mse_list):
    """Run Kolmogorov-Smirnov test and compute Mean Squared Error"""
    kstest_pval = stats.kstest(h0_intercept, stats.t(df).cdf)[1]
    kstest_pvals_list.append(kstest_pval)
    mse = mean_squared_error(df=df, h0_intercept=h0_intercept)
    mse_list.append(mse)


def permuted_ols_no_intercept(tested_var, target_var, n_perm, i):
    n_regressors = 1
    _, _, h0 = permuted_ols(
        tested_var,
        target_var,
        model_intercept=False,
        n_perm=n_perm,
        two_sided_test=False,
        random_state=i,
    )
    assert_equal(h0.shape, (n_regressors, n_perm))
    return h0


def permuted_ols_with_intercept(tested_var, target_var, n_perm, i):
    pval, _, h0 = permuted_ols(
        tested_var,
        target_var,
        model_intercept=True,
        n_perm=n_perm,
        two_sided_test=False,
        random_state=i,
    )
    assert_array_less(pval, 1.0)  # pval should not be significant
    return h0


def ref_score(tested_var, target_var, covars=None):
    """Compute t-scores with linalg or statsmodels."""
    return get_tvalue_with_alternative_library(tested_var, target_var, covars)


def compare_to_ref_score(own_score, tested_var, target_var, covars=None):
    reference = ref_score(tested_var, target_var, covars)
    assert_array_almost_equal(own_score, reference, decimal=6)
    return reference


def _create_design(n_samples, n_descriptors, n_regressors):
    random_state = RANDOM_STATE
    rng = check_random_state(random_state)

    # create design
    target_var = rng.randn(n_samples, n_descriptors)
    tested_var = rng.randn(n_samples, n_regressors)

    return target_var, tested_var, n_descriptors, n_regressors


@pytest.fixture
def design():
    return _create_design(n_samples=N_SAMPLES, n_descriptors=1, n_regressors=1)


@pytest.fixture
def dummy_design():
    """Use to test errors and warnings."""
    return _create_design(n_samples=10, n_descriptors=1, n_regressors=1)


@pytest.fixture
def confounding_vars():
    random_state = RANDOM_STATE
    rng = check_random_state(random_state)
    return rng.randn(N_SAMPLES, N_COVARS)


# General tests for permuted_ols function
def test_permuted_ols_check_h0_noeffect_labelswap(random_state=RANDOM_STATE):
    """Check that h0 is close to the theoretical distribution \
    for permuted OLS with label swap.

    Theoretical distribution is known for this simple design \
        (= t(n_samples - dof)).
    """
    rng = check_random_state(random_state)

    # design parameters
    n_samples = N_SAMPLES

    # create dummy design with no effect
    target_var = rng.randn(n_samples, 1)
    tested_var = np.arange(n_samples, dtype="f8").reshape((-1, 1))
    tested_var_not_centered = tested_var.copy()
    tested_var -= tested_var.mean(0)  # centered

    # we use two models (with and without intercept modelling)
    all_kstest_pvals = []
    all_kstest_pvals_intercept = []
    all_kstest_pvals_intercept2 = []

    # we compute the Mean Squared Error between cumulative Density Function
    # as a proof of consistency of the permutation algorithm
    all_mse = []
    all_mse_intercept = []
    all_mse_intercept2 = []

    # test various number of permutations
    perm_ranges = [10, 100, 1000]

    for i, n_perm in enumerate(np.repeat(perm_ranges, 10)):
        # Case no. 1: no intercept in the model
        h0 = permuted_ols_no_intercept(tested_var, target_var, n_perm, i)
        df = n_samples - 1
        h0_intercept = h0[0, :]
        ks_stat_and_mse(df, h0_intercept, all_kstest_pvals, all_mse)

        # Case no. 2: intercept in the model
        h0 = permuted_ols_with_intercept(tested_var, target_var, n_perm, i)
        df = n_samples - 2
        h0_intercept = h0[0, :]
        ks_stat_and_mse(
            df, h0_intercept, all_kstest_pvals_intercept, all_mse_intercept
        )

        # Case no. 3: intercept in the model, no centering of tested vars
        permuted_ols_with_intercept(
            tested_var_not_centered,
            target_var,
            n_perm,
            i,
        )
        df = n_samples - 2
        h0_intercept = h0[0, :]
        ks_stat_and_mse(
            df, h0_intercept, all_kstest_pvals_intercept2, all_mse_intercept2
        )

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


def test_permuted_ols_check_h0_noeffect_signswap(random_state=RANDOM_STATE):
    """Check that h0 is close to the theoretical distribution \
    for permuted OLS with sign swap.

    Theoretical distribution is known for this simple design \
        (= t(n_samples - dof)).
    """
    rng = check_random_state(random_state)

    # design parameters
    n_samples = N_SAMPLES
    n_regressors = 1

    # create dummy design with no effect
    target_var = rng.randn(n_samples, 1)
    tested_var = np.ones((n_samples, n_regressors))

    # we compute the Mean Squared Error between cumulative Density Function
    # as a proof of consistency of the permutation algorithm
    all_mse = []

    # test various number of permutations
    perm_ranges = [10, 100, 1000]
    all_kstest_pvals = []
    for i, n_perm in enumerate(np.repeat(perm_ranges, 10)):
        h0 = permuted_ols_no_intercept(tested_var, target_var, n_perm, i)
        ############################
        # CHECK if the original value of DoF was correct
        df = n_samples
        ############################
        h0_intercept = h0[0, :]
        ks_stat_and_mse(df, h0_intercept, all_kstest_pvals, all_mse)

    all_kstest_pvals = np.array(all_kstest_pvals).reshape(
        (len(perm_ranges), -1)
    )
    all_mse = np.array(all_mse).reshape((len(perm_ranges), -1))

    # check that a difference between distributions is not rejected by KS test
    assert_array_less(0.01 / (len(perm_ranges) * 10.0), all_kstest_pvals)

    # consistency of the algorithm: the more permutations, the less the MSE
    assert_array_less(np.diff(all_mse.mean(1)), 0)


# Tests for labels swapping permutation scheme
def test_permuted_ols_no_covar(design, random_state=RANDOM_STATE):
    target_var, tested_var, *_ = design

    # permuted OLS
    _, own_score, _ = permuted_ols(
        tested_var,
        target_var,
        model_intercept=False,
        n_perm=0,
        random_state=random_state,
    )
    compare_to_ref_score(own_score, tested_var, target_var)


def test_permuted_ols_no_covar_with_ravelized_tested_var(
    design, random_state=RANDOM_STATE
):
    target_var, tested_var, *_ = design

    _, own_score, _ = permuted_ols(
        np.ravel(tested_var),
        target_var,
        model_intercept=False,
        n_perm=0,
        random_state=random_state,
    )
    compare_to_ref_score(own_score, tested_var, target_var)


def test_permuted_ols_no_covar_with_intercept(
    design, random_state=RANDOM_STATE
):
    # Adds intercept (should be equivalent to centering variates)
    target_var, tested_var, *_ = design

    _, own_score, _ = permuted_ols(
        tested_var,
        target_var,
        model_intercept=True,
        n_perm=0,
        random_state=random_state,
    )
    target_var -= target_var.mean(0)
    tested_var -= tested_var.mean(0)

    compare_to_ref_score(
        own_score, tested_var, target_var, np.ones((N_SAMPLES, 1))
    )


def test_permuted_ols_no_covar_warning(random_state=RANDOM_STATE):
    """Ensure that a warning is raised when a given voxel has all zeros."""
    target_var, tested_var, *_ = _create_design(
        n_samples=N_SAMPLES, n_descriptors=10, n_regressors=1
    )

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


def test_permuted_ols_no_covar_n_job_error(dummy_design):
    """Ensure that a warning is raised when a given voxel has all zeros.

    This test also checks that an invalid n_jobs value will raise a ValueError.
    """
    target_var, tested_var, *_ = dummy_design

    with pytest.raises(ValueError):
        permuted_ols(
            tested_var,
            target_var,
            model_intercept=False,
            n_perm=100,
            n_jobs=0,  # not allowed
            random_state=RANDOM_STATE,
        )


def test_permuted_ols_with_covar(
    design, confounding_vars, random_state=RANDOM_STATE
):
    # design parameters
    target_var, tested_var, n_descriptors, n_regressors = design

    # permuted OLS
    _, own_score, _ = permuted_ols(
        tested_var,
        target_var,
        confounding_vars,
        model_intercept=False,
        n_perm=0,
        random_state=random_state,
    )

    ref_score = compare_to_ref_score(
        own_score, tested_var, target_var, confounding_vars
    )
    assert own_score.shape == (n_regressors, n_descriptors)
    assert ref_score.shape == (n_regressors, n_descriptors)


def test_permuted_ols_with_covar_with_intercept(
    design, confounding_vars, random_state=RANDOM_STATE
):
    # design parameters
    target_var, tested_var, n_descriptors, n_regressors = design

    _, own_score, _ = permuted_ols(
        tested_var,
        target_var,
        confounding_vars,
        model_intercept=True,
        n_perm=0,
        random_state=random_state,
    )

    confounding_vars = np.hstack((confounding_vars, np.ones((N_SAMPLES, 1))))
    ref_score = compare_to_ref_score(
        own_score, tested_var, target_var, confounding_vars
    )
    assert own_score.shape == (n_regressors, n_descriptors)
    assert ref_score.shape == (n_regressors, n_descriptors)


@pytest.mark.parametrize("model_intercept", [True, False])
def test_permuted_ols_with_covar_with_intercept_in_confonding_vars(
    design, model_intercept, random_state=RANDOM_STATE
):
    # design parameters
    target_var, tested_var, n_descriptors, n_regressors = design
    confounding_vars = np.ones([N_SAMPLES, 1])

    _, own_score, _ = permuted_ols(
        tested_var,
        target_var,
        confounding_vars,
        model_intercept=model_intercept,
        n_perm=0,
        random_state=random_state,
    )
    assert own_score.shape == (n_regressors, n_descriptors)


def test_permuted_ols_with_multiple_constants_and_covars(
    design, random_state=RANDOM_STATE
):
    # design parameters
    target_var, tested_var, n_descriptors, n_regressors = design

    n_covars = 2
    rng = check_random_state(random_state)

    confounding_vars = np.hstack(
        (rng.randn(N_SAMPLES, n_covars), np.ones([N_SAMPLES, 2]))
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


def test_permuted_ols_with_multiple_constants_and_covars_warnings(
    design, random_state=RANDOM_STATE
):
    # design parameters
    target_var, tested_var, *_ = design

    # Multiple intercepts should raise a warning
    # In confounding vars
    with pytest.warns(UserWarning, match="Multiple columns across"):
        confounding_vars = np.ones([N_SAMPLES, 2])
        permuted_ols(
            tested_var,
            target_var,
            confounding_vars,
            n_perm=0,
            random_state=random_state,
        )

    # Across tested vars and confounding vars
    with pytest.warns(UserWarning, match="Multiple columns across"):
        confounding_vars = np.ones([N_SAMPLES, 1])
        tested_var = np.ones([N_SAMPLES, 1])
        permuted_ols(
            tested_var,
            target_var,
            confounding_vars,
            n_perm=0,
            random_state=random_state,
        )


def test_permuted_ols_nocovar_multivariate(random_state=RANDOM_STATE):
    """Test permuted_ols with multiple tested variates and no covariate.

    It is equivalent to fitting several models with only one tested variate.
    """
    # design parameters
    n_descriptors = 10
    n_regressors = 2
    n_perm = 10
    target_vars, tested_var, *_ = _create_design(
        n_samples=N_SAMPLES,
        n_descriptors=n_descriptors,
        n_regressors=n_regressors,
    )

    neg_log10_pvals, own_scores, h0_fmax = permuted_ols(
        tested_var,
        target_vars,
        model_intercept=False,
        n_perm=n_perm,
        random_state=random_state,
    )

    compare_to_ref_score(own_scores, tested_var, target_vars)

    assert neg_log10_pvals.shape == (n_regressors, n_descriptors)
    assert h0_fmax.shape == (n_regressors, n_perm)

    # Adds intercept (should be equivalent to centering variates)
    _, own_scores_intercept, _ = permuted_ols(
        tested_var,
        target_vars,
        model_intercept=True,
        n_perm=0,
        random_state=random_state,
    )

    target_vars -= target_vars.mean(0)
    tested_var -= tested_var.mean(0)
    compare_to_ref_score(
        own_scores_intercept, tested_var, target_vars, np.ones((N_SAMPLES, 1))
    )


# Tests for sign swapping permutation scheme


def test_permuted_ols_intercept_nocovar(random_state=RANDOM_STATE):
    rng = check_random_state(random_state)

    # design parameters
    n_samples = N_SAMPLES
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


def test_permuted_ols_intercept_statsmodels_withcovar(
    random_state=RANDOM_STATE,
):
    rng = check_random_state(random_state)

    # design parameters
    n_samples = N_SAMPLES
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
def test_sided_test(random_state=RANDOM_STATE):
    """Check that a positive effect is always better \
    recovered with one-sided."""
    rng = check_random_state(random_state)

    # design parameters
    n_samples = N_SAMPLES
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


def test_sided_test2(random_state=RANDOM_STATE):
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


def test_tfce_smoke(random_state=RANDOM_STATE):
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


@pytest.fixture(scope="module")
def masker():
    import nibabel as nib
    from nilearn.maskers import NiftiMasker

    mask_img = nib.Nifti1Image(np.ones((5, 5, 5)), np.eye(4))
    masker = NiftiMasker(mask_img)
    masker.fit(mask_img)
    return masker


def test_cluster_level_parameters_smoke(masker, random_state=RANDOM_STATE):
    """Test combinations of parameters related to cluster-level inference."""
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
