"""Tests for the permuted_ols function."""

# Author: Virgile Fritsch, <virgile.fritsch@inria.fr>, Feb. 2014

import numpy as np
import pytest
from nibabel import Nifti1Image
from numpy.testing import (
    assert_array_almost_equal,
    assert_array_less,
    assert_equal,
)
from scipy import stats

from nilearn.conftest import _rng
from nilearn.maskers import NiftiMasker
from nilearn.mass_univariate import permuted_ols
from nilearn.mass_univariate.permuted_least_squares import (
    _sanitize_inputs_permuted_ols,
)

N_COVARS = 2

N_PERM = 10

N_SAMPLES = 50


def _tfce_design():
    target_var1 = np.arange(0, 10).reshape((-1, 1))  # positive effect
    target_var = np.hstack(
        (  # corresponds to 3 x 3 x 3 x 10 niimg
            target_var1,  # voxel 1 has positive effect
            -target_var1,  # voxel 2 has negative effect
            _rng().random((10, 25)),  # 25 remaining voxels
        )
    )
    tested_var = np.arange(0, 20, 2)

    mask_img = Nifti1Image(np.ones((3, 3, 3)), np.eye(4))
    masker = NiftiMasker(mask_img)
    masker.fit(mask_img)

    n_descriptors = np.prod(mask_img.shape)
    n_regressors = 1  # tested_var is 1D

    return target_var, tested_var, masker, n_descriptors, n_regressors


def compare_to_ref_score(own_score, tested_var, target_var, covars=None):
    reference = ref_score(tested_var, target_var, covars)
    assert_array_almost_equal(own_score, reference, decimal=6)
    return reference


def ref_score(tested_var, target_var, covars=None):
    """Compute t-scores with linalg or statsmodels."""
    from nilearn.mass_univariate.tests._testing import (
        get_tvalue_with_alternative_library,
    )

    return get_tvalue_with_alternative_library(tested_var, target_var, covars)


def _create_design(rng, n_samples, n_descriptors, n_regressors):
    target_var = rng.standard_normal((n_samples, n_descriptors))
    tested_var = rng.standard_normal((n_samples, n_regressors))

    return target_var, tested_var, n_descriptors, n_regressors


@pytest.fixture
def design(rng):
    return _create_design(
        rng, n_samples=N_SAMPLES, n_descriptors=1, n_regressors=1
    )


@pytest.fixture
def dummy_design(rng):
    """Use to test errors and warnings."""
    return _create_design(rng, n_samples=10, n_descriptors=1, n_regressors=1)


@pytest.fixture
def confounding_vars(rng):
    return rng.standard_normal((N_SAMPLES, N_COVARS))


@pytest.fixture()
def masker(affine_eye):
    mask_img = Nifti1Image(np.ones((5, 5, 5)), affine_eye)
    masker = NiftiMasker(mask_img)
    masker.fit(mask_img)
    return masker


@pytest.fixture()
def cluster_level_design(rng):
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

    return target_var, tested_var


# General tests for permuted_ols function
#
# Check that h0 is close to the theoretical distribution
# for permuted OLS with label swap.
#
# Theoretical distribution is known for this simple design t(n_samples - dof).


PERM_RANGES = [10, 100, 1000]


def run_permutations(tested_var, target_var, model_intercept):
    """Compute the Mean Squared Error between cumulative Density Function \
    as a proof of consistency of the permutation algorithm.
    """
    all_mse = []
    all_kstest_pvals = []

    for i, n_perm in enumerate(np.repeat(PERM_RANGES, 10)):
        if model_intercept:
            h0 = permuted_ols_with_intercept(
                tested_var, target_var, int(n_perm), i
            )
            dof = N_SAMPLES - 2
        else:
            h0 = permuted_ols_no_intercept(
                tested_var, target_var, int(n_perm), i
            )
            dof = N_SAMPLES - 1

        h0_intercept = h0[0, :]
        kstest_pval, mse = ks_stat_and_mse(dof, h0_intercept)

        all_kstest_pvals.append(kstest_pval)
        all_mse.append(mse)

    return all_kstest_pvals, all_mse


def permuted_ols_no_intercept(tested_var, target_var, n_perm, i):
    n_regressors = 1
    output = permuted_ols(
        tested_var,
        target_var,
        model_intercept=False,
        n_perm=n_perm,
        two_sided_test=False,
        random_state=i,
        output_type="dict",
        verbose=1,
    )
    assert_equal(output["h0_max_t"].shape, (n_regressors, n_perm))
    return output["h0_max_t"]


def permuted_ols_with_intercept(tested_var, target_var, n_perm, i):
    output = permuted_ols(
        tested_var,
        target_var,
        model_intercept=True,
        n_perm=n_perm,
        two_sided_test=False,
        random_state=i,
        output_type="dict",
        verbose=1,
    )
    # pval should not be significant
    assert_array_less(output["logp_max_t"], 1.0)
    return output["h0_max_t"]


def ks_stat_and_mse(df, h0_intercept):
    """Run Kolmogorov-Smirnov test and compute Mean Squared Error."""
    kstest_pval = stats.kstest(h0_intercept, stats.t(df).cdf)[1]
    mse = mean_squared_error(df=df, h0_intercept=h0_intercept)
    return kstest_pval, mse


def mean_squared_error(df, h0_intercept):
    return np.mean(
        (
            stats.t(df).cdf(np.sort(h0_intercept))
            - np.linspace(0, 1, h0_intercept.size + 1)[1:]
        )
        ** 2
    )


def check_ktest_p_values_distribution_and_mse(all_kstest_pvals, all_mse):
    # check that a difference between distributions is not rejected by KS test
    all_kstest_pvals = np.array(all_kstest_pvals).reshape(
        (len(PERM_RANGES), -1)
    )
    assert_array_less(0.01, all_kstest_pvals)

    # consistency of the algorithm: the more permutations, the less the MSE
    all_mse = np.array(all_mse).reshape((len(PERM_RANGES), -1))
    assert_array_less(np.diff(all_mse.mean(1)), 0)


@pytest.mark.parametrize("model_intercept", [True, False])
def test_permuted_ols_check_h0_noeffect_labelswap_centered(model_intercept):
    # create dummy design with no effect
    rng = np.random.RandomState(0)
    target_var = rng.randn(N_SAMPLES, 1)

    centered_var = np.arange(N_SAMPLES, dtype="f8").reshape((-1, 1))
    centered_var -= centered_var.mean(0)

    all_kstest_pvals, all_mse = run_permutations(
        centered_var, target_var, model_intercept=model_intercept
    )

    check_ktest_p_values_distribution_and_mse(all_kstest_pvals, all_mse)


def test_permuted_ols_check_h0_noeffect_labelswap_uncentered():
    # create dummy design with no effect
    rng = np.random.RandomState(0)
    target_var = rng.randn(N_SAMPLES, 1)

    uncentered_var = np.arange(N_SAMPLES, dtype="f8").reshape((-1, 1))

    all_kstest_pvals, all_mse = run_permutations(
        uncentered_var, target_var, model_intercept=True
    )

    check_ktest_p_values_distribution_and_mse(all_kstest_pvals, all_mse)


def test_permuted_ols_check_h0_noeffect_signswap():
    """Check that h0 is close to the theoretical distribution \
    for permuted OLS with sign swap.

    Theoretical distribution is known for this simple design \
        (= t(n_samples - dof)).
    """
    # create dummy design with no effect
    rng = np.random.RandomState(0)
    target_var = rng.randn(N_SAMPLES, 1)

    n_regressors = 1
    tested_var = np.ones((N_SAMPLES, n_regressors))

    all_kstest_pvals, all_mse = run_permutations(
        tested_var, target_var, model_intercept=False
    )

    all_kstest_pvals = np.array(all_kstest_pvals).reshape(
        (len(PERM_RANGES), -1)
    )
    all_mse = np.array(all_mse).reshape((len(PERM_RANGES), -1))

    # check that a difference between distributions is not rejected by KS test
    assert_array_less(0.01 / (len(PERM_RANGES) * 10.0), all_kstest_pvals)
    # consistency of the algorithm: the more permutations, the less the MSE
    assert_array_less(np.diff(all_mse.mean(1)), 0)


# Tests for labels swapping permutation scheme


def test_permuted_ols_no_covar(design):
    target_var, tested_var, *_ = design
    output = permuted_ols(
        tested_var,
        target_var,
        model_intercept=False,
        n_perm=0,
        random_state=0,
        output_type="dict",
        verbose=1,
    )
    compare_to_ref_score(output["t"], tested_var, target_var)


def test_permuted_ols_no_covar_with_ravelized_tested_var(design):
    target_var, tested_var, *_ = design

    output = permuted_ols(
        np.ravel(tested_var),
        target_var,
        model_intercept=False,
        n_perm=0,
        random_state=0,
        output_type="dict",
        verbose=1,
    )
    compare_to_ref_score(output["t"], tested_var, target_var)


def test_permuted_ols_no_covar_with_intercept(design):
    # Adds intercept (should be equivalent to centering variates)
    target_var, tested_var, *_ = design

    output = permuted_ols(
        tested_var,
        target_var,
        model_intercept=True,
        n_perm=0,
        random_state=0,
        output_type="dict",
        verbose=1,
    )
    target_var -= target_var.mean(0)
    tested_var -= tested_var.mean(0)

    compare_to_ref_score(
        output["t"], tested_var, target_var, np.ones((N_SAMPLES, 1))
    )


def test_permuted_ols_with_covar(design, confounding_vars):
    target_var, tested_var, n_descriptors, n_regressors = design

    output = permuted_ols(
        tested_var,
        target_var,
        confounding_vars,
        model_intercept=False,
        n_perm=0,
        random_state=0,
        output_type="dict",
        verbose=1,
    )

    ref_score = compare_to_ref_score(
        output["t"], tested_var, target_var, confounding_vars
    )
    assert output["t"].shape == (n_regressors, n_descriptors)
    assert ref_score.shape == (n_regressors, n_descriptors)


def test_permuted_ols_with_covar_with_intercept(design, confounding_vars):
    target_var, tested_var, n_descriptors, n_regressors = design

    output = permuted_ols(
        tested_var,
        target_var,
        confounding_vars,
        model_intercept=True,
        n_perm=0,
        random_state=0,
        output_type="dict",
        verbose=1,
    )

    confounding_vars = np.hstack((confounding_vars, np.ones((N_SAMPLES, 1))))
    ref_score = compare_to_ref_score(
        output["t"], tested_var, target_var, confounding_vars
    )
    assert output["t"].shape == (n_regressors, n_descriptors)
    assert ref_score.shape == (n_regressors, n_descriptors)


@pytest.mark.parametrize("model_intercept", [True, False])
def test_permuted_ols_with_covar_with_intercept_in_confonding_vars(
    design, model_intercept
):
    target_var, tested_var, n_descriptors, n_regressors = design
    confounding_vars = np.ones([N_SAMPLES, 1])

    output = permuted_ols(
        tested_var,
        target_var,
        confounding_vars,
        model_intercept=model_intercept,
        n_perm=0,
        random_state=0,
        output_type="dict",
        verbose=1,
    )
    assert output["t"].shape == (n_regressors, n_descriptors)


def test_permuted_ols_with_multiple_constants_and_covars(design, rng):
    target_var, tested_var, n_descriptors, n_regressors = design

    n_covars = 2

    confounding_vars = np.hstack(
        (rng.standard_normal((N_SAMPLES, n_covars)), np.ones([N_SAMPLES, 2]))
    )
    output = permuted_ols(
        tested_var,
        target_var,
        confounding_vars,
        model_intercept=False,
        n_perm=0,
        random_state=0,
        output_type="dict",
        verbose=1,
    )
    assert output["t"].shape == (n_regressors, n_descriptors)


def test_permuted_ols_nocovar_multivariate(rng):
    """Test permuted_ols with multiple tested variates and no covariate.

    It is equivalent to fitting several models with only one tested variate.
    """
    n_descriptors = 10
    n_regressors = 2
    target_vars, tested_var, *_ = _create_design(
        rng,
        n_samples=N_SAMPLES,
        n_descriptors=n_descriptors,
        n_regressors=n_regressors,
    )

    n_perm = N_PERM
    output = permuted_ols(
        tested_var,
        target_vars,
        model_intercept=False,
        n_perm=n_perm,
        random_state=0,
        output_type="dict",
        verbose=1,
    )

    compare_to_ref_score(output["t"], tested_var, target_vars)

    assert output["logp_max_t"].shape == (n_regressors, n_descriptors)
    assert output["h0_max_t"].shape == (n_regressors, n_perm)

    # Adds intercept (should be equivalent to centering variates)
    output_intercept = permuted_ols(
        tested_var,
        target_vars,
        model_intercept=True,
        n_perm=0,
        random_state=0,
        output_type="dict",
        verbose=1,
    )

    target_vars -= target_vars.mean(0)
    tested_var -= tested_var.mean(0)
    compare_to_ref_score(
        output_intercept["t"], tested_var, target_vars, np.ones((N_SAMPLES, 1))
    )


# Tests for sign swapping permutation scheme


def test_permuted_ols_intercept_nocovar(rng):
    n_descriptors = 10
    n_regressors = 1
    tested_var = np.ones((N_SAMPLES, n_regressors))
    target_var = rng.standard_normal((N_SAMPLES, n_descriptors))

    output = permuted_ols(
        tested_var,
        target_var,
        confounding_vars=None,
        n_perm=N_PERM,
        random_state=0,
        output_type="dict",
        verbose=1,
    )

    ref_score = compare_to_ref_score(output["t"], tested_var, target_var)
    assert ref_score.shape == (n_regressors, n_descriptors)
    assert output["logp_max_t"].shape == (n_regressors, n_descriptors)
    assert output["t"].shape == (n_regressors, n_descriptors)
    assert_array_less(
        output["logp_max_t"], 1.0
    )  # ensure sign swap is correctly done

    # same thing but with model_intercept=True to check it has no effect
    output_addintercept = permuted_ols(
        tested_var,
        target_var,
        confounding_vars=None,
        model_intercept=False,
        n_perm=0,
        random_state=0,
        output_type="dict",
        verbose=1,
    )
    compare_to_ref_score(output_addintercept["t"], tested_var, target_var)
    assert output_addintercept["t"].shape == (n_regressors, n_descriptors)


def test_permuted_ols_intercept_statsmodels_withcovar(
    rng,
):
    n_descriptors = 10
    n_regressors = 1
    n_covars = 2
    tested_var = np.ones((N_SAMPLES, n_regressors))
    target_var = rng.standard_normal((N_SAMPLES, n_descriptors))
    confounding_vars = rng.standard_normal((N_SAMPLES, n_covars))

    output = permuted_ols(
        tested_var,
        target_var,
        confounding_vars,
        n_perm=0,
        random_state=0,
        output_type="dict",
        verbose=1,
    )
    ref_score = compare_to_ref_score(
        output["t"], tested_var, target_var, confounding_vars
    )
    assert ref_score.shape == (n_regressors, n_descriptors)
    assert output["t"].shape == (n_regressors, n_descriptors)

    # same thing but with model_intercept=True to check it has no effect
    output_intercept = permuted_ols(
        tested_var,
        target_var,
        confounding_vars,
        model_intercept=True,
        n_perm=0,
        random_state=0,
        output_type="dict",
        verbose=1,
    )
    compare_to_ref_score(
        output_intercept["t"], tested_var, target_var, confounding_vars
    )
    assert output_intercept["t"].shape == (n_regressors, n_descriptors)


def test_one_sided_versus_two_test(rng):
    """Check that a positive effect is always better \
    recovered with one-sided.
    """
    n_descriptors = 100
    n_regressors = 1
    target_var = rng.standard_normal((N_SAMPLES, n_descriptors))
    tested_var = rng.standard_normal((N_SAMPLES, n_regressors))

    # one-sided
    output_1_sided = permuted_ols(
        tested_var,
        target_var,
        model_intercept=False,
        two_sided_test=False,
        n_perm=N_PERM,
        random_state=0,
        output_type="dict",
        verbose=1,
    )
    assert output_1_sided["logp_max_t"].shape == (n_regressors, n_descriptors)

    # two-sided
    output_2_sided = permuted_ols(
        tested_var,
        target_var,
        model_intercept=False,
        two_sided_test=True,
        n_perm=N_PERM,
        random_state=0,
        output_type="dict",
        verbose=1,
    )
    assert output_2_sided["logp_max_t"].shape == (n_regressors, n_descriptors)

    positive_effect_location = output_1_sided["logp_max_t"] > 1
    assert_equal(
        np.sum(
            output_2_sided["logp_max_t"][positive_effect_location]
            - output_1_sided["logp_max_t"][positive_effect_location]
            > 0
        ),
        0,
    )


def test_two_sided_recover_positive_and_negative_effects():
    """Check that two-sided can actually recover \
    positive and negative effects.
    """
    target_var1 = np.arange(0, 10).reshape((-1, 1))  # positive effect
    target_var = np.hstack((target_var1, -target_var1))
    tested_var = np.arange(0, 20, 2)

    # one-sided
    output_1_sided_1 = permuted_ols(
        tested_var,
        target_var,
        model_intercept=False,
        two_sided_test=False,
        n_perm=N_PERM,
        random_state=0,
        output_type="dict",
        verbose=1,
    )
    output_1_sided_1["logp_max_t"]

    # one-sided (other side)
    output_1_sided_2 = permuted_ols(
        tested_var,
        -target_var,
        model_intercept=False,
        two_sided_test=False,
        n_perm=N_PERM,
        random_state=0,
        output_type="dict",
        verbose=1,
    )

    # two-sided
    output_2_sided = permuted_ols(
        tested_var,
        target_var,
        model_intercept=False,
        two_sided_test=True,
        n_perm=N_PERM,
        random_state=0,
        output_type="dict",
        verbose=1,
    )
    output_2_sided["logp_max_t"]

    assert_array_almost_equal(
        output_1_sided_1["logp_max_t"][0],
        output_1_sided_2["logp_max_t"][0][::-1],
    )
    assert_array_almost_equal(
        output_1_sided_1["logp_max_t"] + output_1_sided_2["logp_max_t"],
        output_2_sided["logp_max_t"],
    )


def test_tfce_smoke_legacy_smoke():
    (
        target_var,
        tested_var,
        masker,
        n_descriptors,
        n_regressors,
    ) = _tfce_design()

    # no permutations and output_type is "dict", so check for "t" and
    # "tfce" maps
    out = permuted_ols(
        tested_var,
        target_var,
        model_intercept=False,
        two_sided_test=False,
        n_perm=0,
        random_state=0,
        masker=masker,
        tfce=True,
        output_type="dict",
        verbose=1,
    )

    assert isinstance(out, dict)
    assert "t" in out
    assert "tfce" in out
    assert out["t"].shape == (n_regressors, n_descriptors)
    assert out["tfce"].shape == (n_regressors, n_descriptors)

    # permutations, TFCE, and masker are defined,
    # so check for TFCE maps
    n_perm = N_PERM
    out = permuted_ols(
        tested_var,
        target_var,
        model_intercept=False,
        two_sided_test=False,
        n_perm=n_perm,
        random_state=0,
        masker=masker,
        tfce=True,
        output_type="dict",
        verbose=1,
    )

    assert isinstance(out, dict)
    assert "t" in out
    assert "tfce" in out
    assert "logp_max_t" in out
    assert "logp_max_tfce" in out
    assert "h0_max_t" in out
    assert "h0_max_tfce" in out
    assert out["t"].shape == (n_regressors, n_descriptors)
    assert out["tfce"].shape == (n_regressors, n_descriptors)
    assert out["logp_max_t"].shape == (n_regressors, n_descriptors)
    assert out["logp_max_tfce"].shape == (n_regressors, n_descriptors)
    assert out["h0_max_t"].size == n_perm
    assert out["h0_max_tfce"].size == n_perm


def test_cluster_level_parameters_smoke(cluster_level_design, masker):
    """Test combinations of parameters related to cluster-level inference."""
    target_var, tested_var = cluster_level_design

    # no permutations and output_type is "dict", so check for "t" map
    out = permuted_ols(
        tested_var,
        target_var,
        model_intercept=False,
        two_sided_test=False,
        n_perm=0,
        random_state=0,
        output_type="dict",
        verbose=1,
    )

    assert isinstance(out, dict)
    assert "t" in out

    # permutations, threshold, and masker are defined,
    # so check for cluster-level maps
    n_perm = N_PERM
    out = permuted_ols(
        tested_var,
        target_var,
        model_intercept=False,
        two_sided_test=True,
        n_perm=n_perm,
        random_state=0,
        threshold=0.001,
        masker=masker,
        output_type="dict",
        verbose=1,
    )

    assert isinstance(out, dict)
    assert "t" in out
    assert "logp_max_t" in out
    assert "logp_max_size" in out
    assert "logp_max_mass" in out
    assert "h0_max_t" in out
    assert "h0_max_size" in out
    assert "h0_max_mass" in out
    assert out["h0_max_t"].size == n_perm
    assert out["h0_max_size"].size == n_perm
    assert out["h0_max_mass"].size == n_perm


def test_sanitize_inputs_permuted_ols(design):
    target_vars, tested_vars, *_ = design
    _sanitize_inputs_permuted_ols(
        n_jobs=-1,
        output_type="dict",
        tfce=False,
        threshold=None,
        target_vars=target_vars,
        tested_vars=tested_vars,
    )


def test_permuted_ols_warnings_n_perm_n_job(cluster_level_design, masker):
    """Check that proper warning are thrown depending on n_job VS n_perm."""
    target_var, tested_var = cluster_level_design

    # n_perm > n_job --> no warning
    with pytest.warns() as record:
        permuted_ols(
            tested_var,
            target_var,
            n_perm=4,
            n_jobs=1,
            masker=masker,
        )
    assert not any(
        "perform more permutations" in str(x.message) for x in record
    )

    # n_perm <= n_job  and n_job > 0 -->  warning
    with pytest.warns(
        UserWarning,
        match="perform more permutations",
    ):
        permuted_ols(tested_var, target_var, n_perm=1, masker=masker, n_jobs=2)


def test_cluster_level_parameters_warnings(cluster_level_design, masker):
    """Test combinations of parameters related to cluster-level inference."""
    target_var, tested_var = cluster_level_design

    # masker is defined, but threshold is not.
    # no cluster-level inference is performed, but there's a warning.
    with pytest.warns(
        DeprecationWarning,
        match='"legacy" output structure for "permuted_ols" is deprecated',
    ):
        out = permuted_ols(
            tested_var,
            target_var,
            model_intercept=False,
            two_sided_test=False,
            n_perm=N_PERM,
            random_state=0,
            masker=masker,
            output_type="legacy",
        )

    assert isinstance(out, tuple)

    # threshold is defined, but output_type is "legacy".
    # raise a warning, and get a dictionary.
    with pytest.warns(
        Warning,
        match='If "threshold" is not None',
    ):
        out = permuted_ols(
            tested_var,
            target_var,
            model_intercept=False,
            two_sided_test=False,
            n_perm=0,
            random_state=0,
            threshold=0.001,
            masker=masker,
            output_type="legacy",
        )

    assert isinstance(out, dict)

    # output_type is "legacy".
    # raise a deprecation warning, but get the standard output.
    with pytest.deprecated_call():
        out = permuted_ols(
            tested_var,
            target_var,
            model_intercept=False,
            two_sided_test=False,
            n_perm=N_PERM,
            random_state=0,
            output_type="legacy",
        )

    assert isinstance(out, tuple)


def test_permuted_ols_no_covar_warning(rng):
    """Ensure that a warning is raised when a given voxel has all zeros."""
    target_var, tested_var, *_ = _create_design(
        rng, n_samples=N_SAMPLES, n_descriptors=10, n_regressors=1
    )
    output_1 = permuted_ols(
        tested_var,
        target_var,
        model_intercept=False,
        n_perm=N_PERM,
        random_state=0,
        output_type="dict",
        verbose=1,
    )

    # test with ravelized tested_var
    target_var[:, 0] = 0

    with pytest.warns(UserWarning, match="have zeros across all samples"):
        output_2 = permuted_ols(
            np.ravel(tested_var),
            target_var,
            model_intercept=False,
            n_perm=N_PERM,
            random_state=0,
            output_type="dict",
        )

    assert np.array_equal(output_1["t"][1:], output_2["t"][1:])


def test_permuted_ols_with_multiple_constants_and_covars_warnings(design):
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
            random_state=0,
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
            random_state=0,
        )


def test_tfce_smoke_legacy_warnings():
    target_var, tested_var, masker, *_ = _tfce_design()

    # tfce is True, but output_type is "legacy".
    # raise a warning, and get a dictionary.
    with pytest.warns(UserWarning, match="Overriding."):
        out = permuted_ols(
            tested_var,
            target_var,
            model_intercept=False,
            two_sided_test=False,
            n_perm=0,
            random_state=0,
            masker=masker,
            tfce=True,
            output_type="legacy",
        )

    assert isinstance(out, dict)

    # output_type is "legacy".
    # raise a deprecation warning, but get the standard output.
    with pytest.deprecated_call():
        out = permuted_ols(
            tested_var,
            target_var,
            model_intercept=False,
            two_sided_test=False,
            n_perm=N_PERM,
            random_state=0,
            output_type="legacy",
        )

    assert isinstance(out, tuple)


def test_permuted_ols_no_covar_n_job_error(dummy_design):
    """Check that an invalid n_jobs value will raise a ValueError."""
    target_var, tested_var, *_ = dummy_design

    with pytest.raises(
        ValueError, match="'n_jobs == 0' is not a valid choice."
    ):
        permuted_ols(
            tested_var,
            target_var,
            n_jobs=0,  # not allowed
        )


def test_permuted_ols_target_vars_error(dummy_design):
    """Checks dimension of target_var."""
    target_var, tested_var, *_ = dummy_design

    with pytest.raises(
        ValueError, match="'target_vars' should be a 2D array."
    ):
        permuted_ols(
            tested_var,
            target_var.ravel(),  # must be 2D
        )


def test_permuted_ols_type_n_perm(dummy_design):
    """Checks type n_perm."""
    target_var, tested_var, *_ = dummy_design

    with pytest.raises(TypeError, match="must be an int"):
        permuted_ols(tested_var, target_var, n_perm=0.1)


def test_tfce_no_masker_error():
    target_var, tested_var, *_ = _tfce_design()

    with pytest.raises(ValueError, match="masker must be provided"):
        permuted_ols(
            tested_var,
            target_var,
            model_intercept=False,
            two_sided_test=False,
            n_perm=N_PERM,
            tfce=True,
        )


def test_cluster_level_parameters_error_no_masker(cluster_level_design):
    """Test combinations of parameters related to cluster-level inference."""
    target_var, tested_var = cluster_level_design

    # threshold is defined, indicating cluster-level inference should be done,
    # but masker is not defined.
    with pytest.raises(
        ValueError,
        match='If "threshold" is not None, masker must be defined as well.',
    ):
        permuted_ols(
            tested_var,
            target_var,
            two_sided_test=False,
            threshold=0.001,
            tfce=False,
        )
