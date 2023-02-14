#!/usr/bin/env python
import numpy as np
import pandas as pd
import pytest
import scipy.stats as sps
import scipy.linalg as spl

from numpy.testing import assert_almost_equal, assert_array_almost_equal
from scipy.stats import norm

from nilearn._utils.data_gen import generate_fake_fmri
from nilearn._utils.glm import (_check_and_load_tables,
                                _check_list_length_match, _check_run_tables,
                                full_rank,
                                multiple_fast_inverse,
                                multiple_mahalanobis,
                                positive_reciprocal, z_score)
from nilearn.maskers import NiftiMasker
from nilearn.glm.first_level import (FirstLevelModel,
                                     make_first_level_design_matrix)


def test_full_rank():
    rng = np.random.RandomState(42)
    n, p = 10, 5
    X = rng.standard_normal(size=(n, p))
    X_, _ = full_rank(X)
    assert_array_almost_equal(X, X_)
    X[:, -1] = X[:, :-1].sum(1)
    X_, cond = full_rank(X)
    assert cond > 1.e10
    assert_array_almost_equal(X, X_)


def test_z_score():
    # ################# Check z-scores computed from t-values #################
    # Randomly draw samples from the standard Student’s t distribution
    tval = np.random.RandomState(42).standard_t(10, size=10)
    # Estimate the p-values using the Survival Function (SF)
    pval = sps.t.sf(tval, 1e10)
    # Estimate the p-values using the Cumulative Distribution Function (CDF)
    cdfval = sps.t.cdf(tval, 1e10)
    # Set a minimum threshold for p-values to avoid infinite z-scores
    pval = np.array(np.minimum(np.maximum(pval, 1.e-300), 1. - 1.e-16))
    cdfval = np.array(np.minimum(np.maximum(cdfval, 1.e-300), 1. - 1.e-16))
    # Compute z-score from the p-value estimated with the SF
    zval_sf = norm.isf(pval)
    # Compute z-score from the p-value estimated with the CDF
    zval_cdf = norm.ppf(cdfval)
    # Create the final array of z-scores, ...
    zval = np.zeros(pval.size)
    # ... in which z-scores < 0 estimated w/ SF are replaced by z-scores < 0
    # estimated w/ CDF
    zval[np.atleast_1d(zval_sf < 0)] = zval_cdf[zval_sf < 0]
    # ... and z-scores >=0 estimated from SF are kept.
    zval[np.atleast_1d(zval_sf >= 0)] = zval_sf[zval_sf >= 0]
    # Test 'z_score' function in 'nilearn/glm/contrasts.py'
    assert_array_almost_equal(z_score(pval, one_minus_pvalue=cdfval), zval)
    # Test 'z_score' function in 'nilearn/glm/contrasts.py',
    # when one_minus_pvalue is None
    assert_array_almost_equal(norm.sf(z_score(pval)), pval)
    # ################# Check z-scores computed from F-values #################
    # Randomly draw samples from the F distribution
    fval = np.random.RandomState(42).f(1, 48, size=10)
    # Estimate the p-values using the Survival Function (SF)
    p_val = sps.f.sf(fval, 42, 1e10)
    # Estimate the p-values using the Cumulative Distribution Function (CDF)
    cdf_val = sps.f.cdf(fval, 42, 1e10)
    # Set a minimum threshold for p-values to avoid infinite z-scores
    p_val = np.array(np.minimum(np.maximum(p_val, 1.e-300), 1. - 1.e-16))
    cdf_val = np.array(np.minimum(np.maximum(cdf_val, 1.e-300), 1. - 1.e-16))
    # Compute z-score from the p-value estimated with the SF
    z_val_sf = norm.isf(p_val)
    # Compute z-score from the p-value estimated with the CDF
    z_val_cdf = norm.ppf(cdf_val)
    # Create the final array of z-scores, ...
    z_val = np.zeros(p_val.size)
    # ... in which z-scores < 0 estimated w/ SF are replaced by z-scores < 0
    # estimated w/ CDF
    z_val[np.atleast_1d(z_val_sf < 0)] = z_val_cdf[z_val_sf < 0]
    # ... and z-scores >=0 estimated from SF are kept.
    z_val[np.atleast_1d(z_val_sf >= 0)] = z_val_sf[z_val_sf >= 0]
    # Test 'z_score' function in 'nilearn/glm/contrasts.py'
    assert_array_almost_equal(z_score(p_val, one_minus_pvalue=cdf_val), z_val)
    # Test 'z_score' function in 'nilearn/glm/contrasts.py',
    # when one_minus_pvalue is None
    assert_array_almost_equal(norm.sf(z_score(p_val)), p_val)
    # ##################### Check the numerical precision #####################
    for t in [33.75, -8.3]:
        p = sps.t.sf(t, 1e10)
        cdf = sps.t.cdf(t, 1e10)
        z_sf = norm.isf(p)
        z_cdf = norm.ppf(cdf)
        if p <= .5:
            z = z_sf
        else:
            z = z_cdf
        assert_array_almost_equal(z_score(p, one_minus_pvalue=cdf), z)


def test_z_score_opposite_contrast():
    fmri, mask = generate_fake_fmri(shape=(50, 20, 50), length=96,
                                    random_state=np.random.RandomState(42))

    nifti_masker = NiftiMasker(mask_img=mask)
    data = nifti_masker.fit_transform(fmri)

    frametimes = np.linspace(0, (96 - 1) * 2, 96)

    for i in [0, 20]:
        design_matrix = make_first_level_design_matrix(
            frametimes, hrf_model='spm',
            add_regs=np.array(data[:, i]).reshape(-1, 1))
        c1 = np.array([1] + [0] * (design_matrix.shape[1] - 1))
        c2 = np.array([0] + [1] + [0] * (design_matrix.shape[1] - 2))
        contrasts = {'seed1 - seed2': c1 - c2, 'seed2 - seed1': c2 - c1}
        fmri_glm = FirstLevelModel(t_r=2.,
                                   noise_model='ar1',
                                   standardize=False,
                                   hrf_model='spm',
                                   drift_model='cosine')
        fmri_glm.fit(fmri, design_matrices=design_matrix)
        z_map_seed1_vs_seed2 = fmri_glm.compute_contrast(
            contrasts['seed1 - seed2'], output_type='z_score')
        z_map_seed2_vs_seed1 = fmri_glm.compute_contrast(
            contrasts['seed2 - seed1'], output_type='z_score')
        assert_almost_equal(z_map_seed1_vs_seed2.get_fdata(
                                dtype="float32"
                            ).min(),
                            -z_map_seed2_vs_seed1.get_fdata(
                                dtype="float32"
                            ).max(),
                            decimal=10)
        assert_almost_equal(z_map_seed1_vs_seed2.get_fdata(
                                dtype="float32"
                            ).max(),
                            -z_map_seed2_vs_seed1.get_fdata(
                                dtype="float32"
                            ).min(),
                            decimal=10)


def test_mahalanobis():
    rng = np.random.RandomState(42)
    n = 50
    x = rng.uniform(size=n) / n
    A = rng.uniform(size=(n, n)) / n
    A = np.dot(A.transpose(), A) + np.eye(n)
    mah = np.dot(x, np.dot(spl.inv(A), x))
    assert_almost_equal(mah, multiple_mahalanobis(x, A), decimal=1)


def test_mahalanobis2():
    rng = np.random.RandomState(42)
    n = 50
    x = rng.standard_normal(size=(n, 3))
    Aa = np.zeros([n, n, 3])
    for i in range(3):
        A = rng.standard_normal(size=(120, n))
        A = np.dot(A.T, A)
        Aa[:, :, i] = A
    i = rng.randint(3)
    mah = np.dot(x[:, i], np.dot(spl.inv(Aa[:, :, i]), x[:, i]))
    f_mah = (multiple_mahalanobis(x, Aa))[i]
    assert np.allclose(mah, f_mah)


def test_mahalanobis_errors():
    effect = np.zeros((1, 2, 3))
    cov = np.zeros((3, 3, 3))
    with pytest.raises(ValueError):
        multiple_mahalanobis(effect, cov)

    cov = np.zeros((1, 2, 3))
    with pytest.raises(ValueError):
        multiple_mahalanobis(effect, cov)


def test_multiple_fast_inv():
    rng = np.random.RandomState(42)
    shape = (10, 20, 20)
    X = rng.standard_normal(size=shape)
    X_inv_ref = np.zeros(shape)
    for i in range(shape[0]):
        X[i] = np.dot(X[i], X[i].T)
        X_inv_ref[i] = spl.inv(X[i])
    X_inv = multiple_fast_inverse(X)
    assert_almost_equal(X_inv_ref, X_inv)


def test_multiple_fast_inverse_errors():
    shape = (2, 2, 2)
    X = np.zeros(shape)
    with pytest.raises(ValueError):
        multiple_fast_inverse(X)

    shape = (10, 20, 20)
    X = np.zeros(shape)
    with pytest.raises(ValueError):
        multiple_fast_inverse(X)


def test_pos_recipr():
    X = np.array([2, 1, -1, 0], dtype=np.int8)
    eX = np.array([0.5, 1, 0, 0])
    Y = positive_reciprocal(X)
    assert_array_almost_equal, Y, eX
    assert Y.dtype.type == np.float64
    X2 = X.reshape((2, 2))
    Y2 = positive_reciprocal(X2)
    assert_array_almost_equal, Y2, eX.reshape((2, 2))
    # check that lists have arrived
    XL = [0, 1, -1]
    assert_array_almost_equal, positive_reciprocal(XL), [0, 1, 0]
    # scalars
    assert positive_reciprocal(-1) == 0
    assert positive_reciprocal(0) == 0
    assert positive_reciprocal(2) == 0.5


def test_img_table_checks():
    # check matching lengths
    with pytest.raises(ValueError):
        _check_list_length_match([''] * 2, [''], "", "")
    # check tables type and that can be loaded
    with pytest.raises(ValueError):
        _check_and_load_tables(['.csv', '.csv'], "")
    with pytest.raises(TypeError):
        _check_and_load_tables([[], pd.DataFrame()], "")  # np.array([0]),
    with pytest.raises(ValueError):
        _check_and_load_tables(['.csv', pd.DataFrame()], "")
    # check high level wrapper keeps behavior
    with pytest.raises(ValueError):
        _check_run_tables([''] * 2, [''], "")
    with pytest.raises(ValueError):
        _check_run_tables([''] * 2, ['.csv', '.csv'], "")
    with pytest.raises(TypeError):
        _check_run_tables([''] * 2, [[0], pd.DataFrame()], "")
    with pytest.raises(ValueError):
        _check_run_tables([''] * 2, ['.csv', pd.DataFrame()], "")
