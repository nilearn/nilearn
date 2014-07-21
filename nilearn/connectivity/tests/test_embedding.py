import copy

import numpy as np
from scipy import linalg
from numpy.testing import assert_array_almost_equal, assert_array_equal
from nose.tools import assert_raises, assert_equal, assert_is_instance

from ..._utils.testing import is_spd
from ..manifold import random_spd, sqrtm, expm
from ..embedding import sym_to_vec, vec_to_sym, prec_to_partial, \
    CovEmbedding


def test_sym_to_vec():
    """Testing sym_to_vec function"""
    # Check output value is correct
    sym = np.ones((3, 3))
    vec = sym_to_vec(sym)
    vec_expected = np.array([1., np.sqrt(2), 1., np.sqrt(2),  np.sqrt(2), 1.])
    assert_array_almost_equal(vec, vec_expected)
    mask = sym_to_vec(sym > 0, isometry=False)
    mask_expected = np.ones(6, dtype=bool)
    assert_array_equal(mask, mask_expected)

    # Check vec_to_sym is the inverse function of sym_to_vec
    n_features = 19
    m = np.random.rand(n_features, n_features)
    sym = m + m.T
    vec = sym_to_vec(sym)
    assert_array_almost_equal(vec_to_sym(vec), sym)
    syms = np.asarray([sym, 2. * sym, 0.5 * sym])
    vecs = sym_to_vec(syms)
    assert_array_almost_equal(vec_to_sym(vecs), syms)

    vec = sym_to_vec(sym, isometry=False)
    assert_array_almost_equal(vec_to_sym(vec, isometry=False),
                              sym)
    assert_array_almost_equal(vec[..., -n_features:], sym[..., -1, :])
    vecs = sym_to_vec(syms, isometry=False)
    assert_array_almost_equal(vec_to_sym(vecs, isometry=False),
                              syms)
    assert_array_almost_equal(vecs[..., -n_features:], syms[..., -1, :])


def test_vec_to_sym():
    """Testing vec_to_sym function"""
    # Check error if unsuitable size
    vec = np.random.rand(31)
    assert_raises(ValueError, vec_to_sym, vec)

    # Check output value is correct
    vec = np.ones(6, )
    sym = vec_to_sym(vec)
    sym_expected = np.array([[np.sqrt(2), 1., 1.], [1., np.sqrt(2), 1.],
                              [1., 1., np.sqrt(2)]]) / np.sqrt(2)
    assert_array_almost_equal(sym, sym_expected)
    mask = vec_to_sym(vec > 0, isometry=False)
    mask_expected = np.ones((3, 3), dtype=bool)
    assert_array_equal(mask, mask_expected)

    # Check sym_to_vec the inverse function of vec_to_sym
    n = 41
    p = n * (n + 1) / 2
    vec = np.random.rand(p)
    sym = vec_to_sym(vec)
    assert_array_almost_equal(sym_to_vec(sym), vec)
    sym = vec_to_sym(vec, isometry=False)
    assert_array_almost_equal(sym_to_vec(sym, isometry=False),
                              vec)
    vecs = np.asarray([vec, 2. * vec, 0.5 * vec])
    syms = vec_to_sym(vecs)
    assert_array_almost_equal(sym_to_vec(syms), vecs)
    syms = vec_to_sym(vecs, isometry=False)
    assert_array_almost_equal(sym_to_vec(syms, isometry=False),
                              vecs)


def test_prec_to_partial():
    """Testing prec_to_partial function"""
    n_features = 101
    prec = random_spd(n_features)
    partial = prec_to_partial(prec)
    d = np.sqrt(np.diag(np.diag(prec)))
    assert_array_almost_equal(d.dot(partial).dot(d), -prec +\
        2 * np.diag(np.diag(prec)))


def test_transform():
    """Testing fit_transform method for class CovEmbedding"""
    n_subjects = 10
    n_features = 49
    n_samples = 200

    # Generate signals and compute empirical covariances
    covs = []
    signals = []
    for k in xrange(n_subjects):
        signal = np.random.randn(n_samples, n_features)
        signals.append(signal)
        signal -= signal.mean(axis=0)
        covs.append((signal.T).dot(signal) / n_samples)

    input_covs = copy.copy(covs)
    for kind in ["correlation", "tangent", "precision", "partial correlation"]:
        estimators = {'kind': kind, 'cov_estimator': None}
        cov_embedding = CovEmbedding(**estimators)
        covs_transformed = cov_embedding.fit_transform(signals)

        # Generic
        assert_is_instance(covs_transformed, np.ndarray)
        assert_equal(len(covs_transformed), len(covs))

        for k, vec in enumerate(covs_transformed):
            assert_equal(vec.size, n_features * (n_features + 1) / 2)
            assert_array_equal(input_covs[k], covs[k])
            cov_new = vec_to_sym(vec)
            assert(is_spd(covs[k]))

            # Positive definiteness if expected and output value checks
            if estimators["kind"] == "tangent":
                assert_array_almost_equal(cov_new, cov_new.T)
                geo_sqrt = sqrtm(cov_embedding.mean_cov_)
                assert(is_spd(geo_sqrt))
                assert(is_spd(cov_embedding.whitening_))
                assert_array_almost_equal(
                cov_embedding.whitening_.dot(geo_sqrt), np.eye(n_features))
                assert_array_almost_equal(geo_sqrt.dot(expm(cov_new)).\
                    dot(geo_sqrt), covs[k])
            if estimators["kind"] == "precision":
                assert(is_spd(cov_new))
                assert_array_almost_equal(cov_new.dot(covs[k]),
                                          np.eye(n_features))
            if estimators["kind"] == "correlation":
                assert(is_spd(cov_new))
                d = np.sqrt(np.diag(np.diag(covs[k])))
                assert_array_almost_equal(d.dot(cov_new).dot(d), covs[k])
            if estimators["kind"] == "partial correlation":
                prec = linalg.inv(covs[k])
                d = np.sqrt(np.diag(np.diag(prec)))
                assert_array_almost_equal(d.dot(cov_new).dot(d), -prec +\
                    2 * np.diag(np.diag(prec)))