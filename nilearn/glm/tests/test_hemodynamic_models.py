import warnings

import numpy as np
import pytest

from numpy.testing import (assert_almost_equal,
                           assert_array_equal,
                           assert_array_almost_equal,
                           )

from nilearn.glm.first_level.hemodynamic_models import \
    (_hrf_kernel, _orthogonalize, _regressor_names, _resample_regressor,
     _sample_condition, compute_regressor, spm_dispersion_derivative,
     spm_hrf, spm_time_derivative, glover_dispersion_derivative,
     glover_hrf, glover_time_derivative)


HRF_MODEL_NAMES = ['spm', 'glover', 'spm + derivative',
                   'glover + derivative',
                   'spm + derivative + dispersion',
                   'glover + derivative + dispersion']


HRF_MODELS = [spm_hrf, glover_hrf, spm_time_derivative,
              glover_time_derivative, spm_dispersion_derivative,
              glover_dispersion_derivative]


@pytest.fixture
def expected_integral(hrf_model):
    return 1 if hrf_model in [spm_hrf, glover_hrf] else 0


@pytest.fixture
def expected_length(tr):
    return int(32 / tr * 50)


@pytest.mark.parametrize('hrf_model', HRF_MODELS)
@pytest.mark.parametrize('tr', [2, 3])
def test_hrf_norm_and_length(hrf_model, tr, expected_integral,
                             expected_length):
    """ test that the hrf models are correctly normalized and
    have correct lengths.
    """
    h = hrf_model(tr)
    assert_almost_equal(h.sum(), expected_integral)
    assert len(h) == expected_length


def test_resample_regressor():
    """ test regressor resampling on a linear function
    """
    x = np.linspace(0, 1, 200)
    y = np.linspace(0, 1, 30)
    z = _resample_regressor(x, x, y)
    assert_almost_equal(z, y)


def test_resample_regressor_nl():
    """ test regressor resampling on a sine function
    """
    x = np.linspace(0, 10, 1000)
    y = np.linspace(0, 10, 30)
    z = _resample_regressor(np.cos(x), x, y)
    assert_almost_equal(z, np.cos(y), decimal=2)


def test_orthogonalize():
    """ test that the orthogonalization is OK """
    rng = np.random.RandomState(42)
    X = rng.standard_normal(size=(100, 5))
    X = _orthogonalize(X)
    K = np.dot(X.T, X)
    K -= np.diag(np.diag(K))
    assert_almost_equal((K ** 2).sum(), 0, 15)


def test_orthogonalize_trivial():
    """ test that the orthogonalization is OK """
    rng = np.random.RandomState(42)
    X = rng.standard_normal(size=100)
    Y = X.copy()
    X = _orthogonalize(X)
    assert_array_equal(Y, X)


def test_sample_condition_1():
    """ Test that the experimental condition is correctly sampled
    """
    condition = ([1, 20, 36.5], [0, 0, 0], [1, 1, 1])
    frame_times = np.linspace(0, 49, 50)
    reg, rf = _sample_condition(condition, frame_times, oversampling=1,
                                min_onset=0)
    assert reg.sum() == 3
    assert reg[1] == 1
    assert reg[20] == 1
    assert reg[37] == 1

    reg, rf = _sample_condition(condition, frame_times, oversampling=1)
    assert reg.sum() == 3
    assert reg[25] == 1
    assert reg[44] == 1
    assert reg[61] == 1


def test_sample_condition_2():
    """ Test the experimental condition sampling -- onset = 0
    """
    condition = ([0, 20, 36.5], [2, 2, 2], [1, 1, 1])
    frame_times = np.linspace(0, 49, 50)
    reg, rf = _sample_condition(condition, frame_times, oversampling=1,
                                min_onset=- 10)
    assert reg.sum() == 6
    assert reg[10] == 1
    assert reg[48] == 1
    assert reg[31] == 1


def test_sample_condition_3():
    """ Test the experimental condition sampling -- oversampling=10
    """
    condition = ([1, 20, 36.5], [2, 2, 2], [1, 1, 1])
    frame_times = np.linspace(0, 49, 50)
    reg, rf = _sample_condition(condition, frame_times, oversampling=10,
                                min_onset=0)
    assert_almost_equal(reg.sum(), 60.)
    assert reg[10] == 1
    assert reg[380] == 1
    assert reg[210] == 1
    assert np.sum(reg > 0) == 60
    # check robustness to non-int oversampling
    reg_, rf_ = _sample_condition(condition, frame_times, oversampling=10.,
                                  min_onset=0)
    assert_almost_equal(reg, reg_)


def test_sample_condition_4():
    """ Test the experimental condition sampling -- negative amplitude
    """
    condition = ([1, 20, 36.5], [2, 2, 2], [1., -1., 5.])
    frame_times = np.linspace(0, 49, 50)
    reg, rf = _sample_condition(condition, frame_times, oversampling=1)
    assert reg.sum() == 10
    assert reg[25] == 1.
    assert reg[44] == -1.
    assert reg[61] == 5.


def test_sample_condition_5():
    """ Test the experimental condition sampling -- negative onset
    """
    condition = ([-10, 0, 36.5], [2, 2, 2], [1., -1., 5.])
    frame_times = np.linspace(0, 49, 50)
    reg, rf = _sample_condition(condition, frame_times, oversampling=1)
    assert reg.sum() == 10
    assert reg[14] == 1.
    assert reg[24] == -1.
    assert reg[61] == 5.


def test_sample_condition_6():
    """ Test the experimental condition sampling -- overalapping onsets,
    different durations
    """
    condition = ([0, 0, 10], [1, 2, 1], [1., 1., 1.])
    frame_times = np.linspace(0, 49, 50)
    reg, rf = _sample_condition(condition, frame_times, oversampling=1)
    assert reg.sum() == 4
    assert reg[24] == 2.
    assert reg[34] == 1.
    assert reg[61] == 0.


def test_sample_condition_7():
    """ Test the experimental condition sampling -- different onsets,
    overlapping offsets
    """
    condition = ([0, 10, 20], [11, 1, 1], [1., 1., 1.])
    frame_times = np.linspace(0, 49, 50)
    reg, rf = _sample_condition(condition, frame_times, oversampling=1)
    assert reg.sum() == 13
    assert reg[24] == 1.
    assert reg[34] == 2.
    assert reg[61] == 0.


def test_names():
    """ Test the regressor naming function
    """
    name = 'con'
    assert _regressor_names(name, 'spm') == [name]
    assert _regressor_names(
        name, 'spm + derivative') == [name, f'{name}_derivative']
    assert _regressor_names(
        name, 'spm + derivative + dispersion') == [name,
                                                   f'{name}_derivative',
                                                   f'{name}_dispersion']
    assert _regressor_names(name, 'glover') == [name]
    assert _regressor_names(
        name, 'glover + derivative') == [name, f'{name}_derivative']
    assert _regressor_names(
        name, 'glover + derivative + dispersion') == [name,
                                                      f'{name}_derivative',
                                                      f'{name}_dispersion']

    assert _regressor_names(name, None) == [name]
    assert _regressor_names(name, [None, None]) == [f"{name}_0", f"{name}_1"]
    assert _regressor_names(name, "typo") == [name]
    assert _regressor_names(name, ["typo", "typo"]) == \
        [f"{name}_0", f"{name}_1"]

    def custom_rf(tr, ov):
        return np.ones(int(tr * ov))

    assert _regressor_names(name, custom_rf) == \
        [f"{name}_{custom_rf.__name__}"]
    assert _regressor_names(name, [custom_rf]) == \
        [f"{name}_{custom_rf.__name__}"]
    assert _regressor_names(name, lambda tr, ov: np.ones(int(tr * ov))) == \
        [f"{name}_lambda"]
    assert _regressor_names(name, [lambda tr, ov: np.ones(int(tr * ov))]) == \
        [f"{name}_lambda"]

    with pytest.raises(ValueError,
                       match="Computed regressor names are not unique"):
        _regressor_names(name, [
            lambda tr, ov: np.ones(int(tr * ov)),
            lambda tr, ov: np.ones(int(tr * ov))
        ])


def test_hkernel():
    """ test the hrf computation
    """
    tr = 2.0
    h = _hrf_kernel('spm', tr)
    assert_almost_equal(h[0], spm_hrf(tr))
    assert len(h) == 1
    h = _hrf_kernel('spm + derivative', tr)
    assert_almost_equal(h[1], spm_time_derivative(tr))
    assert len(h) == 2
    h = _hrf_kernel('spm + derivative + dispersion', tr)
    assert_almost_equal(h[2], spm_dispersion_derivative(tr))
    assert len(h) == 3
    h = _hrf_kernel('glover', tr)
    assert_almost_equal(h[0], glover_hrf(tr))
    assert len(h) == 1
    h = _hrf_kernel('glover + derivative', tr)
    assert_almost_equal(h[1], glover_time_derivative(tr))
    assert_almost_equal(h[0], glover_hrf(tr))
    assert len(h) == 2
    h = _hrf_kernel('glover + derivative + dispersion', tr)
    assert len(h) == 3
    assert_almost_equal(h[2], glover_dispersion_derivative(tr))
    assert_almost_equal(h[1], glover_time_derivative(tr))
    assert_almost_equal(h[0], glover_hrf(tr))
    h = _hrf_kernel('fir', tr, fir_delays=np.arange(4))
    assert len(h) == 4
    for dh in h:
        assert_almost_equal(dh.sum(), 1.)
    h = _hrf_kernel(None, tr)
    assert len(h) == 1
    assert_almost_equal(h[0], np.hstack((1, np.zeros(49))))
    with pytest.raises(ValueError,
                       match="Could not process custom HRF model provided."):
        _hrf_kernel(lambda x: np.ones(int(x)), tr)
        _hrf_kernel([lambda x, y, z: x + y + z], tr)
        _hrf_kernel([lambda x: np.ones(int(x))] * 2, tr)
    h = _hrf_kernel(lambda tr, ov: np.ones(int(tr * ov)), tr)
    assert len(h) == 1
    assert_almost_equal(h[0], np.ones(100))
    h = _hrf_kernel([lambda tr, ov: np.ones(int(tr * ov))], tr)
    assert len(h) == 1
    assert_almost_equal(h[0], np.ones(100))
    with pytest.raises(ValueError,
                       match="is not a known hrf model."):
        _hrf_kernel("foo", tr)


def test_make_regressor_1():
    """ test the generated regressor
    """
    condition = ([1, 20, 36.5], [2, 2, 2], [1, 1, 1])
    frame_times = np.linspace(0, 69, 70)
    hrf_model = 'spm'
    reg, reg_names = compute_regressor(condition, hrf_model, frame_times)
    assert_almost_equal(reg.sum(), 6, 1)
    assert reg_names[0] == 'cond'


def test_make_regressor_2():
    """ test the generated regressor
    """
    condition = ([1, 20, 36.5], [0, 0, 0], [1, 1, 1])
    frame_times = np.linspace(0, 69, 70)
    hrf_model = 'spm'
    reg, reg_names = compute_regressor(condition, hrf_model, frame_times)
    assert_almost_equal(reg.sum() * 50, 3, 1)
    assert reg_names[0] == 'cond'


def test_make_regressor_3():
    """ test the generated regressor
    """
    condition = ([1, 20, 36.5], [2, 2, 2], [1, 1, 1])
    frame_times = np.linspace(0, 138, 70)
    hrf_model = 'fir'
    reg, reg_names = compute_regressor(condition, hrf_model, frame_times,
                                       fir_delays=np.arange(4))
    assert_array_almost_equal(np.sum(reg, 0), np.array([3, 3, 3, 3]))
    assert len(reg_names) == 4
    reg_, reg_names_ = compute_regressor(condition, hrf_model, frame_times,
                                         fir_delays=np.arange(4),
                                         oversampling=50.)
    assert_array_equal(reg, reg_)


def test_design_warnings():
    """
    test that warnings are correctly raised upon weird design specification
    """
    condition = ([-25, 20, 36.5], [0, 0, 0], [1, 1, 1])
    frame_times = np.linspace(0, 69, 70)
    hrf_model = 'spm'
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        with pytest.warns(UserWarning):
            compute_regressor(condition, hrf_model, frame_times)
    condition = ([-25, -25, 36.5], [0, 0, 0], [1, 1, 1])
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        with pytest.warns(UserWarning):
            compute_regressor(condition, hrf_model, frame_times)
