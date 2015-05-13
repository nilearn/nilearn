import numpy as np
from nose.tools import raises
from numpy.testing import (
    assert_almost_equal, assert_equal, assert_array_equal, assert_warns)
import warnings

from ..hemodynamic_models import (
    spm_hrf, spm_time_derivative, spm_dispersion_derivative,
    _resample_regressor, _orthogonalize, _sample_condition,
    _regressor_names, _hrf_kernel, glover_hrf, 
    glover_time_derivative, compute_regressor)


def test_spm_hrf():
    """ test that the spm_hrf is correctly normalized and has correct length
    """
    h = spm_hrf(2.0)
    assert_almost_equal(h.sum(), 1)
    assert_equal(len(h), 256)

def test_spm_hrf_derivative():
    """ test that the spm_hrf is correctly normalized and has correct length
    """
    h = spm_time_derivative(2.0)
    assert_almost_equal(h.sum(), 0)
    assert_equal(len(h), 256)
    h = spm_dispersion_derivative(2.0)
    assert_almost_equal(h.sum(), 0)
    assert_equal(len(h), 256)

def test_glover_hrf():
    """ test that the spm_hrf is correctly normalized and has correct length
    """
    h = glover_hrf(2.0)
    assert_almost_equal(h.sum(), 1)
    assert_equal(len(h), 256)

def test_glover_time_derivative():
    """ test that the spm_hrf is correctly normalized and has correct length
    """
    h = glover_time_derivative(2.0)
    assert_almost_equal(h.sum(), 0)
    assert_equal(len(h), 256)
    
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
    """ test that the orthogonalization is OK 
    """
    X = np.random.randn(100, 5)
    X = _orthogonalize(X)
    K = np.dot(X.T, X)
    K -= np.diag(np.diag(K))
    assert_almost_equal((K ** 2).sum(), 0, 15)

def test_orthogonalize_trivial():
    """ test that the orthogonalization is OK 
    """
    X = np.random.randn(100)
    Y = X.copy()
    X = _orthogonalize(X)
    assert_array_equal(Y, X)

def test_sample_condition_1():
    """ Test that the experimental condition is correctly sampled
    """
    condition = ([1, 20, 36.5], [0, 0, 0], [1, 1, 1])
    frametimes = np.linspace(0, 49, 50)
    reg, rf = _sample_condition(condition, frametimes, oversampling=1, 
                               min_onset=0)
    assert_equal(reg.sum(), 3)
    assert_equal(reg[1], 1)
    assert_equal(reg[20], 1)
    assert_equal(reg[37], 1)

    reg, rf = _sample_condition(condition, frametimes, oversampling=1)
    assert_equal(reg.sum(), 3)
    assert_equal(reg[25], 1)
    assert_equal(reg[44], 1)
    assert_equal(reg[61], 1)


def test_sample_condition_2():
    """ Test the experimental condition sampling -- onset = 0
    """
    condition = ([0, 20, 36.5], [2, 2, 2], [1, 1, 1])
    frametimes = np.linspace(0, 49, 50)
    reg, rf = _sample_condition(condition, frametimes, oversampling=1,
                               min_onset=- 10)
    assert_equal(reg.sum(), 6)
    assert_equal(reg[10], 1)
    assert_equal(reg[48], 1)
    assert_equal(reg[31], 1)

def test_sample_condition_3():
    """ Test the experimental condition sampling -- oversampling=10
    """
    condition = ([1, 20, 36.5], [2, 2, 2], [1, 1, 1])
    frametimes = np.linspace(0, 49, 50)
    reg, rf = _sample_condition(condition, frametimes, oversampling=10,
                               min_onset=0)
    assert_almost_equal(reg.sum(), 60.)
    assert_equal(reg[10], 1)
    assert_equal(reg[380], 1)
    assert_equal(reg[210], 1)
    assert_equal(np.sum(reg > 0), 60)

def test_sample_condition_4():
    """ Test the experimental condition sampling -- negative amplitude
    """
    condition = ([1, 20, 36.5], [2, 2, 2], [1., -1., 5.])
    frametimes = np.linspace(0, 49, 50)
    reg, rf = _sample_condition(condition, frametimes, oversampling=1)
    assert_equal(reg.sum(),10)
    assert_equal(reg[25], 1.)
    assert_equal(reg[44], -1.)
    assert_equal(reg[61], 5.)

def test_sample_condition_5():
    """ Test the experimental condition sampling -- negative onset
    """
    condition = ([-10, 0, 36.5], [2, 2, 2], [1., -1., 5.])
    frametimes = np.linspace(0, 49, 50)
    reg, rf = _sample_condition(condition, frametimes, oversampling=1)
    assert_equal(reg.sum(),10)
    assert_equal(reg[14], 1.)
    assert_equal(reg[24], -1.)
    assert_equal(reg[61], 5.)

def test_names():
    """ Test the regressor naming function
    """
    name = 'con'
    assert_equal(_regressor_names(name, 'spm'), ['con'])
    assert_equal(_regressor_names(name, 'spm_time'), ['con', 'con_derivative'])
    assert_equal(_regressor_names(name, 'spm_time_dispersion'),
        ['con', 'con_derivative', 'con_dispersion'])
    assert_equal(_regressor_names(name, 'canonical'), ['con'])
    assert_equal(_regressor_names(name, 'canonical with derivative'),
        ['con', 'con_derivative'])

def test_hkernel():
    """ test the hrf computation
    """
    tr = 2.0
    h = _hrf_kernel('spm', tr)
    assert_almost_equal(h[0], spm_hrf(tr))
    assert_equal(len(h), 1)
    h = _hrf_kernel('spm_time', tr)
    assert_almost_equal(h[1], spm_time_derivative(tr))
    assert_equal(len(h), 2)
    h = _hrf_kernel('spm_time_dispersion', tr)
    assert_almost_equal(h[2], spm_dispersion_derivative(tr))
    assert_equal(len(h), 3)
    h = _hrf_kernel('canonical', tr)
    assert_almost_equal(h[0], glover_hrf(tr))
    assert_equal(len(h), 1)
    h = _hrf_kernel('canonical with derivative', tr)
    assert_almost_equal(h[1], glover_time_derivative(tr))
    assert_almost_equal(h[0], glover_hrf(tr))
    assert_equal(len(h), 2)
    h = _hrf_kernel('fir', tr, fir_delays = np.arange(4))
    assert_equal(len(h), 4)
    for dh in h:
        assert_equal(dh.sum(), 16.)
    
def test_make_regressor_1():
    """ test the generated regressor
    """
    condition = ([1, 20, 36.5], [2, 2, 2], [1, 1, 1])
    frametimes = np.linspace(0, 69, 70)
    hrf_model = 'spm'
    reg, reg_names = compute_regressor(condition, hrf_model, frametimes)
    assert_almost_equal(reg.sum(), 6, 1)
    assert_equal(reg_names[0], 'cond')

def test_make_regressor_2():
    """ test the generated regressor
    """
    condition = ([1, 20, 36.5], [0, 0, 0], [1, 1, 1])
    frametimes = np.linspace(0, 69, 70)
    hrf_model = 'spm'
    reg, reg_names = compute_regressor(condition, hrf_model, frametimes)
    assert_almost_equal(reg.sum() * 16, 3, 1)
    assert_equal(reg_names[0], 'cond')


def test_make_regressor_3():
    """ test the generated regressor
    """
    condition = ([1, 20, 36.5], [0, 0, 0], [1, 1, 1])
    frametimes = np.linspace(0, 138, 70)
    hrf_model = 'fir'
    reg, reg_names = compute_regressor(condition, hrf_model, frametimes, 
                                       fir_delays=np.arange(4))
    assert_array_equal(np.unique(reg), np.array([0, 1]))
    assert_array_equal(np.sum(reg, 0), np.array([3, 3, 3, 3]))
    assert_equal(len(reg_names), 4)

def test_design_warnings():
    """ test that warnings are correctly raised upon weird design specification
    """
    condition = ([-25, 20, 36.5], [0, 0, 0], [1, 1, 1])
    frametimes = np.linspace(0, 69, 70)
    hrf_model = 'spm'
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        assert_warns(UserWarning, compute_regressor, condition, hrf_model, 
                     frametimes)
    condition = ([-25, -25, 36.5], [0, 0, 0], [1, 1, 1])
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        assert_warns(UserWarning, compute_regressor, condition, hrf_model, 
                     frametimes)

if __name__ == "__main__":
    import nose
    nose.run(argv=['', __file__])
