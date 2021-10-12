"""
Tests for private function
nilearn.plotting.img_plotting._get_colorbar_and_data_ranges.
"""

import pytest
import numpy as np
from nilearn.plotting.img_plotting import _get_colorbar_and_data_ranges


@pytest.fixture
def data():
    """Data with positive and negative range.""" 
    return np.array([[-.5, 1., np.nan],
                     [0., np.nan, -.2],
                     [1.5, 2.5, 3.]])


@pytest.fixture
def data_pos():
    """Data with positive range."""
    return np.array([[0, 1., np.nan],
                     [0., np.nan, 0],
                     [1.5, 2.5, 3.]])


@pytest.fixture
def data_neg():
    """Data with negative range."""
    return np.array([[-.5, 0, np.nan],
                     [0., np.nan, -.2],
                     [0, 0, 0]])


def test_get_colorbar_and_data_ranges_with_vmin(data):
    with pytest.raises(ValueError,
                       match='does not accept a "vmin" argument'):
        _get_colorbar_and_data_ranges(data, vmax=None, symmetric_cbar=True,
                                      kwargs={'vmin': 1.})


def test_get_colorbar_and_data_ranges_pos_neg(data):
    # Reasonable additional arguments that would end up being passed
    # to imshow in a real plotting use case
    kwargs = {'aspect': 'auto', 'alpha': 0.9}

    # symmetric_cbar set to True
    cbar_vmin, cbar_vmax, vmin, vmax = _get_colorbar_and_data_ranges(
        data, vmax=None, symmetric_cbar=True, kwargs=kwargs
    )
    assert vmin == -np.nanmax(data)
    assert vmax == np.nanmax(data)
    assert cbar_vmin == None
    assert cbar_vmax == None
    # same case if vmax has been set
    cbar_vmin, cbar_vmax, vmin, vmax = _get_colorbar_and_data_ranges(
        data, vmax=2, symmetric_cbar=True, kwargs=kwargs
    )
    assert vmin == -2
    assert vmax == 2
    assert cbar_vmin == None
    assert cbar_vmax == None
    # symmetric_cbar is set to False
    cbar_vmin, cbar_vmax, vmin, vmax = _get_colorbar_and_data_ranges(
        data, vmax=None, symmetric_cbar=False, kwargs=kwargs
    )
    assert vmin == -np.nanmax(data)
    assert vmax == np.nanmax(data)
    assert cbar_vmin == np.nanmin(data)
    assert cbar_vmax == np.nanmax(data)
    # same case if vmax has been set
    cbar_vmin, cbar_vmax, vmin, vmax = _get_colorbar_and_data_ranges(
        data, vmax=2, symmetric_cbar=False, kwargs=kwargs
    )
    assert vmin == -2
    assert vmax == 2
    assert cbar_vmin == np.nanmin(data)
    assert cbar_vmax == np.nanmax(data)
    # symmetric_cbar is set to 'auto', same behaviours as True for this case
    cbar_vmin, cbar_vmax, vmin, vmax = _get_colorbar_and_data_ranges(
        data, vmax=None, symmetric_cbar='auto', kwargs=kwargs
    )
    assert vmin == -np.nanmax(data)
    assert vmax == np.nanmax(data)
    assert cbar_vmin == None
    assert cbar_vmax == None
    # same case if vmax has been set
    cbar_vmin, cbar_vmax, vmin, vmax = _get_colorbar_and_data_ranges(
        data, vmax=2, symmetric_cbar='auto', kwargs=kwargs
    )
    assert vmin == -2
    assert vmax == 2
    assert cbar_vmin == None
    assert cbar_vmax == None


def test_get_colorbar_and_data_ranges_pos(data_pos):
    # symmetric_cbar set to True
    cbar_vmin, cbar_vmax, vmin, vmax = _get_colorbar_and_data_ranges(
        data_pos, vmax=None, symmetric_cbar=True, kwargs={}
    )
    assert vmin == -np.nanmax(data_pos)
    assert vmax == np.nanmax(data_pos)
    assert cbar_vmin == None
    assert cbar_vmax == None
    # same case if vmax has been set
    cbar_vmin, cbar_vmax, vmin, vmax = _get_colorbar_and_data_ranges(
        data_pos, vmax=2, symmetric_cbar=True, kwargs={}
    )
    assert vmin == -2
    assert vmax == 2
    assert cbar_vmin == None
    assert cbar_vmax == None
    # symmetric_cbar is set to False
    cbar_vmin, cbar_vmax, vmin, vmax = _get_colorbar_and_data_ranges(
        data_pos, vmax=None, symmetric_cbar=False, kwargs={}
    )
    assert vmin == -np.nanmax(data_pos)
    assert vmax == np.nanmax(data_pos)
    assert cbar_vmin == 0
    assert cbar_vmax == None
    # same case if vmax has been set
    cbar_vmin, cbar_vmax, vmin, vmax = _get_colorbar_and_data_ranges(
        data_pos, vmax=2, symmetric_cbar=False, kwargs={}
    )
    assert vmin == -2
    assert vmax == 2
    assert cbar_vmin == 0
    assert cbar_vmax == None
    # symmetric_cbar is set to 'auto', same behaviour as false in this case
    cbar_vmin, cbar_vmax, vmin, vmax = _get_colorbar_and_data_ranges(
        data_pos, vmax=None, symmetric_cbar='auto', kwargs={}
    )
    assert vmin == -np.nanmax(data_pos)
    assert vmax == np.nanmax(data_pos)
    assert cbar_vmin == 0
    assert cbar_vmax == None
    # same case if vmax has been set
    cbar_vmin, cbar_vmax, vmin, vmax = _get_colorbar_and_data_ranges(
        data_pos, vmax=2, symmetric_cbar='auto', kwargs={}
    )
    assert vmin == -2
    assert vmax == 2
    assert cbar_vmin == 0
    assert cbar_vmax == None


def test_get_colorbar_and_data_ranges_neg(data_neg):
    # symmetric_cbar set to True
    cbar_vmin, cbar_vmax, vmin, vmax = _get_colorbar_and_data_ranges(
        data_neg, vmax=None, symmetric_cbar=True, kwargs={}
    )
    assert vmin == np.nanmin(data_neg)
    assert vmax == -np.nanmin(data_neg)
    assert cbar_vmin == None
    assert cbar_vmax == None
    # same case if vmax has been set
    cbar_vmin, cbar_vmax, vmin, vmax = _get_colorbar_and_data_ranges(
        data_neg, vmax=2, symmetric_cbar=True, kwargs={}
    )
    assert vmin == -2
    assert vmax == 2
    assert cbar_vmin == None
    assert cbar_vmax == None
    # symmetric_cbar is set to False
    cbar_vmin, cbar_vmax, vmin, vmax = _get_colorbar_and_data_ranges(
        data_neg, vmax=None, symmetric_cbar=False, kwargs={}
    )
    assert vmin == np.nanmin(data_neg)
    assert vmax == -np.nanmin(data_neg)
    assert cbar_vmin == None
    assert cbar_vmax == 0
    # same case if vmax has been set
    cbar_vmin, cbar_vmax, vmin, vmax = _get_colorbar_and_data_ranges(
        data_neg, vmax=2, symmetric_cbar=False, kwargs={}
    )
    assert vmin == -2
    assert vmax == 2
    assert cbar_vmin == None
    assert cbar_vmax == 0
    # symmetric_cbar is set to 'auto', same behaviour as False in this case
    cbar_vmin, cbar_vmax, vmin, vmax = _get_colorbar_and_data_ranges(
        data_neg, vmax=None, symmetric_cbar='auto', kwargs={}
    )
    assert vmin == np.nanmin(data_neg)
    assert vmax == -np.nanmin(data_neg)
    assert cbar_vmin == None
    assert cbar_vmax == 0
    # same case if vmax has been set
    cbar_vmin, cbar_vmax, vmin, vmax = _get_colorbar_and_data_ranges(
        data_neg, vmax=2, symmetric_cbar='auto', kwargs={}
    )
    assert vmin == -2
    assert vmax == 2
    assert cbar_vmin == None
    assert cbar_vmax == 0


def test_get_colorbar_and_data_ranges_masked_array(data):
    masked_data = np.ma.masked_greater(data, 2.)
    # Easier to fill masked values with NaN to test against later on
    filled_data = masked_data.filled(np.nan)

    # Reasonable additional arguments that would end up being passed
    # to imshow in a real plotting use case
    kwargs = {'aspect': 'auto', 'alpha': 0.9}

    # symmetric_cbar set to True
    cbar_vmin, cbar_vmax, vmin, vmax = _get_colorbar_and_data_ranges(
        masked_data, vmax=None, symmetric_cbar=True, kwargs=kwargs
    )
    assert vmin == -np.nanmax(filled_data)
    assert vmax == np.nanmax(filled_data)
    assert cbar_vmin == None
    assert cbar_vmax == None
    # same case if vmax has been set
    cbar_vmin, cbar_vmax, vmin, vmax = _get_colorbar_and_data_ranges(
        masked_data, vmax=2, symmetric_cbar=True, kwargs=kwargs
    )
    assert vmin == -2
    assert vmax == 2
    assert cbar_vmin == None
    assert cbar_vmax == None
    # symmetric_cbar is set to False
    cbar_vmin, cbar_vmax, vmin, vmax = _get_colorbar_and_data_ranges(
        masked_data, vmax=None, symmetric_cbar=False, kwargs=kwargs
    )
    assert vmin == -np.nanmax(filled_data)
    assert vmax == np.nanmax(filled_data)
    assert cbar_vmin == np.nanmin(filled_data)
    assert cbar_vmax == np.nanmax(filled_data)
    # same case if vmax has been set
    cbar_vmin, cbar_vmax, vmin, vmax = _get_colorbar_and_data_ranges(
        masked_data, vmax=2, symmetric_cbar=False, kwargs=kwargs
    )
    assert vmin == -2
    assert vmax == 2
    assert cbar_vmin == np.nanmin(filled_data)
    assert cbar_vmax == np.nanmax(filled_data)
    # symmetric_cbar is set to 'auto', same behaviours as True for this case
    cbar_vmin, cbar_vmax, vmin, vmax = _get_colorbar_and_data_ranges(
        masked_data, vmax=None, symmetric_cbar='auto', kwargs=kwargs
    )
    assert vmin == -np.nanmax(filled_data)
    assert vmax == np.nanmax(filled_data)
    assert cbar_vmin == None
    assert cbar_vmax == None
    # same case if vmax has been set
    cbar_vmin, cbar_vmax, vmin, vmax = _get_colorbar_and_data_ranges(
        masked_data, vmax=2, symmetric_cbar='auto', kwargs=kwargs
    )
    assert vmin == -2
    assert vmax == 2
    assert cbar_vmin == None
    assert cbar_vmax == None