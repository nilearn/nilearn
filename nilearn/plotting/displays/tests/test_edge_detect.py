import numpy as np
import pytest

from nilearn.plotting.displays.edge_detect import _edge_detect


@pytest.mark.ai_generated
def test_edge_detect():
    """Test that _edge_detect leaves the input image unchanged."""
    img = np.zeros((10, 10))
    img[:5] = 1
    _, _ = _edge_detect(img)
    np.testing.assert_almost_equal(img[4], 1)


@pytest.mark.ai_generated
def test_edge_nan():
    """Test that _edge_detect handles NaN values in the input image."""
    img = np.zeros((10, 10))
    img[:5] = 1
    img[0] = np.nan
    grad_mag, _ = _edge_detect(img)
    np.testing.assert_almost_equal(img[4], 1)
    assert (grad_mag[0] > 2).all()
