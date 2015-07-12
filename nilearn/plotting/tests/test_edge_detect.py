
import numpy as np

from nilearn.plotting.edge_detect import _edge_detect


###############################################################################

def test_edge_detect():
    img = np.zeros((10, 10))
    img[:5] = 1
    _, edge_mask = _edge_detect(img)
    np.testing.assert_almost_equal(img[4], 1)
