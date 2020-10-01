import nibabel as nib
import numpy as np
import pytest

# Set backend to avoid DISPLAY problems
from nilearn.plotting import _set_mpl_backend

from nilearn.reporting import (get_clusters_table,
                               )
from nilearn.reporting._get_clusters_table import _local_max

# Avoid making pyflakes unhappy
_set_mpl_backend
try:
    import matplotlib.pyplot
    # Avoid making pyflakes unhappy
    matplotlib.pyplot
except ImportError:
    have_mpl = False
else:
    have_mpl = True


@pytest.mark.skipif(not have_mpl,
                    reason='Matplotlib not installed; required for this test')
def test_local_max():
    shape = (9, 10, 11)
    data = np.zeros(shape)
    # Two maxima (one global, one local), 10 voxels apart.
    data[4, 5, :] = [4, 3, 2, 1, 1, 1, 1, 1, 2, 3, 4]
    data[5, 5, :] = [5, 4, 3, 2, 1, 1, 1, 2, 3, 4, 6]
    data[6, 5, :] = [4, 3, 2, 1, 1, 1, 1, 1, 2, 3, 4]
    affine = np.eye(4)

    ijk, vals = _local_max(data, affine, min_distance=9)
    assert np.array_equal(ijk, np.array([[5., 5., 10.], [5., 5., 0.]]))
    assert np.array_equal(vals, np.array([6, 5]))

    ijk, vals = _local_max(data, affine, min_distance=11)
    assert np.array_equal(ijk, np.array([[5., 5., 10.]]))
    assert np.array_equal(vals, np.array([6]))


def test_get_clusters_table():
    shape = (9, 10, 11)
    data = np.zeros(shape)
    data[2:4, 5:7, 6:8] = 5.
    stat_img = nib.Nifti1Image(data, np.eye(4))

    # test one cluster extracted
    cluster_table = get_clusters_table(stat_img, 4, 0)
    assert len(cluster_table) == 1

    # test empty table on high stat threshold
    cluster_table = get_clusters_table(stat_img, 6, 0)
    assert len(cluster_table) == 0

    # test empty table on high cluster threshold
    cluster_table = get_clusters_table(stat_img, 4, 9)
    assert len(cluster_table) == 0
