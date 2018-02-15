from nistats.design_matrix import make_design_matrix
from nistats.reporting import plot_design_matrix, get_clusters_table
import nibabel as nib
import numpy as np
from numpy.testing import dec
from nose.tools import assert_true

# Set the backend to avoid having DISPLAY problems
from nilearn.plotting import _set_mpl_backend
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


@dec.skipif(not have_mpl)
def test_show_design_matrix():
    # test that the show code indeed (formally) runs
    frame_times = np.linspace(0, 127 * 1., 128)
    DM = make_design_matrix(
        frame_times, drift_model='polynomial', drift_order=3)
    ax = plot_design_matrix(DM)
    assert (ax is not None)


def test_get_clusters_table():
    shape = (9, 10, 11)
    data = np.zeros(shape)
    data[2:4, 5:7, 6:8] = 5.
    stat_img = nib.Nifti1Image(data, np.eye(4))

    # test one cluster extracted
    cluster_table = get_clusters_table(stat_img, 4, 0)
    assert_true(len(cluster_table) == 1)

    # test empty table on high stat threshold
    cluster_table = get_clusters_table(stat_img, 6, 0)
    assert_true(len(cluster_table) == 0)

    # test empty table on high cluster threshold
    cluster_table = get_clusters_table(stat_img, 4, 9)
    assert_true(len(cluster_table) == 0)
