import nibabel as nib
import numpy as np
import pytest

from nilearn.image import get_data

# Set backend to avoid DISPLAY problems
from nilearn.plotting import _set_mpl_backend
from nilearn.reporting import get_clusters_table
from nilearn.reporting._get_clusters_table import (
    _cluster_nearest_neighbor,
    _local_max,
)

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


@pytest.mark.skipif(
    not have_mpl, reason="Matplotlib not installed; required for this test"
)
def test_local_max():
    """Basic test of nilearn.reporting._get_clusters_table._local_max()"""
    shape = (9, 10, 11)
    # Two maxima (one global, one local), 10 voxels apart.
    data = np.zeros(shape)
    data[4, 5, :] = [4, 3, 2, 1, 1, 1, 1, 1, 2, 3, 4]
    data[5, 5, :] = [5, 4, 3, 2, 1, 1, 1, 2, 3, 4, 6]
    data[6, 5, :] = [4, 3, 2, 1, 1, 1, 1, 1, 2, 3, 4]
    affine = np.eye(4)

    ijk, vals = _local_max(data, affine, min_distance=9)
    assert np.array_equal(ijk, np.array([[5.0, 5.0, 10.0], [5.0, 5.0, 0.0]]))
    assert np.array_equal(vals, np.array([6, 5]))

    ijk, vals = _local_max(data, affine, min_distance=11)
    assert np.array_equal(ijk, np.array([[5.0, 5.0, 10.0]]))
    assert np.array_equal(vals, np.array([6]))

    # Two global (equal) maxima, 10 voxels apart.
    data = np.zeros(shape)
    data[4, 5, :] = [4, 3, 2, 1, 1, 1, 1, 1, 2, 3, 4]
    data[5, 5, :] = [5, 4, 3, 2, 1, 1, 1, 2, 3, 4, 5]
    data[6, 5, :] = [4, 3, 2, 1, 1, 1, 1, 1, 2, 3, 4]
    affine = np.eye(4)

    ijk, vals = _local_max(data, affine, min_distance=9)
    assert np.array_equal(ijk, np.array([[5.0, 5.0, 0.0], [5.0, 5.0, 10.0]]))
    assert np.array_equal(vals, np.array([5, 5]))

    ijk, vals = _local_max(data, affine, min_distance=11)
    assert np.array_equal(ijk, np.array([[5.0, 5.0, 0.0]]))
    assert np.array_equal(vals, np.array([5]))

    # A donut.
    data = np.zeros(shape)
    data[4, 5, :] = [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0]
    data[5, 5, :] = [0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0]
    data[6, 5, :] = [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0]
    affine = np.eye(4)

    ijk, vals = _local_max(data, affine, min_distance=9)
    assert np.array_equal(ijk, np.array([[4.0, 5.0, 5.0]]))
    assert np.array_equal(vals, np.array([1]))


def test_cluster_nearest_neighbor():
    """Check that _cluster_nearest_neighbor preserves within-cluster voxels,
    projects voxels to the correct cluster, and handles singleton clusters.
    """
    shape = (9, 10, 11)
    labeled = np.zeros(shape)
    # cluster 1 is half the volume, cluster 2 is a single voxel
    labeled[:, 5:, :] = 1
    labeled[4, 2, 6] = 2

    labels_index = np.array([1, 1, 2])
    ijk = np.array(
        [
            [4, 7, 5],  # inside cluster 1
            [4, 2, 5],  # outside, close to 2
            [4, 3, 6],  # outside, close to 2
        ]
    )
    nbrs = _cluster_nearest_neighbor(ijk, labels_index, labeled)
    assert np.array_equal(nbrs, np.array([[4, 7, 5], [4, 5, 5], [4, 2, 6]]))


def test_get_clusters_table(tmp_path):
    shape = (9, 10, 11)
    data = np.zeros(shape)
    data[2:4, 5:7, 6:8] = 5.0
    data[4:6, 7:9, 8:10] = -5.0
    stat_img = nib.Nifti1Image(data, np.eye(4))

    # test one cluster extracted
    cluster_table = get_clusters_table(stat_img, 4, 0, two_sided=False)
    assert len(cluster_table) == 1

    # test empty table on high stat threshold
    cluster_table = get_clusters_table(stat_img, 6, 0, two_sided=False)
    assert len(cluster_table) == 0

    # test empty table on high cluster threshold
    cluster_table = get_clusters_table(stat_img, 4, 9, two_sided=False)
    assert len(cluster_table) == 0

    # test two clusters with different signs extracted
    cluster_table = get_clusters_table(stat_img, 4, 0, two_sided=True)
    assert len(cluster_table) == 2

    # test empty table on high stat threshold
    cluster_table = get_clusters_table(stat_img, 6, 0, two_sided=True)
    assert len(cluster_table) == 0

    # test empty table on high cluster threshold
    cluster_table = get_clusters_table(stat_img, 4, 9, two_sided=True)
    assert len(cluster_table) == 0

    # test with filename
    fname = str(tmp_path / "stat_img.nii.gz")
    stat_img.to_filename(fname)
    cluster_table = get_clusters_table(fname, 4, 0, two_sided=True)
    assert len(cluster_table) == 2

    # test with returning label maps
    cluster_table, label_maps = get_clusters_table(
        stat_img,
        4,
        0,
        two_sided=True,
        return_label_maps=True,
    )
    label_map_positive_data = label_maps[0].get_fdata()
    label_map_negative_data = label_maps[1].get_fdata()
    # make sure positive and negative clusters are returned in the label maps
    assert np.sum(label_map_positive_data[2:4, 5:7, 6:8] != 0) == 8
    assert np.sum(label_map_negative_data[4:6, 7:9, 8:10] != 0) == 8

    # test with extra dimension
    data_extra_dim = data[..., np.newaxis]
    stat_img_extra_dim = nib.Nifti1Image(data_extra_dim, np.eye(4))
    cluster_table = get_clusters_table(
        stat_img_extra_dim,
        4,
        0,
        two_sided=True,
    )
    assert len(cluster_table) == 2

    # Test that nans are handled correctly (No numpy axis errors are raised)
    data[data == 0] = np.nan
    stat_img_nans = nib.Nifti1Image(data, affine=np.eye(4))
    cluster_table = get_clusters_table(stat_img_nans, 1e-2, 0, two_sided=False)
    assert len(cluster_table) == 1

    # Test that subpeaks are handled correctly for len(subpeak_vals) > 1
    # 1 cluster and two subpeaks, 10 voxels apart.
    data = np.zeros(shape)
    data[4, 5, :] = [4, 3, 2, 1, 1, 1, 1, 1, 2, 3, 4]
    data[5, 5, :] = [5, 4, 3, 2, 1, 1, 1, 2, 3, 4, 6]
    data[6, 5, :] = [4, 3, 2, 1, 1, 1, 1, 1, 2, 3, 4]
    stat_img = nib.Nifti1Image(data, np.eye(4))

    cluster_table = get_clusters_table(stat_img, 0, 0, min_distance=9)
    assert len(cluster_table) == 2
    assert 1 in cluster_table["Cluster ID"].values
    assert "1a" in cluster_table["Cluster ID"].values


def test_get_clusters_table_relabel_label_maps():
    """Check that the cluster's labels in label_maps match their corresponding
    cluster IDs in the clusters table.
    """
    shape = (9, 10, 11)
    data = np.zeros(shape)
    data[2:4, 5:7, 6:8] = 6.0
    data[5:7, 7:9, 7:9] = 5.5
    data[0:3, 0:3, 0:3] = 5.0
    stat_img = nib.Nifti1Image(data, np.eye(4))

    cluster_table, label_maps = get_clusters_table(
        stat_img,
        4,
        0,
        return_label_maps=True,
    )

    # Get cluster ids from clusters table
    cluster_ids = cluster_table["Cluster ID"].to_numpy()

    # Find the cluster ids in the label map using the coords from the table.
    coords = cluster_table[["X", "Y", "Z"]].to_numpy().astype(int)
    lb_cluster_ids = label_maps[0].get_fdata()[tuple(coords.T)]

    assert np.array_equal(cluster_ids, lb_cluster_ids)


def test_get_clusters_table_not_modifying_stat_image():
    shape = (9, 10, 11)
    data = np.zeros(shape)
    data[2:4, 5:7, 6:8] = 5.0
    data[0:3, 0:3, 0:3] = 6.0

    stat_img = nib.Nifti1Image(data, np.eye(4))
    data_orig = get_data(stat_img).copy()

    # test one cluster should be removed
    clusters_table = get_clusters_table(
        stat_img,
        4,
        cluster_threshold=10,
        two_sided=True,
    )
    assert np.allclose(data_orig, get_data(stat_img))
    assert len(clusters_table) == 1

    # test no clusters should be removed
    stat_img = nib.Nifti1Image(data, np.eye(4))
    clusters_table = get_clusters_table(
        stat_img,
        4,
        cluster_threshold=7,
        two_sided=False,
    )
    assert np.allclose(data_orig, get_data(stat_img))
    assert len(clusters_table) == 2

    # test cluster threshold is None
    stat_img = nib.Nifti1Image(data, np.eye(4))
    clusters_table = get_clusters_table(
        stat_img,
        4,
        cluster_threshold=None,
        two_sided=False,
    )
    assert np.allclose(data_orig, get_data(stat_img))
    assert len(clusters_table) == 2
