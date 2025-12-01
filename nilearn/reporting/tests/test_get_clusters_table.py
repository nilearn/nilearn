import warnings
from copy import deepcopy

import numpy as np
import pandas as pd
import pytest
from nibabel import Nifti1Image
from numpy.testing import assert_array_equal
from sklearn.utils.estimator_checks import ignore_warnings

from nilearn.datasets import load_fsaverage_data
from nilearn.image import get_data, math_img
from nilearn.reporting.get_clusters_table import (
    _cluster_nearest_neighbor,
    _local_max,
    get_clusters_table,
)
from nilearn.surface.surface import SurfaceImage
from nilearn.surface.surface import get_data as get_surface_data


@pytest.fixture
def shape():
    """Return a shape."""
    return (9, 10, 11)


@pytest.fixture
def simple_stat_img(shape, affine_eye):
    """Create a simple stat image for more tests.

    Contains both positive and negative clusters.
    """
    data = np.zeros(shape)
    data[2:4, 5:7, 6:8] = 5.0
    data[4:6, 7:9, 8:10] = -5.0
    stat_img = Nifti1Image(data, affine_eye)
    return stat_img


def validate_clusters_table(
    clusters_table: pd.DataFrame, expected_n_cluster: int
):
    """Validate the structure of the clusters table."""
    assert len(clusters_table) == expected_n_cluster, clusters_table

    duplicated_ID = clusters_table.duplicated(subset=["Cluster ID"])
    assert not any(duplicated_ID.to_list()), clusters_table

    assert not any(clusters_table["Peak Stat"].to_numpy() == np.nan)

    # VERY unlikely that two different clusters have the same peak stat
    duplicated_stats = clusters_table.duplicated(subset=["Peak Stat"])
    assert not any(duplicated_stats.to_list()), clusters_table


def test_local_max_two_maxima(shape, affine_eye):
    """Basic test of nilearn.reporting._get_clusters_table._local_max()."""
    # Two maxima (one global, one local), 10 voxels apart.
    data = np.zeros(shape)
    data[4, 5, :] = [4, 3, 2, 1, 1, 1, 1, 1, 2, 3, 4]
    data[5, 5, :] = [5, 4, 3, 2, 1, 1, 1, 2, 3, 4, 6]
    data[6, 5, :] = [4, 3, 2, 1, 1, 1, 1, 1, 2, 3, 4]

    ijk, vals = _local_max(data, affine_eye, min_distance=9)
    assert np.array_equal(ijk, np.array([[5.0, 5.0, 10.0], [5.0, 5.0, 0.0]]))
    assert np.array_equal(vals, np.array([6, 5]))

    ijk, vals = _local_max(data, affine_eye, min_distance=11)
    assert np.array_equal(ijk, np.array([[5.0, 5.0, 10.0]]))
    assert np.array_equal(vals, np.array([6]))


def test_local_max_two_global_maxima(shape, affine_eye):
    """Basic test of nilearn.reporting._get_clusters_table._local_max()."""
    # Two global (equal) maxima, 10 voxels apart.
    data = np.zeros(shape)
    data[4, 5, :] = [4, 3, 2, 1, 1, 1, 1, 1, 2, 3, 4]
    data[5, 5, :] = [5, 4, 3, 2, 1, 1, 1, 2, 3, 4, 5]
    data[6, 5, :] = [4, 3, 2, 1, 1, 1, 1, 1, 2, 3, 4]

    ijk, vals = _local_max(data, affine_eye, min_distance=9)
    assert np.array_equal(ijk, np.array([[5.0, 5.0, 0.0], [5.0, 5.0, 10.0]]))
    assert np.array_equal(vals, np.array([5, 5]))

    ijk, vals = _local_max(data, affine_eye, min_distance=11)
    assert np.array_equal(ijk, np.array([[5.0, 5.0, 0.0]]))
    assert np.array_equal(vals, np.array([5]))


def test_local_max_donut(shape, affine_eye):
    """Basic test of nilearn.reporting._get_clusters_table._local_max()."""
    # A donut.
    data = np.zeros(shape)
    data[4, 5, :] = [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0]
    data[5, 5, :] = [0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0]
    data[6, 5, :] = [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0]

    with pytest.warns(UserWarning, match="falls outside of the cluster body."):
        ijk, vals = _local_max(data, affine_eye, min_distance=9)
    assert np.array_equal(ijk, np.array([[4.0, 5.0, 5.0]]))
    assert np.array_equal(vals, np.array([1]))


def test_cluster_nearest_neighbor(shape):
    """Check that _cluster_nearest_neighbor preserves within-cluster voxels, \
       projects voxels to the correct cluster, \
       and handles singleton clusters.
    """
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


@ignore_warnings
@pytest.mark.parametrize(
    "stat_threshold, cluster_threshold, two_sided, expected_n_cluster",
    [
        (4, 0, False, 1),  # test one cluster extracted
        (6, 0, False, 0),  # test empty table on high stat threshold
        (4, 9, False, 0),  # test empty table on high cluster threshold
        (4, 0, True, 2),  # test two clusters with different signs extracted
        (6, 0, True, 0),  # test empty table on high stat threshold
        (4, 9, True, 0),  # test empty table on high cluster threshold
    ],
)
def test_get_clusters_table(
    simple_stat_img,
    stat_threshold,
    cluster_threshold,
    two_sided,
    expected_n_cluster,
):
    """Test several combination of input parameters."""
    clusters_table = get_clusters_table(
        simple_stat_img,
        stat_threshold=stat_threshold,
        cluster_threshold=cluster_threshold,
        two_sided=two_sided,
    )
    validate_clusters_table(clusters_table, expected_n_cluster)


@ignore_warnings
@pytest.mark.parametrize(
    "stat_threshold, cluster_threshold, expected_n_cluster",
    [
        (4, 0, 2),
        (4, 2, 1),
        (6, 0, 0),
        (-4, 0, 2),
        (-4, 2, 0),
        (-6, 0, 0),
    ],
)
def test_get_clusters_table_surface(
    surf_img_1d, stat_threshold, cluster_threshold, expected_n_cluster
):
    """Test n_clusters detected.

    Also check negative thresholds for one sided.
    """
    surf_img_1d.data.parts["left"] = np.asarray([5.1, 5.2, 5.3, -5])
    surf_img_1d.data.parts["right"] = np.asarray([0, 4, 0, 5.4, -5.2])
    stat_img = surf_img_1d

    clusters_table, label_maps = get_clusters_table(
        stat_img,
        stat_threshold=stat_threshold,
        cluster_threshold=cluster_threshold,
        return_label_maps=True,
    )

    validate_clusters_table(clusters_table, expected_n_cluster)

    assert isinstance(label_maps, list)
    assert all(isinstance(x, SurfaceImage) for x in label_maps)
    assert len(label_maps) == 1

    # label_maps should have the correct n_cluster + 1 for background
    cluster_labels = np.unique(get_surface_data(label_maps[0]))
    assert cluster_labels.size == expected_n_cluster + 1


@ignore_warnings
@pytest.mark.parametrize(
    (
        "stat_threshold, cluster_threshold, "
        "expected_n_cluster_left, expected_n_cluster_right, "
        "contain_neg_and_pos"
    ),
    [
        (4, 0, 2, 2, True),
        (4, 2, 1, 0, False),
        (6, 0, 0, 0, False),
    ],
)
def test_get_clusters_table_surface_two_sided(
    surf_img_1d,
    stat_threshold,
    cluster_threshold,
    expected_n_cluster_left,
    expected_n_cluster_right,
    contain_neg_and_pos,
):
    """Test n_clusters detected with two sided."""
    surf_img_1d.data.parts["left"] = np.asarray([5.1, 5.2, 5.3, -5])
    surf_img_1d.data.parts["right"] = np.asarray([0, 4, 0, 5.4, -5.2])
    stat_img = surf_img_1d

    clusters_table, label_maps = get_clusters_table(
        stat_img,
        stat_threshold=stat_threshold,
        cluster_threshold=cluster_threshold,
        two_sided=True,
        return_label_maps=True,
    )

    validate_clusters_table(
        clusters_table, expected_n_cluster_left + expected_n_cluster_right
    )

    if contain_neg_and_pos:
        assert np.any(clusters_table["Peak Stat"].to_numpy() > 0)
        assert np.any(clusters_table["Peak Stat"].to_numpy() < 0)

    assert isinstance(label_maps, list)
    assert all(isinstance(x, SurfaceImage) for x in label_maps)

    # label_maps should have the correct n_cluster + 1 for background
    cluster_labels_positive = np.unique(get_surface_data(label_maps[0]))
    assert cluster_labels_positive.size == expected_n_cluster_left + 1

    cluster_labels_negative = np.unique(get_surface_data(label_maps[1]))
    assert cluster_labels_negative.size == expected_n_cluster_right + 1


@pytest.mark.parametrize(
    "stat_threshold, cluster_threshold, expected_n_cluster_two_sided",
    [
        (1.4, 0, 8),
        (1.4, 10, 4),
    ],
)
def test_get_clusters_table_surface_real_data(
    stat_threshold, cluster_threshold, expected_n_cluster_two_sided
):
    """Test cluster table generation on real surface data.

    Assert that n_clusters two sided equals
    sum of n_clusters one sided \
    with positive and negative threshold.
    """
    stat_img = load_fsaverage_data(mesh_type="inflated")

    clusters_table_two_sided = get_clusters_table(
        stat_img,
        stat_threshold=np.abs(stat_threshold),
        cluster_threshold=cluster_threshold,
        two_sided=True,
    )

    validate_clusters_table(
        clusters_table_two_sided, expected_n_cluster_two_sided
    )

    clusters_table_positive = get_clusters_table(
        stat_img,
        stat_threshold=np.abs(stat_threshold),
        cluster_threshold=cluster_threshold,
        two_sided=False,
    )

    clusters_table_negative = get_clusters_table(
        math_img("img*-1", img=stat_img),
        stat_threshold=stat_threshold,
        cluster_threshold=cluster_threshold,
        two_sided=False,
    )

    assert len(clusters_table_two_sided) == (
        len(clusters_table_positive) + len(clusters_table_negative)
    )


def test_get_clusters_table_surface_min_distance(surf_img_1d, simple_stat_img):
    """Change min_distance parameter raise warning when using surface data."""
    with pytest.warns(
        UserWarning, match="'min_distance' parameter is not used"
    ):
        get_clusters_table(surf_img_1d, stat_threshold=0.1, min_distance=5)

    with warnings.catch_warnings(record=True) as w:
        get_clusters_table(simple_stat_img, stat_threshold=0.1, min_distance=5)
    assert len(w) == 0


def test_get_clusters_table_negative_min_distance_error(simple_stat_img):
    """Check min_distance cannot be negative."""
    with pytest.raises(ValueError, match="'min_distance' must be positive"):
        get_clusters_table(
            simple_stat_img, stat_threshold=0.1, min_distance=-4
        )


def test_get_clusters_table_no_cluster_found_warning(
    surf_img_1d, simple_stat_img
):
    """Check warning is thrown when too high threshold or no cluster found."""
    with pytest.warns(UserWarning, match="But, you have given threshold=1000"):
        clusters_table = get_clusters_table(
            simple_stat_img, stat_threshold=1000
        )
    validate_clusters_table(clusters_table, expected_n_cluster=0)

    with pytest.warns(UserWarning, match="But, you have given threshold=1000"):
        clusters_table = get_clusters_table(surf_img_1d, stat_threshold=1000)
    validate_clusters_table(clusters_table, expected_n_cluster=0)

    with pytest.warns(UserWarning, match="No clusters found"):
        clusters_table = get_clusters_table(
            simple_stat_img, stat_threshold=4.9, cluster_threshold=1000
        )
    validate_clusters_table(clusters_table, expected_n_cluster=0)

    with pytest.warns(UserWarning, match="No clusters found"):
        clusters_table = get_clusters_table(
            surf_img_1d, stat_threshold=1, cluster_threshold=1000
        )
    validate_clusters_table(clusters_table, expected_n_cluster=0)


def test_get_clusters_table_negative_threshold(shape, affine_eye):
    """Check that one sided negative thresholds are handled well."""
    data = np.zeros(shape)
    data[2:4, 5:7, 6:8] = 5.0
    data[4:6, 7:9, 8:10] = -5.0
    stat_img = Nifti1Image(data, affine_eye)

    data_orig = deepcopy(data)

    clusters_table = get_clusters_table(
        stat_img,
        stat_threshold=-1,
        cluster_threshold=0,
        two_sided=False,
    )

    validate_clusters_table(clusters_table, expected_n_cluster=1)

    # sanity check that any sign flip done by get_clusters_table
    # leaves the original data untouched.
    assert_array_equal(stat_img.get_fdata(), data_orig)


def test_get_clusters_table_negative_threshold_one_sided(
    simple_stat_img, surf_img_1d
):
    """Check one sided negative thresholds errors when two_sided=True."""
    with pytest.raises(
        ValueError, match='"threshold" should not be a negative'
    ):
        get_clusters_table(
            surf_img_1d,
            stat_threshold=-1,
            two_sided=True,
        )
    with pytest.raises(
        ValueError, match='"threshold" should not be a negative'
    ):
        get_clusters_table(
            simple_stat_img,
            stat_threshold=-1,
            two_sided=True,
        )


def test_smoke_get_clusters_table_filename(tmp_path, simple_stat_img):
    """Run get_clusters_table on a file."""
    fname = str(tmp_path / "stat_img.nii.gz")
    simple_stat_img.to_filename(fname)
    clusters_table = get_clusters_table(fname, 4, 0, two_sided=True)
    validate_clusters_table(clusters_table, expected_n_cluster=2)


def test_get_clusters_table_4d_image(shape, affine_eye):
    """Run get_clusters_table on 4D image."""
    data = np.zeros((*shape, 1))
    data[2:4, 5:7, 6:8] = 5.0
    data[4:6, 7:9, 8:10] = -5.0
    stat_img = Nifti1Image(data, affine_eye)
    clusters_table = get_clusters_table(
        stat_img,
        4,
        0,
        two_sided=True,
    )
    validate_clusters_table(clusters_table, expected_n_cluster=2)


def test_get_clusters_table_nans(shape, affine_eye):
    """Test nans are handled correctly (No numpy axis errors are raised)."""
    data = np.zeros((*shape, 1))
    data[2:4, 5:7, 6:8] = 5.0
    data[4:6, 7:9, 8:10] = -5.0
    data[data == 0] = np.nan
    stat_img = Nifti1Image(data, affine_eye)
    with pytest.warns(UserWarning, match="Non-finite values detected"):
        clusters_table = get_clusters_table(stat_img, 1e-2, 0, two_sided=False)

    validate_clusters_table(clusters_table, expected_n_cluster=1)


def test_get_clusters_table_subpeaks(shape, affine_eye):
    """Test subpeaks are handled correctly for len(subpeak_vals) > 1."""
    # 1 cluster and two subpeaks, 10 voxels apart.
    data = np.zeros(shape)
    data[4, 5, :] = [4, 3, 2, 1, 1, 1, 1, 1, 2, 3, 4]
    data[5, 5, :] = [5, 4, 3, 2, 1, 1, 1, 2, 3, 4, 6]
    data[6, 5, :] = [4, 3, 2, 1, 1, 1, 1, 1, 2, 3, 4]
    stat_img = Nifti1Image(data, affine_eye)

    clusters_table = get_clusters_table(
        stat_img,
        0,
        0,
        min_distance=9,
    )

    validate_clusters_table(clusters_table, expected_n_cluster=2)
    assert 1 in clusters_table["Cluster ID"].to_numpy()
    assert "1a" in clusters_table["Cluster ID"].to_numpy()


def test_get_clusters_table_relabel_label_maps(simple_stat_img):
    """Check that the cluster's labels in label_maps match \
       their corresponding cluster IDs in the clusters table.
    """
    clusters_table, label_maps = get_clusters_table(
        simple_stat_img,
        4,
        0,
        return_label_maps=True,
    )

    # Get cluster ids from clusters table
    cluster_ids = clusters_table["Cluster ID"].to_numpy()

    # Find the cluster ids in the label map using the coords from the table.
    coords = clusters_table[["X", "Y", "Z"]].to_numpy().astype(int)

    assert len(label_maps) == 1
    lb_cluster_ids = label_maps[0].get_fdata()[tuple(coords.T)]

    assert np.array_equal(cluster_ids, lb_cluster_ids)


def test_get_clusters_table_return_label_maps(simple_stat_img):
    """Test with returning label maps."""
    _, label_maps = get_clusters_table(
        simple_stat_img,
        4,
        0,
        two_sided=True,
        return_label_maps=True,
    )

    assert len(label_maps) == 2

    label_map_positive_data = label_maps[0].get_fdata()
    assert np.sum(label_map_positive_data[2:4, 5:7, 6:8] != 0) == 8
    label_map_negative_data = label_maps[1].get_fdata()
    assert np.sum(label_map_negative_data[4:6, 7:9, 8:10] != 0) == 8


@ignore_warnings
@pytest.mark.parametrize(
    "stat_threshold, cluster_threshold, two_sided, expected_n_cluster",
    [
        (4, 10, True, 1),  # test one cluster should be removed
        (4, 7, False, 2),  # test no clusters should be removed
        (4, 0, False, 2),  # test cluster threshold is 0
    ],
)
def test_get_clusters_table_not_modifying_stat_image(
    affine_eye,
    shape,
    stat_threshold,
    cluster_threshold,
    two_sided,
    expected_n_cluster,
):
    """Make sure original image is not changed."""
    data = np.zeros(shape)
    data[2:4, 5:7, 6:8] = 5.0
    data[0:3, 0:3, 0:3] = 6.0

    stat_img = Nifti1Image(data, affine_eye)
    data_orig = get_data(stat_img).copy()

    clusters_table = get_clusters_table(
        stat_img,
        stat_threshold=stat_threshold,
        cluster_threshold=cluster_threshold,
        two_sided=two_sided,
    )
    validate_clusters_table(clusters_table, expected_n_cluster)
    assert np.allclose(data_orig, get_data(stat_img))
