"""
This module implements plotting functions useful to report analysis results.

Author: Martin Perez-Guevara, Elvis Dohmatob, 2017
"""

import warnings
from string import ascii_lowercase

import numpy as np
import pandas as pd
import nibabel as nib
from scipy import ndimage

from nilearn.image import get_data
from nilearn.image.resampling import coord_transform
from nilearn._utils import check_niimg_3d


def _local_max(data, affine, min_distance):
    """Find all local maxima of the array, separated by at least min_distance.
    Adapted from https://stackoverflow.com/a/22631583/2589328

    Parameters
    ----------
    data : array_like
        3D array of with masked values for cluster.

    affine : np.ndarray
        Square matrix specifying the position of the image array data
        in a reference space.

    min_distance : int
        Minimum distance between local maxima in ``data``, in terms of mm.

    Returns
    -------
    ijk : `numpy.ndarray`
        (n_foci, 3) array of local maxima indices for cluster.

    vals : `numpy.ndarray`
        (n_foci,) array of values from data at ijk.

    """
    ijk, vals = _identify_subpeaks(data)
    xyz, ijk, vals = _sort_subpeaks(ijk, vals, affine)
    ijk, vals = _pare_subpeaks(xyz, ijk, vals, min_distance)
    return ijk, vals


def _identify_subpeaks(data):
    """Identify cluster peak and subpeaks based on minimum distance.

    Parameters
    ----------
    data : `numpy.ndarray`
        3D array of with masked values for cluster.

    Returns
    -------
    ijk : `numpy.ndarray`
        (n_foci, 3) array of local maxima indices for cluster.
    vals : `numpy.ndarray`
        (n_foci,) array of values from data at ijk.

    Notes
    -----
    When a cluster's local maxima correspond to contiguous voxels with the
    same values (as in a binary cluster), this function determines the center
    of mass for those voxels.
    """
    # Initial identification of subpeaks with minimal minimum distance
    data_max = ndimage.filters.maximum_filter(data, 3)
    maxima = data == data_max
    data_min = ndimage.filters.minimum_filter(data, 3)
    diff = (data_max - data_min) > 0
    maxima[diff == 0] = 0

    labeled, n_subpeaks = ndimage.label(maxima)
    labels_index = range(1, n_subpeaks + 1)
    ijk = np.array(ndimage.center_of_mass(data, labeled, labels_index))
    ijk = np.round(ijk).astype(int)
    vals = np.apply_along_axis(
        arr=ijk, axis=1, func1d=_get_val, input_arr=data
    )
    # Determine if all subpeaks are within the cluster
    # They may not be if the cluster is binary and has a shape where the COM is
    # outside the cluster, like a donut.
    cluster_idx = np.vstack(np.where(labeled)).T.tolist()
    subpeaks_outside_cluster = [
        i
        for i, peak_idx in enumerate(ijk.tolist())
        if peak_idx not in cluster_idx
    ]
    vals[subpeaks_outside_cluster] = np.nan
    if subpeaks_outside_cluster:
        warnings.warn(
            "Attention: At least one of the (sub)peaks falls outside of the "
            "cluster body."
        )
    return ijk, vals


def _sort_subpeaks(ijk, vals, affine):
    # Sort subpeaks in cluster in descending order of stat value
    order = (-vals).argsort()
    vals = vals[order]
    ijk = ijk[order, :]
    xyz = nib.affines.apply_affine(affine, ijk)  # Convert to xyz in mm
    return xyz, ijk, vals


def _pare_subpeaks(xyz, ijk, vals, min_distance):
    # Reduce list of subpeaks based on distance
    keep_idx = np.ones(xyz.shape[0]).astype(bool)
    for i in range(xyz.shape[0]):
        for j in range(i + 1, xyz.shape[0]):
            if keep_idx[i] == 1:
                dist = np.linalg.norm(xyz[i, :] - xyz[j, :])
                keep_idx[j] = dist > min_distance
    ijk = ijk[keep_idx, :]
    vals = vals[keep_idx]
    return ijk, vals


def _get_val(row, input_arr):
    """Small function for extracting values from array based on index.
    """
    i, j, k = row
    return input_arr[i, j, k]


def get_clusters_table(stat_img, stat_threshold, cluster_threshold=None,
                       two_sided=False, min_distance=8.):
    """Creates pandas dataframe with img cluster statistics.

    Parameters
    ----------
    stat_img : Niimg-like object,
       Statistical image (presumably in z- or p-scale).

    stat_threshold : `float`
        Cluster forming threshold in same scale as `stat_img` (either a
        p-value or z-scale value).

    cluster_threshold : `int` or `None`, optional
        Cluster size threshold, in voxels.

    two_sided : `bool`, optional
        Whether to employ two-sided thresholding or to evaluate positive values
        only. Default=False.

    min_distance : `float`, optional
        Minimum distance between subpeaks in mm. Default=8mm.

    Returns
    -------
    df : `pandas.DataFrame`
        Table with peaks and subpeaks from thresholded `stat_img`. For binary
        clusters (clusters with >1 voxel containing only one value), the table
        reports the center of mass of the cluster,
        rather than any peaks/subpeaks.

    """
    cols = ['Cluster ID', 'X', 'Y', 'Z', 'Peak Stat', 'Cluster Size (mm3)']

    # check that stat_img is niimg-like object and 3D
    stat_img = check_niimg_3d(stat_img)
    # If cluster threshold is used, there is chance that stat_map will be
    # modified, therefore copy is needed
    if cluster_threshold is None:
        stat_map = get_data(stat_img)
    else:
        stat_map = get_data(stat_img).copy()

    # Define array for 6-connectivity, aka NN1 or "faces"
    conn_mat = np.zeros((3, 3, 3), int)
    conn_mat[1, 1, :] = 1
    conn_mat[1, :, 1] = 1
    conn_mat[:, 1, 1] = 1
    voxel_size = np.prod(stat_img.header.get_zooms())

    signs = [1, -1] if two_sided else [1]
    no_clusters_found = True
    rows = []
    for sign in signs:
        # Flip map if necessary
        temp_stat_map = stat_map * sign

        # Binarize using CDT
        binarized = temp_stat_map > stat_threshold
        binarized = binarized.astype(int)

        # If the stat threshold is too high simply return an empty dataframe
        if np.sum(binarized) == 0:
            warnings.warn(
                'Attention: No clusters with stat {0} than {1}'.format(
                    'higher' if sign == 1 else 'lower',
                    stat_threshold * sign,
                )
            )
            continue

        # Extract connected components above cluster size threshold
        label_map = ndimage.measurements.label(binarized, conn_mat)[0]
        clust_ids = sorted(list(np.unique(label_map)[1:]))
        for c_val in clust_ids:
            if cluster_threshold is not None and np.sum(
                    label_map == c_val) < cluster_threshold:
                temp_stat_map[label_map == c_val] = 0
                binarized[label_map == c_val] = 0

        # If the cluster threshold is too high simply return an empty dataframe
        # this checks for stats higher than threshold after small clusters
        # were removed from temp_stat_map
        if np.sum(temp_stat_map > stat_threshold) == 0:
            warnings.warn(
                'Attention: No clusters with more than {0} voxels'.format(
                    cluster_threshold,
                )
            )
            continue

        # Now re-label and create table
        label_map = ndimage.measurements.label(binarized, conn_mat)[0]
        clust_ids = sorted(list(np.unique(label_map)[1:]))
        peak_vals = np.array(
            [np.max(temp_stat_map * (label_map == c)) for c in clust_ids])
        # Sort by descending max value
        clust_ids = [clust_ids[c] for c in (-peak_vals).argsort()]

        for c_id, c_val in enumerate(clust_ids):
            cluster_mask = label_map == c_val
            masked_data = temp_stat_map * cluster_mask

            cluster_size_mm = int(np.sum(cluster_mask) * voxel_size)

            # Get peaks, subpeaks and associated statistics
            subpeak_ijk, subpeak_vals = _local_max(
                masked_data,
                stat_img.affine,
                min_distance=min_distance,
            )
            subpeak_vals *= sign  # flip signs if necessary
            subpeak_xyz = np.asarray(
                coord_transform(
                    subpeak_ijk[:, 0],
                    subpeak_ijk[:, 1],
                    subpeak_ijk[:, 2],
                    stat_img.affine,
                )
            ).tolist()
            subpeak_xyz = np.array(subpeak_xyz).T

            # Only report peak and, at most, top 3 subpeaks.
            n_subpeaks = np.min((len(subpeak_vals), 4))
            for subpeak in range(n_subpeaks):
                if subpeak == 0:
                    row = [
                        c_id + 1,
                        subpeak_xyz[subpeak, 0],
                        subpeak_xyz[subpeak, 1],
                        subpeak_xyz[subpeak, 2],
                        subpeak_vals[subpeak],
                        cluster_size_mm,
                    ]
                else:
                    # Subpeak naming convention is cluster num+letter:
                    # 1a, 1b, etc
                    sp_id = '{0}{1}'.format(
                        c_id + 1,
                        ascii_lowercase[subpeak - 1],
                    )
                    row = [
                        sp_id,
                        subpeak_xyz[subpeak, 0],
                        subpeak_xyz[subpeak, 1],
                        subpeak_xyz[subpeak, 2],
                        subpeak_vals[subpeak],
                        '',
                    ]
                rows += [row]

        # If we reach this point, there are clusters in this sign
        no_clusters_found = False

    if no_clusters_found:
        df = pd.DataFrame(columns=cols)
    else:
        df = pd.DataFrame(columns=cols, data=rows)

    return df
