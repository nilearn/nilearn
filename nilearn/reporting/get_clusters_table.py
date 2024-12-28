"""Implement plotting functions useful to report analysis results.

Author: Martin Perez-Guevara, Elvis Dohmatob, 2017
"""

import warnings
from string import ascii_lowercase

import numpy as np
import pandas as pd
from nibabel import affines
from scipy.ndimage import (
    center_of_mass,
    generate_binary_structure,
    label,
    maximum_filter,
    minimum_filter,
)

from nilearn._utils import check_niimg_3d
from nilearn._utils.niimg import safe_get_data
from nilearn.image import new_img_like, threshold_img
from nilearn.image.resampling import coord_transform


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
        (n_foci, 3) array of local maximum indices for cluster.
    vals : `numpy.ndarray`
        (n_foci,) array of values from data at ijk.

    Notes
    -----
    When a cluster's local maximum corresponds to contiguous voxels with the
    same values (as in a binary cluster), this function determines the center
    of mass for those voxels. If the center of mass falls outside the cluster,
    we instead report the nearest cluster voxel.
    """
    data_max = maximum_filter(data, 3)
    maxima = data == data_max
    zero_mask = data == 0
    maxima[zero_mask] = 0

    # Don't treat constant patches as maxima unless the entire cluster is
    # constant (as in a binary cluster).
    is_constant = np.isclose(data[~zero_mask].max(), data[~zero_mask].min())
    if not is_constant:
        data_min = minimum_filter(data, 3)
        diff = (data_max - data_min) > 0
        maxima[diff == 0] = 0

    labeled, n_subpeaks = label(maxima)
    labels_index = np.arange(1, n_subpeaks + 1)
    ijk = np.array(center_of_mass(data, labeled, labels_index))
    ijk = np.round(ijk).astype(int)
    # Determine if all subpeaks are within the cluster
    # They may not be if the cluster is binary and has a shape where the COM is
    # outside the cluster, like a donut.
    subpeaks_outside_cluster = (
        labeled[ijk[:, 0], ijk[:, 1], ijk[:, 2]] != labels_index
    )
    if np.any(subpeaks_outside_cluster):
        warnings.warn(
            (
                "Attention: At least one of the (sub)peaks "
                "falls outside of the cluster body. "
                "Identifying the nearest in-cluster voxel."
            ),
            stacklevel=4,
        )
        # Replace centers of mass with their nearest neighbor points in the
        # corresponding clusters. Note this is also equivalent to computing the
        # centers of mass constrained to points within the cluster.
        ijk[subpeaks_outside_cluster] = _cluster_nearest_neighbor(
            ijk[subpeaks_outside_cluster],
            labels_index[subpeaks_outside_cluster],
            labeled,
        )
    vals = data[ijk[:, 0], ijk[:, 1], ijk[:, 2]]
    return ijk, vals


def _cluster_nearest_neighbor(ijk, labels_index, labeled):
    """Find the nearest neighbor for given points in the corresponding cluster.

    Parameters
    ----------
    ijk : :obj:`numpy.ndarray`
        (n_pts, 3) array of query points.
    labels_index : :obj:`numpy.ndarray`
        (n_pts,) array of corresponding cluster indices.
    labeled : :obj:`numpy.ndarray`
        3D array with voxels labeled according to cluster index.

    Returns
    -------
    nbrs : :obj:`numpy.ndarray`
        (n_pts, 3) nearest neighbor points.
    """
    labels = labeled[labeled > 0]
    clusters_ijk = np.array(labeled.nonzero()).T
    nbrs = np.zeros_like(ijk)
    for ii, (lab, point) in enumerate(zip(labels_index, ijk)):
        lab_ijk = clusters_ijk[labels == lab]
        dist = np.linalg.norm(lab_ijk - point, axis=1)
        nbrs[ii] = lab_ijk[np.argmin(dist)]
    return nbrs


def _sort_subpeaks(ijk, vals, affine):
    """Sort subpeaks in cluster in descending order of stat value.

    Parameters
    ----------
    ijk : 2D numpy.ndarray
        The matrix indices of subpeaks to sort.
    vals : 1D numpy.ndarray
        The statistical value associated with each subpeak in ``ijk``.
    affine : (4x4) numpy.ndarray
        The affine of the img from which the subpeaks were extracted.
        Used to convert IJK indices to XYZ coordinates.

    Returns
    -------
    xyz : 2D numpy.ndarray
        The sorted coordinates of the subpeaks.
    ijk : 2D numpy.ndarray
        The sorted matrix indices of subpeaks.
    vals : 1D numpy.ndarray
        The sorted statistical value associated with each subpeak in ``ijk``.
    """
    order = (-vals).argsort()
    vals = vals[order]
    ijk = ijk[order, :]
    xyz = affines.apply_affine(affine, ijk)  # Convert to xyz in mm
    return xyz, ijk, vals


def _pare_subpeaks(xyz, ijk, vals, min_distance):
    """Reduce list of subpeaks based on distance.

    Parameters
    ----------
    xyz : 2D numpy.ndarray
        Subpeak coordinates to reduce. Rows correspond to peaks, columns
        correspond to x, y, and z dimensions.
    ijk : 2D numpy.ndarray
        The subpeak coordinates in ``xyz``, but converted to matrix indices.
    vals : 1D numpy.ndarray
        The statistical value associated with each subpeak in ``xyz``/``ijk``.
    min_distance : float
        The minimum distance between subpeaks, in millimeters.

    Returns
    -------
    ijk : 2D numpy.ndarray
        The reduced index of subpeaks.
    vals : 1D numpy.ndarray
        The statistical values associated with the reduced set of subpeaks.
    """
    keep_idx = np.ones(xyz.shape[0]).astype(bool)
    for i in range(xyz.shape[0]):
        for j in range(i + 1, xyz.shape[0]):
            if keep_idx[i] == 1:
                dist = np.linalg.norm(xyz[i, :] - xyz[j, :])
                keep_idx[j] = dist > min_distance
    ijk = ijk[keep_idx, :]
    vals = vals[keep_idx]
    return ijk, vals


def get_clusters_table(
    stat_img,
    stat_threshold,
    cluster_threshold=None,
    two_sided=False,
    min_distance=8.0,
    return_label_maps=False,
):
    """Create pandas dataframe with img cluster statistics.

    This function should work on any statistical maps where more extreme values
    indicate greater statistical significance.
    For example, z-statistic or -log10(p) maps are valid inputs, but a p-value
    map is not.

    .. important::

        For binary clusters (clusters comprised of only one value),
        the table reports the center of mass of the cluster,
        rather than any peaks/subpeaks.

        This center of mass may, in some cases, appear outside of the cluster.

        .. versionchanged:: 0.9.2
            In this case, the cluster voxel nearest to the center of mass is
            reported.

    Parameters
    ----------
    stat_img : Niimg-like object
       Statistical image to threshold and summarize.

    stat_threshold : :obj:`float`
        Cluster forming threshold. This value must be in the same scale as
        ``stat_img``.

    cluster_threshold : :obj:`int` or None, default=None
        Cluster size threshold, in :term:`voxels<voxel>`.
        If None, then no cluster size threshold will be applied.

    two_sided : :obj:`bool`, default=False
        Whether to employ two-sided thresholding or to evaluate positive values
        only.

    min_distance : :obj:`float`, default=8
        Minimum distance between subpeaks, in millimeters.

        .. note::
            If two different clusters are closer than ``min_distance``, it can
            result in peaks closer than ``min_distance``.

    return_label_maps : :obj:`bool`, default=False
        Whether or not to additionally output cluster label map images.

        .. versionadded:: 0.10.1

    Returns
    -------
    result_table : :obj:`pandas.DataFrame`
                   Table with peaks and subpeaks from thresholded ``stat_img``.
                   The columns in this table include:

        ================== ====================================================
        Cluster ID         The cluster number. Subpeaks have letters after the
                           number.
        X/Y/Z              The coordinate for the peak, in millimeters.
        Peak Stat          The statistical value associated with the peak.
                           The statistic type is dependent on the type of the
                           statistical image.
        Cluster Size (mm3) The size of the cluster, in millimeters cubed.
                           Rows corresponding to subpeaks will not have a value
                           in this column.
        ================== ====================================================

    label_maps : :obj:`list`
        Returned if return_label_maps=True
        List of Niimg-like objects of cluster label maps.
        If two_sided==True, first and second maps correspond
        to positive and negative tails.

        .. versionadded:: 0.10.1

    """
    cols = ["Cluster ID", "X", "Y", "Z", "Peak Stat", "Cluster Size (mm3)"]
    # Replace None with 0
    cluster_threshold = 0 if cluster_threshold is None else cluster_threshold

    # check that stat_img is niimg-like object and 3D
    stat_img = check_niimg_3d(stat_img)
    affine = stat_img.affine
    shape = stat_img.shape

    # Apply threshold(s) to image
    stat_img = threshold_img(
        img=stat_img,
        threshold=stat_threshold,
        cluster_threshold=cluster_threshold,
        two_sided=two_sided,
        mask_img=None,
        copy=True,
        copy_header=True,
    )

    # If cluster threshold is used, there is chance that stat_map will be
    # modified, therefore copy is needed
    stat_map = safe_get_data(
        stat_img,
        ensure_finite=True,
        copy_data=(cluster_threshold is not None),
    )

    # Define array for 6-connectivity, aka NN1 or "faces"
    bin_struct = generate_binary_structure(rank=3, connectivity=1)

    voxel_size = np.prod(stat_img.header.get_zooms())

    signs = [1, -1] if two_sided else [1]
    no_clusters_found = True
    rows = []
    label_maps = []
    for sign in signs:
        # Flip map if necessary
        temp_stat_map = stat_map * sign

        # Binarize using cluster-defining threshold
        binarized = temp_stat_map > stat_threshold
        binarized = binarized.astype(int)

        # If the stat threshold is too high simply return an empty dataframe
        if np.sum(binarized) == 0:
            warnings.warn(
                "Attention: No clusters "
                f'with stat {"higher" if sign == 1 else "lower"} '
                f"than {stat_threshold * sign}",
                category=UserWarning,
                stacklevel=2,
            )
            continue

        # Now re-label and create table
        label_map = label(binarized, bin_struct)[0]
        clust_ids = sorted(np.unique(label_map)[1:])
        peak_vals = np.array(
            [np.max(temp_stat_map * (label_map == c)) for c in clust_ids]
        )
        # Sort by descending max value
        clust_ids = [clust_ids[c] for c in (-peak_vals).argsort()]

        if return_label_maps:
            # Relabel label_map based on sorted ids
            relabel_idx = np.insert(clust_ids, 0, 0).argsort().astype(np.int32)
            relabel_map = relabel_idx[label_map.flatten()].reshape(shape)
            # Save label maps as nifti objects
            label_maps.append(
                new_img_like(stat_img, relabel_map, affine=affine)
            )

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
                    sp_id = f"{c_id + 1}{ascii_lowercase[subpeak - 1]}"
                    row = [
                        sp_id,
                        subpeak_xyz[subpeak, 0],
                        subpeak_xyz[subpeak, 1],
                        subpeak_xyz[subpeak, 2],
                        subpeak_vals[subpeak],
                        "",
                    ]
                rows += [row]

        # If we reach this point, there are clusters in this sign
        no_clusters_found = False

    if no_clusters_found:
        result_table = pd.DataFrame(columns=cols)
    else:
        result_table = pd.DataFrame(columns=cols, data=rows)

    return (result_table, label_maps) if return_label_maps else result_table
