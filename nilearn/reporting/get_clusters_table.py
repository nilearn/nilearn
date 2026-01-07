"""Implement plotting functions useful to report analysis results."""

import inspect
import warnings
from collections import OrderedDict
from decimal import Decimal
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

from nilearn._utils.docs import fill_doc
from nilearn._utils.logger import find_stack_level
from nilearn._utils.niimg import safe_get_data
from nilearn._utils.param_validation import check_params
from nilearn.image import check_niimg_3d, new_img_like, threshold_img
from nilearn.image.resampling import coord_transform
from nilearn.surface.surface import SurfaceImage, find_surface_clusters
from nilearn.typing import ClusterThreshold


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
            stacklevel=find_stack_level(),
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
    for ii, (lab, point) in enumerate(zip(labels_index, ijk, strict=False)):
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


@fill_doc
def get_clusters_table(
    stat_img,
    stat_threshold: float | int | np.floating | np.integer,
    cluster_threshold: ClusterThreshold = 0,
    two_sided: bool = False,
    min_distance: float | int | np.floating | np.integer = 8.0,
    return_label_maps: bool = False,
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

        .. nilearn_versionchanged:: 0.9.2
            In this case, the cluster voxel nearest to the center of mass is
            reported.

    .. seealso::

        This function does not report any named anatomical location
        for the clusters.
        To get the names of the location of the clusters
        according to one or several atlases,
        we recommend using
        the `atlasreader package <https://github.com/miykael/atlasreader>`_.


    Parameters
    ----------
    stat_img : Niimg-like object or :class:`~nilearn.surface.SurfaceImage`
       Statistical image to threshold and summarize.

    stat_threshold : :obj:`float` or :obj:`int`
        Cluster forming threshold. This value must be in the same scale as
        ``stat_img``.

    %(cluster_threshold)s

    two_sided : :obj:`bool`, default=False
        Whether to employ two-sided thresholding or to evaluate positive values
        only.

    min_distance : :obj:`float`, default=8.0
        Minimum distance between subpeaks, in millimeters.

        .. note::
            If two different clusters are closer than ``min_distance``, it can
            result in peaks closer than ``min_distance``.

        .. note::
            Not used for surface data.

    return_label_maps : :obj:`bool`, default=False
        Whether or not to additionally output cluster label map images.

        .. nilearn_versionadded:: 0.10.1

    Returns
    -------
    result_table : :obj:`pandas.DataFrame`

        For volume data the dataframe contains
        the peaks and subpeaks from thresholded ``stat_img``.
        In this case, the columns in this table include:

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

        For surface data, the columns in this table include:

        ======================= ===============================================
        Cluster ID              The cluster number.
        Hemisphere              The hemisphere in which the cluster is found.
        Peak Stat               The statistical value associated
                                with the cluster.
                                The statistic type is dependent
                                on the type of the statistical image.
        Cluster Size (vertices) The size of the cluster, in vertices.
        ======================= ===============================================

    label_maps : :obj:`list` of  Niimg-like objects \
                 or :class:`~nilearn.surface.SurfaceImage`
        List of of cluster label maps.
        Returned if ``return_label_maps=True``.
        If ``two_sided==True``, first and second maps correspond
        to positive and negative tails.

        .. nilearn_versionadded:: 0.10.1

    """
    check_params(locals())

    is_volume = not isinstance(stat_img, SurfaceImage)

    stat_img = threshold_img(
        img=stat_img,
        threshold=stat_threshold,
        cluster_threshold=cluster_threshold,
        two_sided=two_sided,
    )

    if is_volume:
        return _get_clusters_table_volume(
            stat_img,
            stat_threshold,
            cluster_threshold=cluster_threshold,
            two_sided=two_sided,
            min_distance=min_distance,
            return_label_maps=return_label_maps,
        )

    parameters = dict(**inspect.signature(get_clusters_table).parameters)
    if min_distance != parameters["min_distance"].default:
        warnings.warn(
            "The 'min_distance' parameter is not used for surface data "
            "and will be ignored.",
            stacklevel=find_stack_level(),
        )

    return _get_clusters_table_surface(
        stat_img,
        stat_threshold,
        cluster_threshold=cluster_threshold,
        two_sided=two_sided,
        return_label_maps=return_label_maps,
    )


def _get_clusters_table_surface(
    stat_img,
    stat_threshold,
    cluster_threshold: ClusterThreshold = 0,
    two_sided: bool = False,
    return_label_maps: bool = False,
    offset=1,
):
    """Generate cluster table for surface data.

    When two_sided is True, this function calls itself recursively
    for each tail.

    Parameters
    ----------
    offset : int, default=1
        Offset to add to cluster IDs.
        Useful when calling recursively
        for two-sided thresholding.

    For other parameters, see `get_clusters_table`.

    """
    cols = [
        "Cluster ID",
        "Hemisphere",
        "Peak Stat",
        "Cluster Size (vertices)",
    ]

    data = {}
    all_clusters = []
    label_maps = []

    if not two_sided:
        for hemi in stat_img.data.parts:
            clusters, labels = find_surface_clusters(
                stat_img.mesh.parts[hemi],
                stat_img.data.parts[hemi],
                offset=offset,
            )

            peak_stat = []
            for i in clusters["index"].tolist():
                mask = labels == i
                values = stat_img.data.parts[hemi][mask].ravel()

                cluster_max = np.max(values)
                peak_stat.append(cluster_max)

            clusters["Peak Stat"] = peak_stat

            clusters["Hemisphere"] = hemi
            clusters = clusters.rename(
                columns={
                    "name": "Cluster ID",
                    "size": "Cluster Size (vertices)",
                }
            )
            clusters = clusters[cols]

            all_clusters.append(clusters)

            data[hemi] = labels

            offset += len(clusters)

        if offset == 1:
            warnings.warn(
                f"No clusters found for '{stat_threshold=}'.",
                category=UserWarning,
                stacklevel=find_stack_level(),
            )

        label_maps = [new_img_like(stat_img, data)]

    else:
        signs = [1, -1]
        for sign in signs:
            temp_stat_map = threshold_img(
                img=stat_img,
                threshold=stat_threshold * sign,
                cluster_threshold=cluster_threshold,
                two_sided=False,
            )
            clusters, label_map = _get_clusters_table_surface(
                temp_stat_map,
                stat_threshold * sign,
                cluster_threshold=cluster_threshold,
                two_sided=False,
                return_label_maps=True,
                offset=offset,
            )

            offset += len(clusters)

            all_clusters.append(clusters)

            label_maps.append(label_map[0])

    result_table = pd.concat(all_clusters, ignore_index=True)

    if return_label_maps:
        return (result_table, label_maps)

    return result_table


def _get_clusters_table_volume(
    stat_img,
    stat_threshold: float | int | np.floating | np.integer,
    cluster_threshold: ClusterThreshold = 0,
    two_sided: bool = False,
    min_distance: float | int | np.floating | np.integer = 8.0,
    return_label_maps: bool = False,
):
    """Generate cluster table for volume data.

    For parameters, see `get_clusters_table`.
    """
    if min_distance <= 0:
        raise ValueError("'min_distance' must be positive.")

    cols = ["Cluster ID", "X", "Y", "Z", "Peak Stat", "Cluster Size (mm3)"]

    label_maps = []

    # check that stat_img is niimg-like object and 3D
    stat_img = check_niimg_3d(stat_img)
    affine = stat_img.affine
    shape = stat_img.shape

    # If cluster threshold is used, there is chance that stat_map will be
    # modified, therefore copy is needed
    stat_map = safe_get_data(
        stat_img,
        ensure_finite=True,
        copy_data=(cluster_threshold != 0),
    )

    # Define array for 6-connectivity, aka NN1 or "faces"
    bin_struct = generate_binary_structure(rank=3, connectivity=1)

    voxel_size = np.prod(stat_img.header.get_zooms())

    clusters_found = False
    signs = [1, -1] if two_sided else [1]
    rows: list = []

    for sign in signs:
        offset = len(rows)

        # Flip map if necessary
        temp_stat_map = stat_map * sign

        # Binarize using cluster-defining threshold
        if not two_sided and stat_threshold < 0:
            binarized = temp_stat_map < stat_threshold
        else:
            binarized = temp_stat_map > stat_threshold
        binarized = binarized.astype(int)

        # If the stat threshold is too high
        # simply return an empty dataframe
        if np.sum(binarized) == 0:
            warnings.warn(
                "No clusters found "
                f"with stat {'higher' if sign == 1 else 'lower'} "
                f"than {stat_threshold * sign}",
                category=UserWarning,
                stacklevel=find_stack_level(),
            )
            continue

        # Now re-label and create table
        label_map = label(binarized, bin_struct)[0]
        clust_ids = sorted(np.unique(label_map)[1:])
        if not two_sided and stat_threshold < 0:
            peak_vals = np.array(
                [np.min(temp_stat_map * (label_map == c)) for c in clust_ids]
            )
        else:
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
            if not two_sided and stat_threshold < 0:
                # in this we will want to find the local minima
                masked_data *= -1

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
                        c_id + offset + 1,
                        subpeak_xyz[subpeak, 0],
                        subpeak_xyz[subpeak, 1],
                        subpeak_xyz[subpeak, 2],
                        subpeak_vals[subpeak],
                        cluster_size_mm,
                    ]
                else:
                    # Subpeak naming convention is cluster num+letter:
                    # 1a, 1b, etc
                    sp_id = (
                        f"{c_id + offset + 1}{ascii_lowercase[subpeak - 1]}"
                    )
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
        clusters_found = True

    if clusters_found:
        result_table = pd.DataFrame(columns=cols, data=rows)
    else:
        result_table = pd.DataFrame(columns=cols)

    return (result_table, label_maps) if return_label_maps else result_table


@fill_doc
def clustering_params_to_dataframe(
    threshold,
    cluster_threshold,
    min_distance,
    height_control,
    alpha,
    is_volume_glm,
) -> pd.DataFrame:
    """Create a Pandas DataFrame from the supplied arguments.

    For use as part of the Cluster Table.

    Parameters
    ----------
    threshold : float
        Cluster forming threshold in same scale as `stat_img` (either a
        p-value or z-scale value).

    %(cluster_threshold)s

    min_distance : float
        For display purposes only.
        Minimum distance between subpeaks in mm.

    height_control : :obj:`str` or None
        False positive control meaning of cluster forming
        threshold: 'fpr' (default) or 'fdr' or 'bonferroni' or None

    alpha : float
        Number controlling the thresholding (either a p-value or q-value).
        Its actual meaning depends on the height_control parameter.
        This function translates alpha to a z-scale threshold.

    is_volume_glm: bool
        True if we are dealing with volume data.

    Returns
    -------
    table_details : Pandas.DataFrame
        Dataframe with clustering parameters.

    """
    check_params(locals())
    table_details = OrderedDict()
    threshold = np.around(threshold, 3)

    if height_control:
        table_details.update({"Height control": height_control})
        # HTMLDocument.get_iframe() invoked in Python2 Jupyter Notebooks
        # mishandles certain unicode characters
        # & raises error due to greek alpha symbol.
        # This is simpler than overloading the class using inheritance,
        # especially given limited Python2 use at time of release.
        if alpha < 0.001:
            alpha = f"{Decimal(alpha):.2E}"
        table_details.update({"\u03b1": alpha})
        table_details.update({"Threshold (computed)": threshold})
    else:
        table_details.update({"Height control": "None"})
        table_details.update({"Threshold Z": threshold})

    if is_volume_glm:
        table_details.update(
            {"Cluster size threshold (voxels)": cluster_threshold}
        )
        table_details.update({"Minimum distance (mm)": min_distance})

    return pd.DataFrame.from_dict(
        table_details,
        orient="index",
    )
