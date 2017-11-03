""" Utilities to describe the result of cluster-level analysis of statistical
maps.

Author: Bertrand Thirion, 2015
"""
import numpy as np
from scipy.ndimage import label
from scipy.stats import norm
from nilearn.input_data import NiftiMasker
from nilearn.image.resampling import coord_transform
import pandas as pd
from warnings import warn


def fdr_threshold(z_vals, alpha):
    """ return the BH fdr for the input z_vals"""
    z_vals_ = - np.sort(- z_vals)
    p_vals = norm.sf(z_vals_)
    n_samples = len(p_vals)
    pos = p_vals < alpha * np.linspace(
        .5 / n_samples, 1 - .5 / n_samples, n_samples)
    if pos.any():
        return (z_vals_[pos][-1] - 1.e-8)
    else:
        return np.infty


def map_threshold(stat_img, mask_img=None, threshold=.001,
                  height_control='fpr', cluster_threshold=0):
    """ Threshold the provided map

    Parameters
    ----------
    stat_img : Niimg-like object,
       statistical image (presumably in z scale)

    mask_img : Niimg-like object, optional,
        mask image

    threshold: float, optional
        cluster forming threshold (either a p-value or z-scale value)

    height_control: string, optional
        false positive control meaning of cluster forming
        threshold: 'fpr'|'fdr'|'bonferroni'|'none'

    cluster_threshold : float, optional
        cluster size threshold

    Returns
    -------
    thresholded_map : Nifti1Image,
        the stat_map theresholded at the prescribed voxel- and cluster-level

    threshold: float,
        the voxel-level threshold used actually
    """
    # Masking
    if mask_img is None:
        masker = NiftiMasker(mask_strategy='background').fit(stat_img)
    else:
        masker = NiftiMasker(mask_img=mask_img).fit()
    stats = np.ravel(masker.transform(stat_img))
    n_voxels = np.size(stats)

    # Thresholding
    if height_control == 'fpr':
        z_th = norm.isf(threshold)
    elif height_control == 'fdr':
        z_th = fdr_threshold(stats, threshold)
    elif height_control == 'bonferroni':
        z_th = norm.isf(threshold / n_voxels)
    else:  # Brute-force thresholding
        z_th = threshold
    stats *= (stats > z_th)

    # embed it back to 3D grid
    stat_map = masker.inverse_transform(stats).get_data()

    # Extract connected components above threshold
    label_map, n_labels = label(stat_map > z_th)
    labels = label_map[masker.mask_img_.get_data() > 0]

    for label_ in range(1, n_labels + 1):
        if np.sum(labels == label_) < cluster_threshold:
            stats[labels == label_] = 0

    return masker.inverse_transform(stats), z_th


def get_clusters_table(stat_img, stat_threshold, cluster_threshold):
    """Creates pandas dataframe with img cluster statistics.

    Parameters
    ----------
    stat_img : Niimg-like object,
       statistical image (presumably in z scale)

    stat_threshold: float, optional
        cluster forming threshold (either a p-value or z-scale value)

    cluster_threshold : int, optional
        cluster size threshold

    Returns
    -------
    Pandas dataframe with img clusters
    """

    stat_map = stat_img.get_data()

    # If the stat threshold is too high simply return an empty dataframe
    if np.sum(stat_map > stat_threshold) == 0:
        warn('Attention: No clusters with stat higher than %f' %
             stat_threshold)
        return pd.DataFrame()

    # Extract connected components above threshold
    label_map, n_labels = label(stat_map > stat_threshold)

    for label_ in range(1, n_labels + 1):
        if np.sum(label_map == label_) < cluster_threshold:
            stat_map[label_map == label_] = 0

    # If the cluster threshold is too high simply return an empty dataframe
    # this checks for stats higher than threshold after small clusters
    # were removed from stat_map
    if np.sum(stat_map > stat_threshold) == 0:
        warn('Attention: No clusters with more than %d voxels' %
             cluster_threshold)
        return pd.DataFrame()

    label_map, n_labels = label(stat_map > stat_threshold)
    label_map = np.ravel(label_map)
    stat_map = np.ravel(stat_map)

    peaks = []
    max_stat = []
    clusters_size = []
    coords = []
    for label_ in range(1, n_labels + 1):
        cluster = stat_map.copy()
        cluster[label_map != label_] = 0

        peak = np.unravel_index(np.argmax(cluster),
                                stat_img.get_data().shape)
        peaks.append(peak)

        max_stat.append(np.max(cluster))

        clusters_size.append(np.sum(label_map == label_))

        x_map, y_map, z_map = peak
        mni_coords = np.asarray(
            coord_transform(
                x_map, y_map, z_map, stat_img.get_affine())).tolist()
        mni_coords = [round(x) for x in mni_coords]
        coords.append(mni_coords)

    vx, vy, vz = zip(*peaks)
    x, y, z = zip(*coords)

    columns = ['Vx', 'Vy', 'Vz', 'X', 'Y', 'Z', 'Peak stat', 'Cluster size']
    clusters_table = pd.DataFrame(
        list(zip(vx, vy, vz, x, y, z, max_stat, clusters_size)),
        columns=columns)

    return clusters_table
