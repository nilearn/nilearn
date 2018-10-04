"""
This module implements plotting functions useful to report analysis results.

Author: Martin Perez-Guevara, Elvis Dohmatob, 2017
"""

import os
import warnings
from string import ascii_lowercase

import numpy as np
import pandas as pd
import nibabel as nib
from scipy import stats
from scipy import ndimage
import nilearn.plotting  # overrides the backend on headless servers
from nilearn.image.resampling import coord_transform
import matplotlib.pyplot as plt
from patsy import DesignInfo

from .design_matrix import check_design_matrix


def _local_max(data, affine, min_distance):
    """Find all local maxima of the array, separated by at least min_distance.
    Adapted from https://stackoverflow.com/a/22631583/2589328

    Parameters
    ----------
    data : array_like
        3D array of with masked values for cluster.

    min_distance : :obj:`int`
        Minimum distance between local maxima in ``data``, in terms of mm.

    Returns
    -------
    ijk : :obj:`numpy.ndarray`
        (n_foci, 3) array of local maxima indices for cluster.

    vals : :obj:`numpy.ndarray`
        (n_foci,) array of values from data at ijk.
    """
    # Initial identification of subpeaks with minimal minimum distance
    data_max = ndimage.filters.maximum_filter(data, 3)
    maxima = (data == data_max)
    data_min = ndimage.filters.minimum_filter(data, 3)
    diff = ((data_max - data_min) > 0)
    maxima[diff == 0] = 0

    labeled, n_subpeaks = ndimage.label(maxima)
    ijk = np.array(ndimage.center_of_mass(data, labeled,
                                          range(1, n_subpeaks + 1)))
    ijk = np.round(ijk).astype(int)

    vals = np.apply_along_axis(arr=ijk, axis=1, func1d=_get_val,
                               input_arr=data)

    # Sort subpeaks in cluster in descending order of stat value
    order = (-vals).argsort()
    vals = vals[order]
    ijk = ijk[order, :]
    xyz = nib.affines.apply_affine(affine, ijk)  # Convert to xyz in mm

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
                       min_distance=8.):
    """Creates pandas dataframe with img cluster statistics.

    Parameters
    ----------
    stat_img : Niimg-like object,
       Statistical image (presumably in z- or p-scale).

    stat_threshold: :obj:`float`
        Cluster forming threshold in same scale as `stat_img` (either a
        p-value or z-scale value).

    cluster_threshold : :obj:`int` or :obj:`None`, optional
        Cluster size threshold, in voxels.

    min_distance: :obj:`float`, optional
        Minimum distance between subpeaks in mm. Default is 8 mm.

    Returns
    -------
    df : :obj:`pandas.DataFrame`
        Table with peaks and subpeaks from thresholded `stat_img`. For binary
        clusters (clusters with >1 voxel containing only one value), the table
        reports the center of mass of the cluster, rather than any peaks/subpeaks.
    """
    cols = ['Cluster ID', 'X', 'Y', 'Z', 'Peak Stat', 'Cluster Size (mm3)']
    stat_map = stat_img.get_data()
    conn_mat = np.zeros((3, 3, 3), int)  # 6-connectivity, aka NN1 or "faces"
    conn_mat[1, 1, :] = 1
    conn_mat[1, :, 1] = 1
    conn_mat[:, 1, 1] = 1
    voxel_size = np.prod(stat_img.header.get_zooms())

    # Binarize using CDT
    binarized = stat_map > stat_threshold
    binarized = binarized.astype(int)

    # If the stat threshold is too high simply return an empty dataframe
    if np.sum(binarized) == 0:
        warnings.warn('Attention: No clusters with stat higher than %f' %
                      stat_threshold)
        return pd.DataFrame(columns=cols)

    # Extract connected components above cluster size threshold
    label_map = ndimage.measurements.label(binarized, conn_mat)[0]
    clust_ids = sorted(list(np.unique(label_map)[1:]))
    for c_val in clust_ids:
        if cluster_threshold is not None and np.sum(label_map == c_val) < cluster_threshold:
            stat_map[label_map == c_val] = 0
            binarized[label_map == c_val] = 0

    # If the cluster threshold is too high simply return an empty dataframe
    # this checks for stats higher than threshold after small clusters
    # were removed from stat_map
    if np.sum(stat_map > stat_threshold) == 0:
        warnings.warn('Attention: No clusters with more than %d voxels' %
                      cluster_threshold)
        return pd.DataFrame(columns=cols)

    # Now re-label and create table
    label_map = ndimage.measurements.label(binarized, conn_mat)[0]
    clust_ids = sorted(list(np.unique(label_map)[1:]))
    peak_vals = np.array([np.max(stat_map * (label_map == c)) for c in clust_ids])
    clust_ids = [clust_ids[c] for c in (-peak_vals).argsort()]  # Sort by descending max value

    rows = []
    for c_id, c_val in enumerate(clust_ids):
        cluster_mask = label_map == c_val
        masked_data = stat_map * cluster_mask

        cluster_size_mm = int(np.sum(cluster_mask) * voxel_size)

        # Get peaks, subpeaks and associated statistics
        subpeak_ijk, subpeak_vals = _local_max(masked_data, stat_img.affine,
                                               min_distance=min_distance)
        subpeak_xyz = np.asarray(coord_transform(subpeak_ijk[:, 0],
                                                 subpeak_ijk[:, 1],
                                                 subpeak_ijk[:, 2],
                                                 stat_img.affine)).tolist()
        subpeak_xyz = np.array(subpeak_xyz).T

        # Only report peak and, at most, top 3 subpeaks.
        n_subpeaks = np.min((len(subpeak_vals), 4))
        for subpeak in range(n_subpeaks):
            if subpeak == 0:
                row = [c_id + 1, subpeak_xyz[subpeak, 0],
                       subpeak_xyz[subpeak, 1], subpeak_xyz[subpeak, 2],
                       subpeak_vals[subpeak], cluster_size_mm]
            else:
                # Subpeak naming convention is cluster num + letter (1a, 1b, etc.)
                sp_id = '{0}{1}'.format(c_id + 1, ascii_lowercase[subpeak - 1])
                row = [sp_id, subpeak_xyz[subpeak, 0], subpeak_xyz[subpeak, 1],
                       subpeak_xyz[subpeak, 2], subpeak_vals[subpeak], '']
            rows += [row]
    df = pd.DataFrame(columns=cols, data=rows)
    return df


def compare_niimgs(ref_imgs, src_imgs, masker, plot_hist=True, log=True,
                   ref_label="image set 1", src_label="image set 2",
                   output_dir=None, axes=None):
    """Creates plots to compare two lists of images and measure correlation.

    The first plot displays linear correlation between voxel values
    The second plot superimposes histograms to compare values distribution

    Parameters
    ----------
    ref_imgs: nifti_like
        reference images.

    src_imgs: nifti_like
        Source images.

    log: Boolean, optional (default True)
        Passed to plt.hist

    plot_hist: Boolean, optional (default True)
        If True then histograms of each img in ref_imgs will be plotted
        along-side the histogram of the corresponding image in src_imgs

    ref_label: str
        name of reference images

    src_label: str
        name of source images

    output_dir: string, optional (default None)
        Directory where plotted figures will be stored.

    axes: list of two matplotlib Axes objects, optional (default None)
        Can receive a list of the form [ax1, ax2] to render the plots.
        By default new axes will be created

    Returns
    -------
    Pearsonr correlation between the images

    Examples
    --------
    [1] check_zscores.compare_niimgs(["/home/elvis/Downloads/zstat2.nii.gz"],
            ["/home/elvis/Downloads/zstat8.nii.gz"], output_dir="/tmp/toto")
    """
    corrs = []
    for i, (ref_img, src_img) in enumerate(zip(ref_imgs, src_imgs)):
        if axes is None:
            _, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        else:
            (ax1, ax2) = axes
        ref_data = masker.transform(ref_img).ravel()
        src_data = masker.transform(src_img).ravel()
        if ref_data.shape != src_data.shape:
            warnings.warn("Images are not shape-compatible")
            return

        corr = stats.pearsonr(ref_data, src_data)[0]
        corrs.append(corr)

        if plot_hist:
            ax1.scatter(
                ref_data, src_data, label="Pearsonr: %.2f" % corr, c="g",
                alpha=.6)
            x = np.linspace(*ax1.get_xlim(), num=100)
            ax1.plot(x, x, linestyle="--", c="k")
            ax1.grid("on")
            ax1.set_xlabel(ref_label)
            ax1.set_ylabel(src_label)
            ax1.legend(loc="best")

            ax2.hist(ref_data, alpha=.6, bins=128, log=log, label=ref_label)
            ax2.hist(src_data, alpha=.6, bins=128, log=log, label=src_label)
            ax2.set_title("Histogram of imgs values")
            ax2.grid("on")
            ax2.legend(loc="best")

            if output_dir is not None:
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                plt.savefig(os.path.join(output_dir, "%04i.png" % i))

        plt.tight_layout()

    return corrs


def plot_design_matrix(design_matrix, rescale=True, ax=None, output_file=None):
    """Plot a design matrix provided as a DataFrame

    Parameters
    ----------
    design matrix : pandas DataFrame,
        Describes a design matrix.

    rescale : bool, optional
        Rescale columns magnitude for visualization or not.

    ax : axis handle, optional
        Handle to axis onto which we will draw design matrix.

    output_file: string or None, optional,
        The name of an image file to export the plot to. Valid extensions
        are .png, .pdf, .svg. If output_file is not None, the plot
        is saved to a file, and the display is closed.
   
    Returns
    -------
    ax: axis handle
        The axis used for plotting.
    """
    # We import _set_mpl_backend because just the fact that we are
    # importing it sets the backend
    from nilearn.plotting import _set_mpl_backend
    # avoid unhappy pyflakes
    _set_mpl_backend

    # normalize the values per column for better visualization
    _, X, names = check_design_matrix(design_matrix)
    if rescale:
        X = X / np.maximum(1.e-12, np.sqrt(np.sum(X ** 2, 0)))  # pylint: disable=no-member
    if ax is None:
        plt.figure()
        ax = plt.subplot(1, 1, 1)

    ax.imshow(X, interpolation='nearest', aspect='auto')
    ax.set_label('conditions')
    ax.set_ylabel('scan number')

    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=60, ha='right')

    plt.tight_layout()
    if output_file is not None:
        plt.savefig(output_file)
        plt.close()
        ax = None
    return ax


def plot_contrast_matrix(contrast_def, design_matrix, colorbar=False, ax=None,
                         output_file=None):
    """Creates plot for contrast definition.

    Parameters
    ----------
    contrast_def : str or array of shape (n_col) or list of (string or
                   array of shape (n_col))
        where ``n_col`` is the number of columns of the design matrix,
        (one array per run). If only one array is provided when there
        are several runs, it will be assumed that the same contrast is
        desired for all runs. The string can be a formula compatible with
        the linear constraint of the Patsy library. Basically one can use
        the name of the conditions as they appear in the design matrix of
        the fitted model combined with operators /*+- and numbers.
        Please checks the patsy documentation for formula examples:
        http://patsy.readthedocs.io/en/latest/API-reference.html#patsy.DesignInfo.linear_constraint

    design_matrix: pandas DataFrame

    colorbar: Boolean, optional (default False)
        Include a colorbar in the contrast matrix plot.

    ax: matplotlib Axes object, optional (default None)
        Directory where plotted figures will be stored.
    
    output_file: string or None, optional,
        The name of an image file to export the plot to. Valid extensions
        are .png, .pdf, .svg. If output_file is not None, the plot
        is saved to a file, and the display is closed.


    Returns
    -------
    Plot Axes object
    """

    design_column_names = design_matrix.columns.tolist()
    if isinstance(contrast_def, str):
        design_info = DesignInfo(design_column_names)
        contrast_def = design_info.linear_constraint(contrast_def).coefs

    maxval = np.max(np.abs(contrast_def))
    con_matrix = np.asmatrix(contrast_def)

    if ax is None:
        plt.figure(figsize=(8, 4))
        ax = plt.gca()

    mat = ax.matshow(con_matrix, aspect='equal',
                     extent=[0, con_matrix.shape[1], 0, con_matrix.shape[0]],
                     cmap='gray', vmin=-maxval, vmax=maxval)

    ax.set_label('conditions')
    ax.set_ylabel('')
    ax.set_yticklabels(['' for x in ax.get_yticklabels()])

    # Shift ticks to be at 0.5, 1.5, etc
    ax.xaxis.set(ticks=np.arange(1.0, len(design_column_names) + 1.0),
                 ticklabels=design_column_names)
    ax.set_xticklabels(design_column_names, rotation=60, ha='right')

    if colorbar:
        plt.colorbar(mat, fraction=0.025, pad=0.04)

    plt.tight_layout()
    if output_file is not None:
        plt.savefig(output_file)
        plt.close()
        ax = None

    return ax
