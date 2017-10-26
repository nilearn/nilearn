import os
import warnings
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from nilearn.image.image import check_niimg_3d
from patsy import DesignInfo
from scipy.ndimage import label, binary_dilation, center_of_mass
from nilearn.image.resampling import coord_transform
import pandas as pd
import matplotlib 
matplotlib.rc('xtick', labelsize=20) 


def compare_niimgs(ref_imgs, src_imgs, masker, plot_hist=True, log=True,
                   ref_label="image set 1", src_label="image set 2",
                   output_dir=None):
    """Compares two lists of images.

    Parameters
    ----------
    ref_imgs: nifti_like
        reference images.

    src_imgs: nifti_like
        Source images.

    log: boolearn, optional (default True)
        Passed to plt.hist

    plot_hist: boolean, optional (default True)
        If True then then histograms of each img in ref_imgs will be plotted
        along-side the histogram of the corresponding image in src_imgs

    output_dir: string, optional (default None)
        Directory where plotted figures will be stored.

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
        _, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
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


def plot_contrast_matrix(contrast_def, design_matrix, colorbar=False, ax=None):
    design_column_names = design_matrix.columns.tolist()
    if isinstance(contrast_def, str):
        di = DesignInfo(design_column_names)
        contrast_def = di.linear_constraint(contrast_def).coefs

    if ax is None:
        plt.figure(figsize=(8, 4))
        ax = plt.gca()

    maxval = np.max(np.abs(contrast_def))

    con_mx = np.asmatrix(contrast_def)
    mat = ax.matshow(con_mx, aspect='equal', extent=[0, con_mx.shape[1],
                     0, con_mx.shape[0]], cmap='gray', vmin=-maxval,
                     vmax=maxval)
    ax.set_label('conditions')
    ax.set_ylabel('')
    ax.set_yticklabels(['' for x in ax.get_yticklabels()])

    # Shift ticks to be at 0.5, 1.5, etc
    ax.xaxis.set(ticks=np.arange(1.0, len(design_column_names) + 1.0),
                 ticklabels=design_column_names)
    ax.set_xticklabels(design_column_names, rotation=90, ha='right')

    if colorbar:
        plt.colorbar(mat, fraction=0.025, pad=0.04)

    plt.tight_layout()

    return ax


def get_clusters_table(stat_img, stat_threshold, cluster_threshold):
    stat_map = stat_img.get_data()

    # Extract connected components above threshold
    label_map, n_labels = label(stat_map > stat_threshold)

    # labels = label_map[search_mask.get_data() > 0]
    for label_ in range(1, n_labels + 1):
        if np.sum(label_map == label_) < cluster_threshold:
            stat_map[label_map == label_] = 0

    label_map, n_labels = label(stat_map > stat_threshold)
    label_map = np.ravel(label_map)
    stat_map = np.ravel(stat_map)

    peaks = []
    max_stat = []
    clusters_size = []
    # centers = []
    coords = []
    for label_ in range(1, n_labels + 1):
        cluster = stat_map.copy()
        cluster[label_map != label_] = 0

        peak = np.unravel_index(np.argmax(cluster),
                                stat_img.get_data().shape)
        peaks.append(peak)

        max_stat.append(np.max(cluster))

        clusters_size.append(np.sum(label_map == label_))

        # center = center_of_mass(cluster)
        # centers.append([round(x) for x in center[:3]])

        x_map, y_map, z_map = peak
        mni_coords = np.asarray(
            coord_transform(x_map, y_map, z_map, stat_img.affine)).tolist()
        mni_coords = [round(x) for x in mni_coords]
        coords.append(mni_coords)

    vx, vy, vz = zip(*peaks)
    x, y, z = zip(*coords)

    columns = ['Vx', 'Vy', 'Vz', 'X', 'Y', 'Z', 'Peak stat', 'Cluster size']
    clusters_table = pd.DataFrame(
        list(zip(vx, vy, vz, x, y, z, max_stat, clusters_size)),
        columns=columns)
    # clusters_table.index.name = 'Cluster'

    # d = dict(selector="th",
    #     props=[('text-align', 'center')])

    # clusters_table.style.set_properties(**{'width':'10em', 'text-align':'left'}).set_table_styles([d])

    return clusters_table
