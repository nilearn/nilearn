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
                   output_dir=None, axes=None):
    """Creates plots to compare two lists of images and measure correlation.

    The first plot tests linear correlation between voxel values
    The second plot superimpose histograms to compare values distribution

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


def plot_contrast_matrix(contrast_def, design_matrix, colorbar=False, ax=None):
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

    colorbar: boolean, optional (default False)
        Include a colorbar in the contrast matrix plot.

    ax: matplotlib Axes object, optional (default None)
        Directory where plotted figures will be stored.

    Returns
    -------
    Plot Axes object
    """

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
