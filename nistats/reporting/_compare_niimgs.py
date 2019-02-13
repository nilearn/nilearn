"""
This module implements plotting functions useful to report analysis results.

Author: Martin Perez-Guevara, Elvis Dohmatob, 2017
"""

import os
import warnings

import numpy as np
from scipy import stats
import nilearn.plotting  # overrides headless server backend, preempts
                         # MatPlotLib import error when it's not installed.
import matplotlib.pyplot as plt


def compare_niimgs(ref_imgs, src_imgs, masker, plot_hist=True, log=True,
                   ref_label="image set 1", src_label="image set 2",
                   output_dir=None, axes=None):
    """Creates plots to compare two lists of images and measure correlation.

    The first plot displays linear correlation between voxel values.
    The second plot superimposes histograms to compare values distribution.

    Parameters
    ----------
    ref_imgs: nifti_like
        Reference images.

    src_imgs: nifti_like
        Source images.

    masker: NiftiMasker object
        Mask to be used on data.
    
    plot_hist: Boolean, optional (default True)
        If True then histograms of each img in ref_imgs will be plotted
        along-side the histogram of the corresponding image in src_imgs

    log: Boolean, optional (default True)
        Passed to plt.hist

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
    corrs: numpy.ndarray
        Pearson correlation between the images
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
