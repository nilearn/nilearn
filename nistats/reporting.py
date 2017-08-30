import os
import warnings
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from nilearn.image.image import check_niimg_3d


def compare_niimgs(ref_imgs, src_imgs, plot_hist=True, log=True,
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
        ref_img = check_niimg_3d(ref_img).get_data()
        src_img = check_niimg_3d(src_img).get_data()
        if ref_img.shape != src_img.shape:
            warnings.warn("Images are not shape-compatible")
            return

        ref_img = ref_img.ravel()
        src_img = src_img.ravel()
        corr = stats.pearsonr(ref_img, src_img)[0]
        corrs.append(corr)

        if plot_hist:
            ax1.scatter(ref_img, src_img, label="Pearsonr: %.2f" % corr, c="g",
                        alpha=.6)
            x = np.linspace(*ax1.get_xlim(), num=100)
            ax1.plot(x, x, linestyle="--", c="k")
            ax1.grid("on")
            ax1.set_xlabel(ref_label)
            ax1.set_ylabel(src_label)
            ax1.legend(loc="best")

            ax2.hist(ref_img, alpha=.6, bins=128, log=log, label=ref_label)
            ax2.hist(src_img, alpha=.6, bins=128, log=log, label=src_label)
            ax2.set_title("histogram of values")
            ax2.grid("on")
            ax2.legend(loc="best")

            if output_dir is not None:
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                plt.savefig(os.path.join(output_dir, "%04i.png" % i))

        plt.tight_layout()

    return corrs
