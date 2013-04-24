"""Example showing how to compute a covariance matrix between brain regions.

**Running this script may require to change the path specified in
get_ho_parcellation() below**.

The following things are performed:
- improvement of fMRI data SNR (confound removal, filtering, etc.)
- parcellation loading, and signals extraction
- covariance/precision matrices computation
- display of matrices

This script is intended for advanced users only, since it only makes
use of low-level functions.

"""
import numpy as np
import pylab as pl
import matplotlib

from scipy import ndimage
from sklearn import covariance

import nibabel

import nisl.signals
import nisl.masking
import nisl.region
import nisl.datasets as datasets


def load_harvard_oxford():
    """Get Harvard-Oxford parcellation.

    Split every symmetric region in left and right parts. Effectively
    doubles the number of regions in the case of the Harvard-Oxford atlas.

    Returns
    =======
    regions (nibabel.Nifti1Image)
        regions definition, as a label image.
    """
    # This is neurodebian's location for FSL data. Adapt to your installation.
    # The Harvard-Oxford atlas is distributed with FSL only.
    filename = "/usr/share/data/harvard-oxford-atlases/"\
               + "HarvardOxford/HarvardOxford-cort-maxprob-thr25-2mm.nii.gz"

    regions_img = nibabel.load(filename)
    regions = regions_img.get_data()

    labels = np.unique(regions)
    slices = ndimage.find_objects(regions)
    middle_ind = (regions.shape[0] - 1) / 2
    crosses_middle = [s.start < middle_ind and s.stop > middle_ind
             for s, _, _ in slices]

    # Split every zone crossing the median plane into two parts.
    # Assumes that the background label is zero.
    half = np.zeros(regions.shape, dtype=np.bool)
    half[:middle_ind, ...] = True
    new_label = max(labels) + 1
    # Put zeros on the median plane
    regions[middle_ind, ...] = 0
    for label, crosses in zip(labels[1:], crosses_middle):
        if not crosses:
            continue
        regions[np.logical_and(regions == label, half)] = new_label
        new_label += 1
    return nibabel.Nifti1Image(regions, regions_img.get_affine())


def clean_signals(subject_n=0):
    """Load data, mask and clean them, return timeseries and mask."""
    print("-- Loading raw data ({0:d}) and masking ...".format(subject_n))
    dataset = datasets.fetch_adhd()
    filename = dataset["func"][subject_n]
    confound_file = dataset["confounds"][subject_n]

    ho_regions_img = load_harvard_oxford()
    ho_mask_img = nisl.region.regions_to_mask(ho_regions_img)
    fmri_masked = nisl.masking.apply_mask(filename, ho_mask_img)

    print("-- Computing confounds ...")
    hv_confounds = nisl.signals.high_variance_confounds(fmri_masked)
    mvt_confounds = np.loadtxt(confound_file, skiprows=1)
    confounds = np.hstack((hv_confounds, mvt_confounds))

    print("-- Cleaning signals ...")
    fmri_masked_c = nisl.signals.clean(fmri_masked, low_pass=None,
                                    detrend=True, standardize=True,
                                    confounds=confounds,
                                    t_r=2.5, high_pass=0.01
                                       )

    return fmri_masked_c, ho_regions_img, ho_mask_img


def get_region_ts(timeseries, region_img, mask_img):
    """Compute timeseries for regions."""

    print("-- Computing region signals ...")
    regions_masked = nisl.region.apply_mask_to_regions(region_img, mask_img)
    region_ts = nisl.region.apply_regions(timeseries, regions_masked)
    region_ts /= region_ts.std(axis=0)
    return region_ts


# Copied from matplotlib 1.2.0 for matplotlib 0.99
_bwr_data = ((0.0, 0.0, 1.0), (1.0, 1.0, 1.0), (1.0, 0.0, 0.0))
pl.cm.register_cmap(cmap=matplotlib.colors.LinearSegmentedColormap.from_list(
    "bwr", _bwr_data))


def plot_matrices(cov, prec, title, subject_n=0):
    """Plot covariance and precision matrices, for a given processing. """

    # Put zeros on the diagonal, for graph clarity.
    size = prec.shape[0]
    prec[range(size), range(size)] = 0

    span = max(abs(prec.min()), abs(prec.max()))
    title = "{0:d} {1}".format(subject_n, title)

    # display covariance matrix
    pl.figure()
    pl.imshow(cov, interpolation="nearest",
              vmin=-1, vmax=1, cmap=pl.cm.get_cmap("bwr"))
    pl.hlines([(pl.ylim()[0] + pl.ylim()[1]) / 2],
              pl.xlim()[0], pl.xlim()[1])
    pl.vlines([(pl.xlim()[0] + pl.xlim()[1]) / 2],
              pl.ylim()[0], pl.ylim()[1])
    pl.colorbar()
    pl.title(title + " / covariance")

    # display precision matrix
    pl.figure()
    pl.imshow(prec, interpolation="nearest",
              vmin=-span, vmax=span,
              cmap=pl.cm.get_cmap("bwr"))
    pl.hlines([(pl.ylim()[0] + pl.ylim()[1]) / 2],
              pl.xlim()[0], pl.xlim()[1])
    pl.vlines([(pl.xlim()[0] + pl.xlim()[1]) / 2],
              pl.ylim()[0], pl.ylim()[1])
    pl.colorbar()
    pl.title(title + " / precision")


def graph_lasso_covariance(region_ts, subject_n):
    """Compute graph lasso covariance and display it."""
    print("Computing covariance matrices ...")
    estimator = covariance.GraphLassoCV()
    estimator.fit(region_ts)
    print("   selected alpha: {0:.3f}".format(estimator.alpha_))

    plot_matrices(estimator.covariance_, -estimator.precision_,
                  title="Graph Lasso CV ({0:.3f})".format(estimator.alpha_),
                  subject_n=subject_n)


if __name__ == "__main__":
    subject_n = 1
    timeseries, region_img, mask_img = clean_signals(subject_n)
    region_ts = get_region_ts(timeseries, region_img, mask_img)
    graph_lasso_covariance(region_ts, subject_n)
    pl.show()
