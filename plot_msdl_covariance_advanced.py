"""
Computation of covariance matrix between brain regions (advanced)
=================================================================

The following things are performed:

- parcellation loading, and signals extraction
- improvement of fMRI data SNR (confound removal, filtering, etc.)
- covariance/precision matrices computation
- display of matrices

This script is intended for advanced users only, because it makes use of
low-level functions.
"""

import numpy as np
import pylab as pl
import matplotlib

from sklearn import covariance

import nisl.datasets
import nisl.image
import nisl.signal
import nisl.io

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

    # Display covariance matrix
    pl.figure()
    pl.imshow(cov, interpolation="nearest",
              vmin=-1, vmax=1, cmap=pl.cm.get_cmap("bwr"))
    pl.colorbar()
    pl.title(title + " / covariance")

    # Display precision matrix
    pl.figure()
    pl.imshow(prec, interpolation="nearest",
              vmin=-span, vmax=span,
              cmap=pl.cm.get_cmap("bwr"))
    pl.colorbar()
    pl.title(title + " / precision")

subject_n = 1

dataset = nisl.datasets.fetch_adhd()
filename = dataset["func"][subject_n]
confound_file = dataset["confounds"][subject_n]

print("-- Loading raw data ({0:d}) and masking ...".format(subject_n))
msdl_atlas = nisl.datasets.fetch_msdl_atlas()

print("-- Computing confounds ...")
# Compcor on full image
hv_confounds = nisl.image.high_variance_confounds(filename)
mvt_confounds = np.loadtxt(confound_file, skiprows=1)
confounds = np.hstack((hv_confounds, mvt_confounds))

print("-- Computing region signals ...")
masker = nisl.io.NiftiMapsMasker(msdl_atlas["maps"], resampling_target="maps",
                                 low_pass=None, high_pass=0.01, t_r=2.5)
region_ts = masker.fit_transform(filename, confounds=confounds)

print("-- Computing covariance matrices ...")
estimator = covariance.GraphLassoCV()
estimator.fit(region_ts)

plot_matrices(estimator.covariance_, -estimator.precision_,
              title="Graph Lasso CV ({0:.3f})".format(estimator.alpha_),
              subject_n=subject_n)
pl.show()


