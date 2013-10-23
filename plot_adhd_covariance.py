"""
Computation of covariance matrix between brain regions
======================================================

This example shows how to extract signals from regions defined by an atlas,
and to estimate a covariance matrix based on these signals.
"""
 
plotted_subject = 0  # subject to plot


import pylab as pl
import matplotlib
# Copied from matplotlib 1.2.0 for matplotlib 0.99 compatibility.
_bwr_data = ((0.0, 0.0, 1.0), (1.0, 1.0, 1.0), (1.0, 0.0, 0.0))
pl.cm.register_cmap(cmap=matplotlib.colors.LinearSegmentedColormap.from_list(
    "bwr", _bwr_data))


def plot_matrices(cov, prec, title):
    """Plot covariance and precision matrices, for a given processing. """

    prec = prec.copy()  # avoid side effects

    # Display sparsity pattern
    sparsity = prec == 0
    pl.figure()
    pl.imshow(sparsity, interpolation="nearest")
    pl.title("%s / sparsity" % title)

    # Put zeros on the diagonal, for graph clarity.
    size = prec.shape[0]
    prec[range(size), range(size)] = 0
    span = max(abs(prec.min()), abs(prec.max()))

    # Display covariance matrix
    pl.figure()
    pl.imshow(cov, interpolation="nearest",
              vmin=-1, vmax=1, cmap=pl.cm.get_cmap("bwr"))
    pl.colorbar()
    pl.title("%s / covariance" % title)

    # Display precision matrix
    pl.figure()
    pl.imshow(prec, interpolation="nearest",
              vmin=-span, vmax=span,
              cmap=pl.cm.get_cmap("bwr"))
    pl.colorbar()
    pl.title("%s / precision" % title)


print("-- Fetching datasets ...")
import nilearn.datasets
atlas = nilearn.datasets.fetch_msdl_atlas()
dataset = nilearn.datasets.fetch_adhd()

import nilearn.image
import nilearn.input_data

import joblib
mem = joblib.Memory(".")

# Number of subjects to consider for group-sparse covariance
n_subjects = 10
subjects = []

for subject_n in range(n_subjects):
    filename = dataset["func"][subject_n]
    print("Processing file %s" % filename)

    print("-- Computing confounds ...")
    confound_file = dataset["confounds"][subject_n]
    hv_confounds = mem.cache(nilearn.image.high_variance_confounds)(filename)

    print("-- Computing region signals ...")
    masker = nilearn.input_data.NiftiMapsMasker(
        atlas["maps"], resampling_target="maps", detrend=True,
        low_pass=None, high_pass=0.01, t_r=2.5, standardize=True,
        memory=mem, memory_level=1, verbose=1)
    region_ts = masker.fit_transform(filename,
                                     confounds=[hv_confounds, confound_file])
    subjects.append(region_ts)


print("-- Computing group-sparse precision matrices ...")
from nilearn.group_sparse_covariance import GroupSparseCovarianceCV
gsc = GroupSparseCovarianceCV(verbose=2, n_jobs=3)
gsc.fit(subjects)

print("-- Computing graph-lasso precision matrices ...")
from sklearn import covariance
gl = covariance.GraphLassoCV(n_jobs=3)
gl.fit(subjects[plotted_subject])

print("-- Displaying results")
title = "{0:d} GroupSparseCovariance $\\alpha={1:.2e}$".format(plotted_subject,
                                                     gsc.alpha_)
plot_matrices(gsc.covariances_[..., plotted_subject],
              gsc.precisions_[..., plotted_subject], title)

title = "{0:d} GraphLasso $\\alpha={1:.2e}$".format(plotted_subject,
                                                     gl.alpha_)
plot_matrices(gl.covariance_, gl.precision_, title)

pl.show()
