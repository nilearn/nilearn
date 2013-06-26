"""
Computation of covariance matrix between brain regions
======================================================

This example shows how to extract signals from regions defined by an atlas,
and to estimate a covariance matrix based on these signals.
"""

import numpy as np
import pylab as pl
import matplotlib

#from sklearn import covariance
import joblib

import nibabel

import nilearn.datasets
import nilearn.image
import nilearn.signal
import nilearn.io

# Copied from matplotlib 1.2.0 for matplotlib 0.99 compatibility.
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
#              vmin=-1, vmax=1,
              cmap=pl.cm.get_cmap("bwr"))
    pl.colorbar()
    pl.title(title + " / covariance")

    # Display precision matrix
    pl.figure()
    pl.imshow(prec, interpolation="nearest",
              vmin=-span, vmax=span,
              cmap=pl.cm.get_cmap("bwr"))
    pl.colorbar()
    pl.title(title + " / precision")


def region_signals(subject_n):
    dataset = nilearn.datasets.fetch_adhd()
    filename = dataset["func"][subject_n]
    confound_file = dataset["confounds"][subject_n]

    print("Processing file %s" % filename)

    print("-- Loading raw data ({0:d}) and masking ...".format(subject_n))
    msdl_atlas = nilearn.datasets.fetch_msdl_atlas()
    niimgs = nibabel.load(filename)

    print("-- Computing confounds ...")
    hv_confounds = nilearn.image.high_variance_confounds(niimgs)

    print("-- Computing region signals ...")
    masker = nilearn.io.NiftiMapsMasker(msdl_atlas["maps"],
                                     resampling_target="maps",
                                     low_pass=None, high_pass=0.01, t_r=2.5,
                                     standardize=True,
                                     verbose=1)
    region_ts = masker.fit_transform(niimgs,
                                     confounds=[hv_confounds, confound_file])

    return region_ts


if __name__ == "__main__":
    n_subjects = 40
    tasks = []
    rho = 0.05
    mem = joblib.Memory(".")

    print("-- Computing covariance matrices ...")
    for n in xrange(n_subjects):
        tasks.append(mem.cache(region_signals)(n))

    print("-- Computing precision matrices ...")
    from nilearn.group_sparse_covariance import GroupSparseCovariance
    gsc = GroupSparseCovariance(rho, n_iter=5, verbose=2, return_costs=True)
    gsc.fit(tasks)

    pl.figure()
    pl.plot(gsc.duality_gap_, '+-')
    pl.grid()

    pl.figure()
    pl.plot(gsc.objective_, '+-')
    pl.grid()

    for n, value in enumerate(zip(np.rollaxis(gsc.covariances_, -1),
                                  np.rollaxis(gsc.precisions_, -1))):
        break
        emp_cov, prec = value
        plot_matrices(emp_cov, -prec,
                      title="Group sparse estimator", subject_n=n)
        if n == 2:
            break

    pl.show()
