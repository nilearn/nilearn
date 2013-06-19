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

import nisl.datasets
import nisl.image
import nisl.signal
import nisl.io

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


def covariance_matrix(subject_n):
    dataset = nisl.datasets.fetch_adhd()
    filename = dataset["func"][subject_n]
    confound_file = dataset["confounds"][subject_n]

    print("-- Loading raw data ({0:d}) and masking ...".format(subject_n))
    msdl_atlas = nisl.datasets.fetch_msdl_atlas()

    print("-- Computing confounds ...")
    hv_confounds = nisl.image.high_variance_confounds(filename)

    print("-- Computing region signals ...")
    masker = nisl.io.NiftiMapsMasker(msdl_atlas["maps"],
                                     resampling_target="maps",
                                     low_pass=None, high_pass=0.01, t_r=2.5,
                                     standardize=True,
                                     verbose=1)
    region_ts = masker.fit_transform(filename,
                                     confounds=[hv_confounds, confound_file])

    data_img = nibabel.load(filename)
    n_samples = data_img.shape[-1]

    return np.dot(region_ts.T, region_ts), n_samples


if __name__ == "__main__":
    n_subjects = 2
    data = []
    rho = 0.1
#    rho = 20
    mem = joblib.Memory(".")

    for n in xrange(n_subjects):
        data.append(mem.cache(covariance_matrix)(n))

    emp_covs, n_samples = zip(*data)

    # Normalize covariance matrices
    for cov, n in zip(emp_covs, n_samples):
        cov /= n
        np.testing.assert_almost_equal(np.diag(cov), np.ones(cov.shape[0]))

    print("-- Computing covariance matrices ...")
    from nisl.honorio_samaras import honorio_samaras
    est_precs, all_crit = honorio_samaras(emp_covs, rho, n_samples,
                                          normalize_n_samples=True,
                                          n_iter=5,
                                          debug=True, verbose=1)

    for n, value in enumerate(zip(emp_covs,
                                  np.rollaxis(est_precs, -1))):
        emp_cov, prec = value
        plot_matrices(emp_cov, -prec, title="Honorio Samaras", subject_n=n)

    pl.show()
