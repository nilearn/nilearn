"""
Computation of covariance matrix between brain regions
======================================================

This example shows how to extract signals from regions defined by an atlas,
and to estimate a covariance matrix based on these signals.
"""
 
plotted_subject = 0  # subject to plot


import matplotlib.pyplot as plt
from nilearn.plotting import cm


def plot_matrices(cov, prec, title):
    """Plot covariance and precision matrices, for a given processing. """

    # Compute sparsity pattern
    sparsity = (prec == 0)
    
    prec = prec.copy()  # avoid side effects


    # Put zeros on the diagonal, for graph clarity.
    size = prec.shape[0]
    prec[range(size), range(size)] = 0
    span = max(abs(prec.min()), abs(prec.max()))

    # Display covariance matrix
    plt.figure()
    plt.imshow(cov, interpolation="nearest",
               vmin=-1, vmax=1, cmap=cm.bwr)
    plt.colorbar()
    plt.title("%s / covariance" % title)
    
    # Display sparsity pattern
    plt.figure()
    plt.imshow(sparsity, interpolation="nearest")
    plt.title("%s / sparsity" % title)

    # Display precision matrix
    plt.figure()
    plt.imshow(prec, interpolation="nearest",
               vmin=-span, vmax=span,
               cmap=cm.bwr)
    plt.colorbar()
    plt.title("%s / precision" % title)


# Fetching datasets ###########################################################
print("-- Fetching datasets ...")
from nilearn import datasets
msdl_atlas_dataset = datasets.fetch_msdl_atlas()
adhd_dataset = datasets.fetch_adhd()

# Extracting region signals ###################################################
import nilearn.image
import nilearn.input_data

from sklearn.externals.joblib import Memory
mem = Memory(".")

# Number of subjects to consider for group-sparse covariance
n_subjects = 10
subjects = []

func_filenames = adhd_dataset.func
confound_filenames = adhd_dataset.confounds
for func_filename, confound_filename in zip(func_filenames,
                                            confound_filenames):
    print("Processing file %s" % func_filename)

    print("-- Computing confounds ...")
    hv_confounds = mem.cache(nilearn.image.high_variance_confounds)(
        func_filename)

    print("-- Computing region signals ...")
    masker = nilearn.input_data.NiftiMapsMasker(
        msdl_atlas_dataset.maps, resampling_target="maps", detrend=True,
        low_pass=None, high_pass=0.01, t_r=2.5, standardize=True,
        memory=mem, memory_level=1, verbose=1)
    region_ts = masker.fit_transform(func_filename,
                                     confounds=[hv_confounds,
                                                confound_filename])
    subjects.append(region_ts)

# Computing group-sparse precision matrices ###################################
print("-- Computing group-sparse precision matrices ...")
from nilearn.group_sparse_covariance import GroupSparseCovarianceCV
gsc = GroupSparseCovarianceCV(verbose=2, n_jobs=3)
gsc.fit(subjects)

print("-- Computing graph-lasso precision matrices ...")
from sklearn import covariance
gl = covariance.GraphLassoCV(n_jobs=3)
gl.fit(subjects[plotted_subject])

# Displaying results ##########################################################
print("-- Displaying results")
title = "{0:d} GroupSparseCovariance $\\alpha={1:.2e}$".format(plotted_subject,
                                                     gsc.alpha_)
plot_matrices(gsc.covariances_[..., plotted_subject],
              gsc.precisions_[..., plotted_subject], title)

title = "{0:d} GraphLasso $\\alpha={1:.2e}$".format(plotted_subject,
                                                     gl.alpha_)
plot_matrices(gl.covariance_, gl.precision_, title)

plt.show()
