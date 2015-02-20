"""
Computation of covariance matrix between brain regions
======================================================

This example shows how to extract signals from regions defined by an atlas,
and to estimate a covariance matrix based on these signals.
"""
n_subjects = 10  # Number of subjects to consider for group-sparse covariance
plotted_subject = 0  # subject to plot
n_jobs = 1

import numpy as np

import matplotlib.pyplot as plt
from sklearn.externals.joblib import Memory

import nilearn
from nilearn import plotting, image
from nilearn.input_data.hemisphere_masker import split_bilateral_rois
from nilearn.plotting import cm


def plot_connectome(cov, atlas_maps, **kwargs):
    """Plot connectome given a covariance matrix and atlas maps"""
    imgs = image.iter_img(atlas_maps)
    regions_coords = np.array([
        map(np.asscalar, plotting.find_xyz_cut_coords(img)) for img in imgs])
    np.random.seed(42)
    node_colors = np.random.rand(len(regions_coords) // 2, 3)
    node_colors = np.concatenate([node_colors, node_colors], axis=0)
    node_colors = np.sort(node_colors, axis=0)
    plotting.plot_connectome(cov, regions_coords,
                             nodes_kwargs={'s': 50, 'c': node_colors},
                             **kwargs)


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
adhd_dataset = datasets.fetch_adhd(n_subjects=n_subjects)

# Extracting region signals ###################################################
import nibabel
import nilearn.image
import nilearn.input_data

mem = Memory('nilearn_cache')

maps_img = nibabel.load(msdl_atlas_dataset.maps)
maps_img = split_bilateral_rois(maps_img)

masker = nilearn.input_data.NiftiMapsMasker(
    maps_img, resampling_target="maps", detrend=True,
    low_pass=None, high_pass=0.01, t_r=2.5, standardize=True,
    memory=mem, memory_level=1, verbose=2)
masker.fit()

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
    region_ts = masker.transform(func_filename,
                                 confounds=[hv_confounds, confound_filename])
    subjects.append(region_ts)

# Computing group-sparse precision matrices ###################################
print("-- Computing group-sparse precision matrices ...")
from nilearn.group_sparse_covariance import GroupSparseCovarianceCV
gsc = GroupSparseCovarianceCV(n_jobs=n_jobs, verbose=2)
gsc.fit(subjects)

print("-- Computing graph-lasso precision matrices ...")
from sklearn import covariance
gl = covariance.GraphLassoCV(n_jobs=n_jobs, verbose=2)
gl.fit(subjects[plotted_subject])

# Displaying results ##########################################################
print("-- Displaying results")
title = "{0:d} GroupSparseCovariance $\\alpha={1:.2e}$".format(plotted_subject,
                                                               gsc.alpha_)

plot_connectome(gsc.covariances_[..., plotted_subject],
                maps_img, edges_threshold='80%',
                title=title)
plot_matrices(gsc.covariances_[..., plotted_subject],
              gsc.precisions_[..., plotted_subject], title)

title = "{0:d} GraphLasso $\\alpha={1:.2e}$".format(plotted_subject,
                                                    gl.alpha_)
plot_connectome(gl.covariance_,
                maps_img, edges_threshold='80%',
                title=title)
plot_matrices(gl.covariance_, gl.precision_, title)

plt.show()

imgs = list(image.iter_img(maps_img))
coords = np.array([
    map(np.asscalar, plotting.find_xyz_cut_coords(img)) for img in imgs])

cov = gsc.covariances_[..., plotted_subject]

from nilearn import plotting
np.random.seed(42)
node_colors = np.random.rand(maps_img.shape[-1], 3)
display = plotting.plot_connectome(cov, coords,
                                   edges_threshold=0.38,
                                   title='threshold=0.38',
                                   nodes_kwargs={
                                       's': 50, 'c': node_colors})

display = plotting.plot_connectome(cov, coords,
                                   edges_threshold='70%',
                                   title='threshold=70%',
                                   nodes_kwargs={
                                       's': 50, 'c': node_colors})
